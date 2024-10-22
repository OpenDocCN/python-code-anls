# `.\diffusers\models\unets\unet_2d_blocks.py`

```py
# 版权所有 2024 HuggingFace 团队，保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，否则根据许可证分发的软件按“现状”基础分发，
# 不提供任何形式的保证或条件，无论是明示还是暗示。
# 请参阅许可证以了解特定语言的权限和
# 限制。
from typing import Any, Dict, Optional, Tuple, Union  # 导入类型注解相关的模块

import numpy as np  # 导入 NumPy 库用于数值计算
import torch  # 导入 PyTorch 库用于深度学习
import torch.nn.functional as F  # 导入 PyTorch 的功能性 API
from torch import nn  # 导入 PyTorch 的神经网络模块

from ...utils import deprecate, is_torch_version, logging  # 从工具模块导入日志和版本检测功能
from ...utils.torch_utils import apply_freeu  # 从 PyTorch 工具模块导入 apply_freeu 函数
from ..activations import get_activation  # 从激活函数模块导入 get_activation 函数
from ..attention_processor import Attention, AttnAddedKVProcessor, AttnAddedKVProcessor2_0  # 导入注意力处理器相关的类
from ..normalization import AdaGroupNorm  # 从归一化模块导入 AdaGroupNorm 类
from ..resnet import (  # 从 ResNet 模块导入多个下采样和上采样类
    Downsample2D,
    FirDownsample2D,
    FirUpsample2D,
    KDownsample2D,
    KUpsample2D,
    ResnetBlock2D,
    ResnetBlockCondNorm2D,
    Upsample2D,
)
from ..transformers.dual_transformer_2d import DualTransformer2DModel  # 导入双重变换器模型
from ..transformers.transformer_2d import Transformer2DModel  # 导入二维变换器模型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例


def get_down_block(  # 定义获取下采样块的函数
    down_block_type: str,  # 下采样块的类型
    num_layers: int,  # 下采样层的数量
    in_channels: int,  # 输入通道数
    out_channels: int,  # 输出通道数
    temb_channels: int,  # 时间嵌入通道数
    add_downsample: bool,  # 是否添加下采样标志
    resnet_eps: float,  # ResNet 中的 epsilon 参数
    resnet_act_fn: str,  # ResNet 使用的激活函数
    transformer_layers_per_block: int = 1,  # 每个块中的变换器层数，默认为 1
    num_attention_heads: Optional[int] = None,  # 注意力头的数量，默认为 None
    resnet_groups: Optional[int] = None,  # ResNet 中的组数，默认为 None
    cross_attention_dim: Optional[int] = None,  # 交叉注意力维度，默认为 None
    downsample_padding: Optional[int] = None,  # 下采样填充参数，默认为 None
    dual_cross_attention: bool = False,  # 是否使用双重交叉注意力标志
    use_linear_projection: bool = False,  # 是否使用线性投影标志
    only_cross_attention: bool = False,  # 是否仅使用交叉注意力标志
    upcast_attention: bool = False,  # 是否上升注意力标志
    resnet_time_scale_shift: str = "default",  # ResNet 时间缩放移位，默认为“default”
    attention_type: str = "default",  # 注意力类型，默认为“default”
    resnet_skip_time_act: bool = False,  # ResNet 跳过时间激活标志
    resnet_out_scale_factor: float = 1.0,  # ResNet 输出缩放因子，默认为 1.0
    cross_attention_norm: Optional[str] = None,  # 交叉注意力归一化，默认为 None
    attention_head_dim: Optional[int] = None,  # 注意力头维度，默认为 None
    downsample_type: Optional[str] = None,  # 下采样类型，默认为 None
    dropout: float = 0.0,  # dropout 比例，默认为 0.0
):
    # 如果没有定义注意力头维度，默认设置为头的数量
    if attention_head_dim is None:
        logger.warning(  # 记录警告信息
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."  # 提醒用户使用默认的注意力头维度
        )
        attention_head_dim = num_attention_heads  # 将注意力头维度设置为头的数量

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type  # 处理下采样块类型的字符串
    # 检查下行块的类型是否为 "DownBlock2D"
        if down_block_type == "DownBlock2D":
            # 返回 DownBlock2D 实例，传入相关参数
            return DownBlock2D(
                # 传入层数
                num_layers=num_layers,
                # 输入通道数
                in_channels=in_channels,
                # 输出通道数
                out_channels=out_channels,
                # 时间嵌入通道数
                temb_channels=temb_channels,
                # dropout 比率
                dropout=dropout,
                # 是否添加下采样
                add_downsample=add_downsample,
                # ResNet 的 epsilon 值
                resnet_eps=resnet_eps,
                # ResNet 的激活函数
                resnet_act_fn=resnet_act_fn,
                # ResNet 的分组数
                resnet_groups=resnet_groups,
                # 下采样的填充
                downsample_padding=downsample_padding,
                # ResNet 的时间尺度偏移
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
        # 检查下行块的类型是否为 "ResnetDownsampleBlock2D"
        elif down_block_type == "ResnetDownsampleBlock2D":
            # 返回 ResnetDownsampleBlock2D 实例，传入相关参数
            return ResnetDownsampleBlock2D(
                # 传入层数
                num_layers=num_layers,
                # 输入通道数
                in_channels=in_channels,
                # 输出通道数
                out_channels=out_channels,
                # 时间嵌入通道数
                temb_channels=temb_channels,
                # dropout 比率
                dropout=dropout,
                # 是否添加下采样
                add_downsample=add_downsample,
                # ResNet 的 epsilon 值
                resnet_eps=resnet_eps,
                # ResNet 的激活函数
                resnet_act_fn=resnet_act_fn,
                # ResNet 的分组数
                resnet_groups=resnet_groups,
                # ResNet 的时间尺度偏移
                resnet_time_scale_shift=resnet_time_scale_shift,
                # ResNet 的时间激活跳过标志
                skip_time_act=resnet_skip_time_act,
                # ResNet 的输出缩放因子
                output_scale_factor=resnet_out_scale_factor,
            )
        # 检查下行块的类型是否为 "AttnDownBlock2D"
        elif down_block_type == "AttnDownBlock2D":
            # 如果不添加下采样，则将下采样类型设为 None
            if add_downsample is False:
                downsample_type = None
            else:
                # 如果添加下采样，则默认下采样类型为 'conv'
                downsample_type = downsample_type or "conv"  # default to 'conv'
            # 返回 AttnDownBlock2D 实例，传入相关参数
            return AttnDownBlock2D(
                # 传入层数
                num_layers=num_layers,
                # 输入通道数
                in_channels=in_channels,
                # 输出通道数
                out_channels=out_channels,
                # 时间嵌入通道数
                temb_channels=temb_channels,
                # dropout 比率
                dropout=dropout,
                # ResNet 的 epsilon 值
                resnet_eps=resnet_eps,
                # ResNet 的激活函数
                resnet_act_fn=resnet_act_fn,
                # ResNet 的分组数
                resnet_groups=resnet_groups,
                # 下采样的填充
                downsample_padding=downsample_padding,
                # 注意力头的维度
                attention_head_dim=attention_head_dim,
                # ResNet 的时间尺度偏移
                resnet_time_scale_shift=resnet_time_scale_shift,
                # 下采样类型
                downsample_type=downsample_type,
            )
    # 检查下行块类型是否为 CrossAttnDownBlock2D
    elif down_block_type == "CrossAttnDownBlock2D":
        # 如果 cross_attention_dim 未指定，则抛出错误
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        # 返回 CrossAttnDownBlock2D 实例，使用提供的参数进行初始化
        return CrossAttnDownBlock2D(
            # 设置层的数量
            num_layers=num_layers,
            # 每个块的变换层数量
            transformer_layers_per_block=transformer_layers_per_block,
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # 时间嵌入通道数
            temb_channels=temb_channels,
            # dropout 比例
            dropout=dropout,
            # 是否添加下采样
            add_downsample=add_downsample,
            # ResNet 中的 epsilon 值
            resnet_eps=resnet_eps,
            # ResNet 激活函数
            resnet_act_fn=resnet_act_fn,
            # ResNet 分组数
            resnet_groups=resnet_groups,
            # 下采样填充
            downsample_padding=downsample_padding,
            # 跨注意力维度
            cross_attention_dim=cross_attention_dim,
            # 注意力头数
            num_attention_heads=num_attention_heads,
            # 是否使用双向跨注意力
            dual_cross_attention=dual_cross_attention,
            # 是否使用线性投影
            use_linear_projection=use_linear_projection,
            # 是否仅使用跨注意力
            only_cross_attention=only_cross_attention,
            # 是否上溯注意力
            upcast_attention=upcast_attention,
            # ResNet 时间尺度偏移
            resnet_time_scale_shift=resnet_time_scale_shift,
            # 注意力类型
            attention_type=attention_type,
        )
    # 检查下行块类型是否为 SimpleCrossAttnDownBlock2D
    elif down_block_type == "SimpleCrossAttnDownBlock2D":
        # 如果 cross_attention_dim 未指定，则抛出错误
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for SimpleCrossAttnDownBlock2D")
        # 返回 SimpleCrossAttnDownBlock2D 实例，使用提供的参数进行初始化
        return SimpleCrossAttnDownBlock2D(
            # 设置层的数量
            num_layers=num_layers,
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # 时间嵌入通道数
            temb_channels=temb_channels,
            # dropout 比例
            dropout=dropout,
            # 是否添加下采样
            add_downsample=add_downsample,
            # ResNet 中的 epsilon 值
            resnet_eps=resnet_eps,
            # ResNet 激活函数
            resnet_act_fn=resnet_act_fn,
            # ResNet 分组数
            resnet_groups=resnet_groups,
            # 跨注意力维度
            cross_attention_dim=cross_attention_dim,
            # 注意力头的维度
            attention_head_dim=attention_head_dim,
            # ResNet 时间尺度偏移
            resnet_time_scale_shift=resnet_time_scale_shift,
            # 是否跳过时间激活
            skip_time_act=resnet_skip_time_act,
            # 输出缩放因子
            output_scale_factor=resnet_out_scale_factor,
            # 是否仅使用跨注意力
            only_cross_attention=only_cross_attention,
            # 跨注意力规范化
            cross_attention_norm=cross_attention_norm,
        )
    # 检查下行块类型是否为 SkipDownBlock2D
    elif down_block_type == "SkipDownBlock2D":
        # 返回 SkipDownBlock2D 实例，使用提供的参数进行初始化
        return SkipDownBlock2D(
            # 设置层的数量
            num_layers=num_layers,
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # 时间嵌入通道数
            temb_channels=temb_channels,
            # dropout 比例
            dropout=dropout,
            # 是否添加下采样
            add_downsample=add_downsample,
            # ResNet 中的 epsilon 值
            resnet_eps=resnet_eps,
            # ResNet 激活函数
            resnet_act_fn=resnet_act_fn,
            # 下采样填充
            downsample_padding=downsample_padding,
            # ResNet 时间尺度偏移
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    # 检查下采样块的类型是否为 "AttnSkipDownBlock2D"
    elif down_block_type == "AttnSkipDownBlock2D":
        # 返回一个 AttnSkipDownBlock2D 对象，初始化参数传入
        return AttnSkipDownBlock2D(
            # 设置层数
            num_layers=num_layers,
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # 时间嵌入通道数
            temb_channels=temb_channels,
            # dropout 率
            dropout=dropout,
            # 是否添加下采样
            add_downsample=add_downsample,
            # ResNet 的 epsilon 值
            resnet_eps=resnet_eps,
            # ResNet 的激活函数
            resnet_act_fn=resnet_act_fn,
            # 注意力头的维度
            attention_head_dim=attention_head_dim,
            # ResNet 时间缩放偏移
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    # 检查下采样块的类型是否为 "DownEncoderBlock2D"
    elif down_block_type == "DownEncoderBlock2D":
        # 返回一个 DownEncoderBlock2D 对象，初始化参数传入
        return DownEncoderBlock2D(
            # 设置层数
            num_layers=num_layers,
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # dropout 率
            dropout=dropout,
            # 是否添加下采样
            add_downsample=add_downsample,
            # ResNet 的 epsilon 值
            resnet_eps=resnet_eps,
            # ResNet 的激活函数
            resnet_act_fn=resnet_act_fn,
            # ResNet 的组数
            resnet_groups=resnet_groups,
            # 下采样填充
            downsample_padding=downsample_padding,
            # ResNet 时间缩放偏移
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    # 检查下采样块的类型是否为 "AttnDownEncoderBlock2D"
    elif down_block_type == "AttnDownEncoderBlock2D":
        # 返回一个 AttnDownEncoderBlock2D 对象，初始化参数传入
        return AttnDownEncoderBlock2D(
            # 设置层数
            num_layers=num_layers,
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # dropout 率
            dropout=dropout,
            # 是否添加下采样
            add_downsample=add_downsample,
            # ResNet 的 epsilon 值
            resnet_eps=resnet_eps,
            # ResNet 的激活函数
            resnet_act_fn=resnet_act_fn,
            # ResNet 的组数
            resnet_groups=resnet_groups,
            # 下采样填充
            downsample_padding=downsample_padding,
            # 注意力头的维度
            attention_head_dim=attention_head_dim,
            # ResNet 时间缩放偏移
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    # 检查下采样块的类型是否为 "KDownBlock2D"
    elif down_block_type == "KDownBlock2D":
        # 返回一个 KDownBlock2D 对象，初始化参数传入
        return KDownBlock2D(
            # 设置层数
            num_layers=num_layers,
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # 时间嵌入通道数
            temb_channels=temb_channels,
            # dropout 率
            dropout=dropout,
            # 是否添加下采样
            add_downsample=add_downsample,
            # ResNet 的 epsilon 值
            resnet_eps=resnet_eps,
            # ResNet 的激活函数
            resnet_act_fn=resnet_act_fn,
        )
    # 检查下采样块的类型是否为 "KCrossAttnDownBlock2D"
    elif down_block_type == "KCrossAttnDownBlock2D":
        # 返回一个 KCrossAttnDownBlock2D 对象，初始化参数传入
        return KCrossAttnDownBlock2D(
            # 设置层数
            num_layers=num_layers,
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # 时间嵌入通道数
            temb_channels=temb_channels,
            # dropout 率
            dropout=dropout,
            # 是否添加下采样
            add_downsample=add_downsample,
            # ResNet 的 epsilon 值
            resnet_eps=resnet_eps,
            # ResNet 的激活函数
            resnet_act_fn=resnet_act_fn,
            # 跨注意力的维度
            cross_attention_dim=cross_attention_dim,
            # 注意力头的维度
            attention_head_dim=attention_head_dim,
            # 是否添加自注意力
            add_self_attention=True if not add_downsample else False,
        )
    # 如果下采样块类型不匹配，则抛出异常
    raise ValueError(f"{down_block_type} does not exist.")
# 根据给定参数生成中间块（mid block）
def get_mid_block(
    # 中间块的类型
    mid_block_type: str,
    # 嵌入通道数
    temb_channels: int,
    # 输入通道数
    in_channels: int,
    # ResNet 的 epsilon 值
    resnet_eps: float,
    # ResNet 的激活函数类型
    resnet_act_fn: str,
    # ResNet 的组数
    resnet_groups: int,
    # 输出缩放因子，默认为 1.0
    output_scale_factor: float = 1.0,
    # 每个块的变换层数，默认为 1
    transformer_layers_per_block: int = 1,
    # 注意力头的数量，默认为 None
    num_attention_heads: Optional[int] = None,
    # 跨注意力的维度，默认为 None
    cross_attention_dim: Optional[int] = None,
    # 是否使用双重跨注意力，默认为 False
    dual_cross_attention: bool = False,
    # 是否使用线性投影，默认为 False
    use_linear_projection: bool = False,
    # 是否仅使用跨注意力作为中间块，默认为 False
    mid_block_only_cross_attention: bool = False,
    # 是否提升注意力精度，默认为 False
    upcast_attention: bool = False,
    # ResNet 的时间缩放偏移，默认为 "default"
    resnet_time_scale_shift: str = "default",
    # 注意力类型，默认为 "default"
    attention_type: str = "default",
    # ResNet 是否跳过时间激活，默认为 False
    resnet_skip_time_act: bool = False,
    # 跨注意力的归一化类型，默认为 None
    cross_attention_norm: Optional[str] = None,
    # 注意力头的维度，默认为 1
    attention_head_dim: Optional[int] = 1,
    # dropout 概率，默认为 0.0
    dropout: float = 0.0,
):
    # 根据中间块的类型生成对应的对象
    if mid_block_type == "UNetMidBlock2DCrossAttn":
        # 创建 UNet 的 2D 跨注意力中间块
        return UNetMidBlock2DCrossAttn(
            # 设置变换层数
            transformer_layers_per_block=transformer_layers_per_block,
            # 设置输入通道数
            in_channels=in_channels,
            # 设置嵌入通道数
            temb_channels=temb_channels,
            # 设置 dropout 概率
            dropout=dropout,
            # 设置 ResNet epsilon 值
            resnet_eps=resnet_eps,
            # 设置 ResNet 激活函数
            resnet_act_fn=resnet_act_fn,
            # 设置输出缩放因子
            output_scale_factor=output_scale_factor,
            # 设置时间缩放偏移
            resnet_time_scale_shift=resnet_time_scale_shift,
            # 设置跨注意力维度
            cross_attention_dim=cross_attention_dim,
            # 设置注意力头数量
            num_attention_heads=num_attention_heads,
            # 设置 ResNet 组数
            resnet_groups=resnet_groups,
            # 设置是否使用双重跨注意力
            dual_cross_attention=dual_cross_attention,
            # 设置是否使用线性投影
            use_linear_projection=use_linear_projection,
            # 设置是否提升注意力精度
            upcast_attention=upcast_attention,
            # 设置注意力类型
            attention_type=attention_type,
        )
    # 检查是否为简单跨注意力中间块
    elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
        # 创建 UNet 的 2D 简单跨注意力中间块
        return UNetMidBlock2DSimpleCrossAttn(
            # 设置输入通道数
            in_channels=in_channels,
            # 设置嵌入通道数
            temb_channels=temb_channels,
            # 设置 dropout 概率
            dropout=dropout,
            # 设置 ResNet epsilon 值
            resnet_eps=resnet_eps,
            # 设置 ResNet 激活函数
            resnet_act_fn=resnet_act_fn,
            # 设置输出缩放因子
            output_scale_factor=output_scale_factor,
            # 设置跨注意力维度
            cross_attention_dim=cross_attention_dim,
            # 设置注意力头的维度
            attention_head_dim=attention_head_dim,
            # 设置 ResNet 组数
            resnet_groups=resnet_groups,
            # 设置时间缩放偏移
            resnet_time_scale_shift=resnet_time_scale_shift,
            # 设置是否跳过时间激活
            skip_time_act=resnet_skip_time_act,
            # 设置是否仅使用跨注意力
            only_cross_attention=mid_block_only_cross_attention,
            # 设置跨注意力的归一化类型
            cross_attention_norm=cross_attention_norm,
        )
    # 检查是否为标准的 2D 中间块
    elif mid_block_type == "UNetMidBlock2D":
        # 创建 UNet 的 2D 中间块
        return UNetMidBlock2D(
            # 设置输入通道数
            in_channels=in_channels,
            # 设置嵌入通道数
            temb_channels=temb_channels,
            # 设置 dropout 概率
            dropout=dropout,
            # 设置层数为 0
            num_layers=0,
            # 设置 ResNet epsilon 值
            resnet_eps=resnet_eps,
            # 设置 ResNet 激活函数
            resnet_act_fn=resnet_act_fn,
            # 设置输出缩放因子
            output_scale_factor=output_scale_factor,
            # 设置 ResNet 组数
            resnet_groups=resnet_groups,
            # 设置时间缩放偏移
            resnet_time_scale_shift=resnet_time_scale_shift,
            # 不添加注意力
            add_attention=False,
        )
    # 检查中间块类型是否为 None
    elif mid_block_type is None:
        # 返回 None
        return None
    # 抛出未知类型的异常
    else:
        raise ValueError(f"unknown mid_block_type : {mid_block_type}")
    # 输出通道的数量
        out_channels: int,
        # 前一层输出通道的数量
        prev_output_channel: int,
        # 嵌入层通道的数量
        temb_channels: int,
        # 是否添加上采样层
        add_upsample: bool,
        # ResNet 中的 epsilon 值，用于数值稳定性
        resnet_eps: float,
        # ResNet 中的激活函数类型
        resnet_act_fn: str,
        # 分辨率索引，默认为 None
        resolution_idx: Optional[int] = None,
        # 每个块中的变换层数量
        transformer_layers_per_block: int = 1,
        # 注意力头的数量，默认为 None
        num_attention_heads: Optional[int] = None,
        # ResNet 中的组数量，默认为 None
        resnet_groups: Optional[int] = None,
        # 交叉注意力的维度，默认为 None
        cross_attention_dim: Optional[int] = None,
        # 是否使用双重交叉注意力
        dual_cross_attention: bool = False,
        # 是否使用线性投影
        use_linear_projection: bool = False,
        # 是否仅使用交叉注意力
        only_cross_attention: bool = False,
        # 是否上采样时提高注意力精度
        upcast_attention: bool = False,
        # ResNet 时间缩放偏移的类型，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # 注意力类型，默认为 "default"
        attention_type: str = "default",
        # ResNet 中跳过时间激活的标志
        resnet_skip_time_act: bool = False,
        # ResNet 输出缩放因子，默认为 1.0
        resnet_out_scale_factor: float = 1.0,
        # 交叉注意力的归一化方式，默认为 None
        cross_attention_norm: Optional[str] = None,
        # 注意力头的维度，默认为 None
        attention_head_dim: Optional[int] = None,
        # 上采样类型，默认为 None
        upsample_type: Optional[str] = None,
        # 丢弃率，默认为 0.0
        dropout: float = 0.0,
) -> nn.Module:  # 指定该函数的返回类型为 nn.Module
    # 如果未定义注意力头的维度，默认设置为注意力头的数量
    if attention_head_dim is None:
        logger.warning(  # 记录警告信息
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."  # 提示用户提供 attention_head_dim
        )
        attention_head_dim = num_attention_heads  # 将 attention_head_dim 设置为 num_attention_heads

    # 如果 up_block_type 以 "UNetRes" 开头，则去掉前缀
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    # 检查 up_block_type 是否为 "UpBlock2D"
    if up_block_type == "UpBlock2D":
        return UpBlock2D(  # 返回 UpBlock2D 对象
            num_layers=num_layers,  # 设置网络层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            prev_output_channel=prev_output_channel,  # 设置前一个输出通道
            temb_channels=temb_channels,  # 设置时间嵌入通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 参数
            add_upsample=add_upsample,  # 设置是否添加上采样
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
            resnet_groups=resnet_groups,  # 设置 ResNet 的组数
            resnet_time_scale_shift=resnet_time_scale_shift,  # 设置 ResNet 的时间缩放偏移
        )
    # 检查 up_block_type 是否为 "ResnetUpsampleBlock2D"
    elif up_block_type == "ResnetUpsampleBlock2D":
        return ResnetUpsampleBlock2D(  # 返回 ResnetUpsampleBlock2D 对象
            num_layers=num_layers,  # 设置网络层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            prev_output_channel=prev_output_channel,  # 设置前一个输出通道
            temb_channels=temb_channels,  # 设置时间嵌入通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 参数
            add_upsample=add_upsample,  # 设置是否添加上采样
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
            resnet_groups=resnet_groups,  # 设置 ResNet 的组数
            resnet_time_scale_shift=resnet_time_scale_shift,  # 设置 ResNet 的时间缩放偏移
            skip_time_act=resnet_skip_time_act,  # 设置是否跳过时间激活
            output_scale_factor=resnet_out_scale_factor,  # 设置输出缩放因子
        )
    # 检查 up_block_type 是否为 "CrossAttnUpBlock2D"
    elif up_block_type == "CrossAttnUpBlock2D":
        # 如果未定义交叉注意力维度，则抛出异常
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")  # 抛出值错误
        return CrossAttnUpBlock2D(  # 返回 CrossAttnUpBlock2D 对象
            num_layers=num_layers,  # 设置网络层数
            transformer_layers_per_block=transformer_layers_per_block,  # 设置每个块的变换层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            prev_output_channel=prev_output_channel,  # 设置前一个输出通道
            temb_channels=temb_channels,  # 设置时间嵌入通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 参数
            add_upsample=add_upsample,  # 设置是否添加上采样
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
            resnet_groups=resnet_groups,  # 设置 ResNet 的组数
            cross_attention_dim=cross_attention_dim,  # 设置交叉注意力维度
            num_attention_heads=num_attention_heads,  # 设置注意力头的数量
            dual_cross_attention=dual_cross_attention,  # 设置双重交叉注意力
            use_linear_projection=use_linear_projection,  # 设置是否使用线性投影
            only_cross_attention=only_cross_attention,  # 设置是否仅使用交叉注意力
            upcast_attention=upcast_attention,  # 设置是否提升注意力
            resnet_time_scale_shift=resnet_time_scale_shift,  # 设置 ResNet 的时间缩放偏移
            attention_type=attention_type,  # 设置注意力类型
        )
    # 检查上采样块的类型是否为 SimpleCrossAttnUpBlock2D
    elif up_block_type == "SimpleCrossAttnUpBlock2D":
        # 如果未指定 cross_attention_dim，则抛出值错误
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for SimpleCrossAttnUpBlock2D")
        # 返回 SimpleCrossAttnUpBlock2D 实例，使用相关参数初始化
        return SimpleCrossAttnUpBlock2D(
            num_layers=num_layers,  # 设置层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            prev_output_channel=prev_output_channel,  # 设置前一层的输出通道数
            temb_channels=temb_channels,  # 设置时间嵌入通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 概率
            add_upsample=add_upsample,  # 设置是否添加上采样
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
            resnet_groups=resnet_groups,  # 设置 ResNet 的组数
            cross_attention_dim=cross_attention_dim,  # 设置交叉注意力维度
            attention_head_dim=attention_head_dim,  # 设置注意力头维度
            resnet_time_scale_shift=resnet_time_scale_shift,  # 设置 ResNet 的时间缩放偏移
            skip_time_act=resnet_skip_time_act,  # 设置是否跳过时间激活
            output_scale_factor=resnet_out_scale_factor,  # 设置输出缩放因子
            only_cross_attention=only_cross_attention,  # 设置是否仅使用交叉注意力
            cross_attention_norm=cross_attention_norm,  # 设置交叉注意力的归一化方式
        )
    # 检查上采样块的类型是否为 AttnUpBlock2D
    elif up_block_type == "AttnUpBlock2D":
        # 如果未添加上采样，则将上采样类型设为 None
        if add_upsample is False:
            upsample_type = None
        else:
            # 默认将上采样类型设为 'conv'
            upsample_type = upsample_type or "conv"

        # 返回 AttnUpBlock2D 实例，使用相关参数初始化
        return AttnUpBlock2D(
            num_layers=num_layers,  # 设置层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            prev_output_channel=prev_output_channel,  # 设置前一层的输出通道数
            temb_channels=temb_channels,  # 设置时间嵌入通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 概率
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
            resnet_groups=resnet_groups,  # 设置 ResNet 的组数
            attention_head_dim=attention_head_dim,  # 设置注意力头维度
            resnet_time_scale_shift=resnet_time_scale_shift,  # 设置 ResNet 的时间缩放偏移
            upsample_type=upsample_type,  # 设置上采样类型
        )
    # 检查上采样块的类型是否为 SkipUpBlock2D
    elif up_block_type == "SkipUpBlock2D":
        # 返回 SkipUpBlock2D 实例，使用相关参数初始化
        return SkipUpBlock2D(
            num_layers=num_layers,  # 设置层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            prev_output_channel=prev_output_channel,  # 设置前一层的输出通道数
            temb_channels=temb_channels,  # 设置时间嵌入通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 概率
            add_upsample=add_upsample,  # 设置是否添加上采样
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
            resnet_time_scale_shift=resnet_time_scale_shift,  # 设置 ResNet 的时间缩放偏移
        )
    # 检查上采样块的类型是否为 AttnSkipUpBlock2D
    elif up_block_type == "AttnSkipUpBlock2D":
        # 返回 AttnSkipUpBlock2D 实例，使用相关参数初始化
        return AttnSkipUpBlock2D(
            num_layers=num_layers,  # 设置层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            prev_output_channel=prev_output_channel,  # 设置前一层的输出通道数
            temb_channels=temb_channels,  # 设置时间嵌入通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 概率
            add_upsample=add_upsample,  # 设置是否添加上采样
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
            attention_head_dim=attention_head_dim,  # 设置注意力头维度
            resnet_time_scale_shift=resnet_time_scale_shift,  # 设置 ResNet 的时间缩放偏移
        )
    # 检查上采样块类型是否为 UpDecoderBlock2D
    elif up_block_type == "UpDecoderBlock2D":
        # 返回 UpDecoderBlock2D 的实例，传入相应参数
        return UpDecoderBlock2D(
            num_layers=num_layers,  # 设置层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 比例
            add_upsample=add_upsample,  # 设置是否添加上采样
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon 值
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
            resnet_groups=resnet_groups,  # 设置 ResNet 的组数
            resnet_time_scale_shift=resnet_time_scale_shift,  # 设置时间尺度偏移
            temb_channels=temb_channels,  # 设置时间嵌入通道数
        )
    # 检查上采样块类型是否为 AttnUpDecoderBlock2D
    elif up_block_type == "AttnUpDecoderBlock2D":
        # 返回 AttnUpDecoderBlock2D 的实例，传入相应参数
        return AttnUpDecoderBlock2D(
            num_layers=num_layers,  # 设置层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 比例
            add_upsample=add_upsample,  # 设置是否添加上采样
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon 值
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
            resnet_groups=resnet_groups,  # 设置 ResNet 的组数
            attention_head_dim=attention_head_dim,  # 设置注意力头维度
            resnet_time_scale_shift=resnet_time_scale_shift,  # 设置时间尺度偏移
            temb_channels=temb_channels,  # 设置时间嵌入通道数
        )
    # 检查上采样块类型是否为 KUpBlock2D
    elif up_block_type == "KUpBlock2D":
        # 返回 KUpBlock2D 的实例，传入相应参数
        return KUpBlock2D(
            num_layers=num_layers,  # 设置层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            temb_channels=temb_channels,  # 设置时间嵌入通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 比例
            add_upsample=add_upsample,  # 设置是否添加上采样
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon 值
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
        )
    # 检查上采样块类型是否为 KCrossAttnUpBlock2D
    elif up_block_type == "KCrossAttnUpBlock2D":
        # 返回 KCrossAttnUpBlock2D 的实例，传入相应参数
        return KCrossAttnUpBlock2D(
            num_layers=num_layers,  # 设置层数
            in_channels=in_channels,  # 设置输入通道数
            out_channels=out_channels,  # 设置输出通道数
            temb_channels=temb_channels,  # 设置时间嵌入通道数
            resolution_idx=resolution_idx,  # 设置分辨率索引
            dropout=dropout,  # 设置 dropout 比例
            add_upsample=add_upsample,  # 设置是否添加上采样
            resnet_eps=resnet_eps,  # 设置 ResNet 的 epsilon 值
            resnet_act_fn=resnet_act_fn,  # 设置 ResNet 的激活函数
            cross_attention_dim=cross_attention_dim,  # 设置交叉注意力维度
            attention_head_dim=attention_head_dim,  # 设置注意力头维度
        )

    # 如果未匹配到任何上采样块类型，抛出值错误
    raise ValueError(f"{up_block_type} does not exist.")
# 定义一个小型自编码器块，继承自 nn.Module
class AutoencoderTinyBlock(nn.Module):
    """
    Tiny Autoencoder block used in [`AutoencoderTiny`]. It is a mini residual module consisting of plain conv + ReLU
    blocks.

    Args:
        in_channels (`int`): The number of input channels.
        out_channels (`int`): The number of output channels.
        act_fn (`str`):
            ` The activation function to use. Supported values are `"swish"`, `"mish"`, `"gelu"`, and `"relu"`.

    Returns:
        `torch.Tensor`: A tensor with the same shape as the input tensor, but with the number of channels equal to
        `out_channels`.
    """

    # 初始化函数，接受输入通道数、输出通道数和激活函数类型
    def __init__(self, in_channels: int, out_channels: int, act_fn: str):
        # 调用父类初始化
        super().__init__()
        # 获取指定的激活函数
        act_fn = get_activation(act_fn)
        # 定义一个序列，包括多个卷积层和激活函数
        self.conv = nn.Sequential(
            # 第一层卷积，输入通道数、输出通道数、卷积核大小和填充方式
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 添加激活函数
            act_fn,
            # 第二层卷积，保持输出通道数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # 添加激活函数
            act_fn,
            # 第三层卷积，保持输出通道数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        # 判断输入和输出通道是否相同，决定使用卷积或身份映射
        self.skip = (
            # 如果通道数不一致，使用 1x1 卷积进行跳跃连接
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        # 使用 ReLU 进行特征融合
        self.fuse = nn.ReLU()

    # 定义前向传播函数，接受输入张量 x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回卷积输出和跳跃连接的和，经过融合激活函数处理
        return self.fuse(self.conv(x) + self.skip(x))


# 定义一个 2D UNet 中间块，继承自 nn.Module
class UNetMidBlock2D(nn.Module):
    """
    A 2D UNet mid-block [`UNetMidBlock2D`] with multiple residual blocks and optional attention blocks.
    # 参数说明
    Args:
        in_channels (`int`): 输入通道的数量。
        temb_channels (`int`): 时间嵌入通道的数量。
        dropout (`float`, *optional*, defaults to 0.0): dropout 比率，用于防止过拟合。
        num_layers (`int`, *optional*, defaults to 1): 残差块的数量。
        resnet_eps (`float`, *optional*, 1e-6 ): 残差块的 epsilon 值，用于数值稳定性。
        resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
            应用于时间嵌入的归一化类型。可以改善模型在长时间依赖任务上的表现。
        resnet_act_fn (`str`, *optional*, defaults to `swish`): 残差块的激活函数类型。
        resnet_groups (`int`, *optional*, defaults to 32):
            残差块中组归一化层使用的组数量。
        attn_groups (`Optional[int]`, *optional*, defaults to None): 注意力块的组数量。
        resnet_pre_norm (`bool`, *optional*, defaults to `True`):
            是否在残差块中使用预归一化。
        add_attention (`bool`, *optional*, defaults to `True`): 是否添加注意力块。
        attention_head_dim (`int`, *optional*, defaults to 1):
            单个注意力头的维度。注意力头的数量由该值和输入通道的数量决定。
        output_scale_factor (`float`, *optional*, defaults to 1.0): 输出的缩放因子。

    # 返回值说明
    Returns:
        `torch.Tensor`: 最后一个残差块的输出，形状为 `(batch_size, in_channels,
        height, width)`。

    """

    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout比率，默认为0.0
        num_layers: int = 1,  # 残差块数量，默认为1
        resnet_eps: float = 1e-6,  # 残差块的epsilon值
        resnet_time_scale_shift: str = "default",  # 时间尺度归一化的类型
        resnet_act_fn: str = "swish",  # 残差块的激活函数
        resnet_groups: int = 32,  # 残差块的组数量
        attn_groups: Optional[int] = None,  # 注意力块的组数量
        resnet_pre_norm: bool = True,  # 是否使用预归一化
        add_attention: bool = True,  # 是否添加注意力块
        attention_head_dim: int = 1,  # 注意力头的维度
        output_scale_factor: float = 1.0,  # 输出缩放因子
    # 前向传播函数，接受隐藏状态和时间嵌入作为输入
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 将输入的隐藏状态通过第一个残差块进行处理
        hidden_states = self.resnets[0](hidden_states, temb)
        # 遍历剩余的注意力块和残差块进行处理
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            # 如果当前存在注意力块，则进行处理
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            # 将处理后的隐藏状态通过当前的残差块进行处理
            hidden_states = resnet(hidden_states, temb)

        # 返回最终的隐藏状态
        return hidden_states
# 定义一个继承自 nn.Module 的类 UNetMidBlock2DCrossAttn
class UNetMidBlock2DCrossAttn(nn.Module):
    # 初始化方法，定义各个参数
    def __init__(
        # 输入通道数
        in_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # 输出通道数（可选，默认为 None）
        out_channels: Optional[int] = None,
        # dropout 概率（默认为 0.0）
        dropout: float = 0.0,
        # 层数（默认为 1）
        num_layers: int = 1,
        # 每个块的变换器层数（默认为 1）
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        # ResNet 的 epsilon 值（默认为 1e-6）
        resnet_eps: float = 1e-6,
        # ResNet 的时间缩放偏移方式（默认为 "default"）
        resnet_time_scale_shift: str = "default",
        # ResNet 的激活函数类型（默认为 "swish"）
        resnet_act_fn: str = "swish",
        # ResNet 的组数（默认为 32）
        resnet_groups: int = 32,
        # 输出的 ResNet 组数（可选，默认为 None）
        resnet_groups_out: Optional[int] = None,
        # 是否进行 ResNet 预归一化（默认为 True）
        resnet_pre_norm: bool = True,
        # 注意力头的数量（默认为 1）
        num_attention_heads: int = 1,
        # 输出缩放因子（默认为 1.0）
        output_scale_factor: float = 1.0,
        # 交叉注意力维度（默认为 1280）
        cross_attention_dim: int = 1280,
        # 是否使用双重交叉注意力（默认为 False）
        dual_cross_attention: bool = False,
        # 是否使用线性投影（默认为 False）
        use_linear_projection: bool = False,
        # 是否提升注意力计算精度（默认为 False）
        upcast_attention: bool = False,
        # 注意力类型（默认为 "default"）
        attention_type: str = "default",
    # 前向传播方法
    def forward(
        # 隐藏状态的张量
        hidden_states: torch.Tensor,
        # 时间嵌入的张量（可选，默认为 None）
        temb: Optional[torch.Tensor] = None,
        # 编码器隐藏状态的张量（可选，默认为 None）
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 注意力掩码（可选，默认为 None）
        attention_mask: Optional[torch.Tensor] = None,
        # 交叉注意力参数（可选，默认为 None）
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 编码器注意力掩码（可选，默认为 None）
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # 定义返回类型为 torch.Tensor
        if cross_attention_kwargs is not None:  # 检查交叉注意力参数是否存在
            if cross_attention_kwargs.get("scale", None) is not None:  # 检查是否有 scale 参数
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")  # 记录警告，提示 scale 参数已弃用

        hidden_states = self.resnets[0](hidden_states, temb)  # 使用第一个残差网络处理隐藏状态和时间嵌入
        for attn, resnet in zip(self.attentions, self.resnets[1:]):  # 遍历注意力层和残差网络（跳过第一个）
            if self.training and self.gradient_checkpointing:  # 如果处于训练模式并且启用了梯度检查点

                def create_custom_forward(module, return_dict=None):  # 定义自定义前向传播函数
                    def custom_forward(*inputs):  # 定义实际的前向传播实现
                        if return_dict is not None:  # 如果返回字典不为 None
                            return module(*inputs, return_dict=return_dict)  # 调用模块并返回字典
                        else:  # 如果返回字典为 None
                            return module(*inputs)  # 直接调用模块并返回结果

                    return custom_forward  # 返回自定义前向传播函数

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}  # 设置检查点的关键字参数
                hidden_states = attn(  # 调用注意力层处理隐藏状态
                    hidden_states,  # 输入隐藏状态
                    encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
                    cross_attention_kwargs=cross_attention_kwargs,  # 交叉注意力参数
                    attention_mask=attention_mask,  # 注意力掩码
                    encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码
                    return_dict=False,  # 不返回字典格式
                )[0]  # 取处理结果的第一个元素
                hidden_states = torch.utils.checkpoint.checkpoint(  # 使用梯度检查点
                    create_custom_forward(resnet),  # 创建残差网络的自定义前向函数
                    hidden_states,  # 输入隐藏状态
                    temb,  # 输入时间嵌入
                    **ckpt_kwargs,  # 解包关键字参数
                )
            else:  # 如果不处于训练模式或不启用梯度检查点
                hidden_states = attn(  # 调用注意力层处理隐藏状态
                    hidden_states,  # 输入隐藏状态
                    encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
                    cross_attention_kwargs=cross_attention_kwargs,  # 交叉注意力参数
                    attention_mask=attention_mask,  # 注意力掩码
                    encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码
                    return_dict=False,  # 不返回字典格式
                )[0]  # 取处理结果的第一个元素
                hidden_states = resnet(hidden_states, temb)  # 使用残差网络处理隐藏状态和时间嵌入

        return hidden_states  # 返回最终的隐藏状态
# 定义一个 UNet 中间块类，继承自 nn.Module
class UNetMidBlock2DSimpleCrossAttn(nn.Module):
    # 初始化方法，设置类的参数
    def __init__(
        # 输入通道数
        in_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # Dropout 比例，默认为 0.0
        dropout: float = 0.0,
        # 层数，默认为 1
        num_layers: int = 1,
        # ResNet 的小 epsilon 值，默认为 1e-6
        resnet_eps: float = 1e-6,
        # ResNet 时间尺度偏移，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 激活函数类型，默认为 "swish"
        resnet_act_fn: str = "swish",
        # ResNet 组数，默认为 32
        resnet_groups: int = 32,
        # 是否使用预归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头维度，默认为 1
        attention_head_dim: int = 1,
        # 输出缩放因子，默认为 1.0
        output_scale_factor: float = 1.0,
        # 交叉注意力维度，默认为 1280
        cross_attention_dim: int = 1280,
        # 是否跳过时间激活，默认为 False
        skip_time_act: bool = False,
        # 是否仅使用交叉注意力，默认为 False
        only_cross_attention: bool = False,
        # 交叉注意力的归一化方法，可选参数
        cross_attention_norm: Optional[str] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 设置是否使用交叉注意力
        self.has_cross_attention = True

        # 设置注意力头的维度
        self.attention_head_dim = attention_head_dim
        
        # 计算 ResNet 的组数，如果未提供，则取输入通道数的四分之一和 32 的最小值
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # 计算注意力头的数量
        self.num_heads = in_channels // self.attention_head_dim

        # 至少存在一个 ResNet 块
        resnets = [
            # 创建一个 ResNet 块
            ResnetBlock2D(
                # 输入通道数
                in_channels=in_channels,
                # 输出通道数
                out_channels=in_channels,
                # 时间嵌入通道数
                temb_channels=temb_channels,
                # 正则化参数
                eps=resnet_eps,
                # ResNet 组数
                groups=resnet_groups,
                # dropout 概率
                dropout=dropout,
                # 时间嵌入归一化方法
                time_embedding_norm=resnet_time_scale_shift,
                # 非线性激活函数
                non_linearity=resnet_act_fn,
                # 输出缩放因子
                output_scale_factor=output_scale_factor,
                # 是否进行预归一化
                pre_norm=resnet_pre_norm,
                # 是否跳过时间激活
                skip_time_act=skip_time_act,
            )
        ]
        # 初始化注意力层列表
        attentions = []

        # 循环创建指定数量的层
        for _ in range(num_layers):
            # 根据是否具有缩放点积注意力，选择处理器
            processor = (
                AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            )

            # 添加注意力层到列表中
            attentions.append(
                Attention(
                    # 查询的维度
                    query_dim=in_channels,
                    # 交叉注意力的维度
                    cross_attention_dim=in_channels,
                    # 注意力头的数量
                    heads=self.num_heads,
                    # 每个头的维度
                    dim_head=self.attention_head_dim,
                    # 额外的 KV 投影维度
                    added_kv_proj_dim=cross_attention_dim,
                    # 归一化的组数
                    norm_num_groups=resnet_groups,
                    # 是否使用偏置
                    bias=True,
                    # 是否上cast softmax
                    upcast_softmax=True,
                    # 是否仅使用交叉注意力
                    only_cross_attention=only_cross_attention,
                    # 交叉注意力的归一化方法
                    cross_attention_norm=cross_attention_norm,
                    # 设置处理器
                    processor=processor,
                )
            )
            # 添加另一个 ResNet 块到列表中
            resnets.append(
                ResnetBlock2D(
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

        # 将注意力层转为可训练模块列表
        self.attentions = nn.ModuleList(attentions)
        # 将 ResNet 块转为可训练模块列表
        self.resnets = nn.ModuleList(resnets)

    def forward(
        # 前向传播函数的定义
        self,
        # 输入的隐状态
        hidden_states: torch.Tensor,
        # 可选的时间嵌入
        temb: Optional[torch.Tensor] = None,
        # 可选的编码器隐状态
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 可选的注意力掩码
        attention_mask: Optional[torch.Tensor] = None,
        # 可选的交叉注意力参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 可选的编码器注意力掩码
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 如果 cross_attention_kwargs 为 None，则初始化为空字典
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        # 检查 cross_attention_kwargs 中是否有 "scale" 参数，如果有则记录警告信息
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # 如果 attention_mask 为 None
        if attention_mask is None:
            # 如果 encoder_hidden_states 已定义，表示正在进行交叉注意力，因此使用交叉注意力掩码
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            # 当 attention_mask 已定义时，不检查 encoder_attention_mask
            # 这是为了与 UnCLIP 兼容，UnCLIP 使用 'attention_mask' 参数作为交叉注意力掩码
            # TODO: UnCLIP 应该通过 encoder_attention_mask 参数而不是 attention_mask 参数来表达交叉注意力掩码
            #       然后可以简化整个 if/else 块为：
            #         mask = attention_mask if encoder_hidden_states is None else encoder_attention_mask
            mask = attention_mask

        # 通过第一个残差网络处理隐藏状态
        hidden_states = self.resnets[0](hidden_states, temb)
        # 遍历注意力层和后续的残差网络
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            # 使用当前的注意力层处理隐藏状态
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,  # 传递编码器的隐藏状态
                attention_mask=mask,  # 传递注意力掩码
                **cross_attention_kwargs,  # 解包交叉注意力参数
            )

            # 通过当前的残差网络处理隐藏状态
            hidden_states = resnet(hidden_states, temb)

        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个名为 AttnDownBlock2D 的类，继承自 nn.Module
class AttnDownBlock2D(nn.Module):
    # 初始化方法，接受多个参数以设置层的属性
    def __init__(
        # 输入通道数
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # dropout 概率，默认为 0.0
        dropout: float = 0.0,
        # 层数，默认为 1
        num_layers: int = 1,
        # ResNet 的 epsilon 值，防止除零错误，默认为 1e-6
        resnet_eps: float = 1e-6,
        # ResNet 时间尺度偏移，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 激活函数，默认为 "swish"
        resnet_act_fn: str = "swish",
        # ResNet 组数，默认为 32
        resnet_groups: int = 32,
        # 是否在 ResNet 中使用预归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头的维度，默认为 1
        attention_head_dim: int = 1,
        # 输出缩放因子，默认为 1.0
        output_scale_factor: float = 1.0,
        # 下采样的填充大小，默认为 1
        downsample_padding: int = 1,
        # 下采样类型，默认为 "conv"
        downsample_type: str = "conv",
    ):
        # 调用父类构造函数初始化
        super().__init__()
        # 初始化空列表，用于存储 ResNet 块
        resnets = []
        # 初始化空列表，用于存储注意力机制模块
        attentions = []
        # 保存下采样类型
        self.downsample_type = downsample_type

        # 如果未指定注意力头的维度，则发出警告，并默认使用输出通道数
        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            # 将注意力头维度设置为输出通道数
            attention_head_dim = out_channels

        # 遍历层数以构建 ResNet 块和注意力模块
        for i in range(num_layers):
            # 确定输入通道数，第一层使用初始通道数，其余层使用输出通道数
            in_channels = in_channels if i == 0 else out_channels
            # 创建并添加 ResNet 块到 resnets 列表
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # 防止除零的epsilon值
                    groups=resnet_groups,  # 分组数
                    dropout=dropout,  # dropout 比率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入的归一化方式
                    non_linearity=resnet_act_fn,  # 非线性激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 是否在前面进行归一化
                )
            )
            # 创建并添加注意力模块到 attentions 列表
            attentions.append(
                Attention(
                    out_channels,  # 输出通道数
                    heads=out_channels // attention_head_dim,  # 注意力头的数量
                    dim_head=attention_head_dim,  # 每个注意力头的维度
                    rescale_output_factor=output_scale_factor,  # 输出缩放因子
                    eps=resnet_eps,  # 防止除零的epsilon值
                    norm_num_groups=resnet_groups,  # 归一化的组数
                    residual_connection=True,  # 是否使用残差连接
                    bias=True,  # 是否使用偏置
                    upcast_softmax=True,  # 是否上溯 softmax
                    _from_deprecated_attn_block=True,  # 是否来自于已弃用的注意力块
                )
            )

        # 将注意力模块列表转换为 nn.ModuleList，便于管理
        self.attentions = nn.ModuleList(attentions)
        # 将 ResNet 块列表转换为 nn.ModuleList，便于管理
        self.resnets = nn.ModuleList(resnets)

        # 根据下采样类型选择相应的下采样方法
        if downsample_type == "conv":
            # 创建卷积下采样模块并存储
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,  # 输出通道数
                        use_conv=True,  # 是否使用卷积
                        out_channels=out_channels,  # 输出通道数
                        padding=downsample_padding,  # 填充
                        name="op"  # 模块名称
                    )
                ]
            )
        elif downsample_type == "resnet":
            # 创建 ResNet 下采样块并存储
            self.downsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,  # 输入通道数
                        out_channels=out_channels,  # 输出通道数
                        temb_channels=temb_channels,  # 时间嵌入通道数
                        eps=resnet_eps,  # 防止除零的epsilon值
                        groups=resnet_groups,  # 分组数
                        dropout=dropout,  # dropout 比率
                        time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入的归一化方式
                        non_linearity=resnet_act_fn,  # 非线性激活函数
                        output_scale_factor=output_scale_factor,  # 输出缩放因子
                        pre_norm=resnet_pre_norm,  # 是否在前面进行归一化
                        down=True,  # 指示为下采样
                    )
                ]
            )
        else:
            # 如果没有匹配的下采样类型，则将下采样模块设置为 None
            self.downsamplers = None
    # 前向传播方法，处理隐状态和可选的其他参数，返回处理后的隐状态和输出状态
    def forward(
            self,
            hidden_states: torch.Tensor,  # 输入的隐状态张量
            temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
            upsample_size: Optional[int] = None,  # 可选的上采样尺寸
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的交叉注意力参数字典
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:  # 返回隐状态和输出状态的元组
            # 如果没有提供交叉注意力参数，则初始化为空字典
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            # 检查是否传入了 scale 参数，如果有，则发出警告，因为这个参数已被弃用
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    
            output_states = ()  # 初始化输出状态为一个空元组
    
            # 遍历自定义的残差网络和注意力层
            for resnet, attn in zip(self.resnets, self.attentions):
                # 将隐状态传递给残差网络并更新隐状态
                hidden_states = resnet(hidden_states, temb)
                # 将隐状态传递给注意力层，并更新隐状态
                hidden_states = attn(hidden_states, **cross_attention_kwargs)
                # 将当前隐状态添加到输出状态元组中
                output_states = output_states + (hidden_states,)
    
            # 检查是否存在下采样层
            if self.downsamplers is not None:
                # 遍历每个下采样层
                for downsampler in self.downsamplers:
                    # 根据下采样类型选择不同的处理方式
                    if self.downsample_type == "resnet":
                        hidden_states = downsampler(hidden_states, temb=temb)  # 使用时间嵌入处理
                    else:
                        hidden_states = downsampler(hidden_states)  # 不使用时间嵌入处理
    
                # 将最后的隐状态添加到输出状态中
                output_states += (hidden_states,)
    
            # 返回处理后的隐状态和输出状态
            return hidden_states, output_states
# 定义一个名为 CrossAttnDownBlock2D 的类，继承自 nn.Module
class CrossAttnDownBlock2D(nn.Module):
    # 初始化方法，接收多个参数以配置模块
    def __init__(
        # 输入通道数
        self,
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # dropout 比例，默认为 0.0
        dropout: float = 0.0,
        # 层数，默认为 1
        num_layers: int = 1,
        # 每个块的变换器层数，可以是单个整数或整数元组
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        # ResNet 的 epsilon 值，默认为 1e-6
        resnet_eps: float = 1e-6,
        # ResNet 时间缩放偏移的类型，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 的激活函数，默认为 "swish"
        resnet_act_fn: str = "swish",
        # ResNet 的分组数，默认为 32
        resnet_groups: int = 32,
        # 是否使用预归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头的数量，默认为 1
        num_attention_heads: int = 1,
        # 交叉注意力维度，默认为 1280
        cross_attention_dim: int = 1280,
        # 输出缩放因子，默认为 1.0
        output_scale_factor: float = 1.0,
        # 下采样填充的大小，默认为 1
        downsample_padding: int = 1,
        # 是否添加下采样，默认为 True
        add_downsample: bool = True,
        # 是否使用双重交叉注意力，默认为 False
        dual_cross_attention: bool = False,
        # 是否使用线性投影，默认为 False
        use_linear_projection: bool = False,
        # 是否仅使用交叉注意力，默认为 False
        only_cross_attention: bool = False,
        # 是否提升注意力精度，默认为 False
        upcast_attention: bool = False,
        # 注意力类型，默认为 "default"
        attention_type: str = "default",
    # 初始化父类
        ):
            super().__init__()
            # 初始化残差块列表
            resnets = []
            # 初始化注意力机制列表
            attentions = []
    
            # 设置是否使用交叉注意力
            self.has_cross_attention = True
            # 设置注意力头的数量
            self.num_attention_heads = num_attention_heads
            # 如果每个块的变换层是整数，则扩展为列表
            if isinstance(transformer_layers_per_block, int):
                transformer_layers_per_block = [transformer_layers_per_block] * num_layers
    
            # 遍历每一层
            for i in range(num_layers):
                # 确定输入通道数
                in_channels = in_channels if i == 0 else out_channels
                # 添加残差块到列表
                resnets.append(
                    ResnetBlock2D(
                        in_channels=in_channels,  # 输入通道数
                        out_channels=out_channels,  # 输出通道数
                        temb_channels=temb_channels,  # 时间嵌入通道数
                        eps=resnet_eps,  #  epsilon值
                        groups=resnet_groups,  # 组数
                        dropout=dropout,  # dropout比率
                        time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化
                        non_linearity=resnet_act_fn,  # 非线性激活函数
                        output_scale_factor=output_scale_factor,  # 输出缩放因子
                        pre_norm=resnet_pre_norm,  # 是否使用预归一化
                    )
                )
                # 检查是否使用双重交叉注意力
                if not dual_cross_attention:
                    # 添加普通的变换模型到列表
                    attentions.append(
                        Transformer2DModel(
                            num_attention_heads,  # 注意力头数量
                            out_channels // num_attention_heads,  # 每个头的输出通道数
                            in_channels=out_channels,  # 输入通道数
                            num_layers=transformer_layers_per_block[i],  # 层数
                            cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                            norm_num_groups=resnet_groups,  # 归一化组数
                            use_linear_projection=use_linear_projection,  # 是否使用线性投影
                            only_cross_attention=only_cross_attention,  # 是否仅使用交叉注意力
                            upcast_attention=upcast_attention,  # 是否向上投射注意力
                            attention_type=attention_type,  # 注意力类型
                        )
                    )
                else:
                    # 添加双重变换模型到列表
                    attentions.append(
                        DualTransformer2DModel(
                            num_attention_heads,  # 注意力头数量
                            out_channels // num_attention_heads,  # 每个头的输出通道数
                            in_channels=out_channels,  # 输入通道数
                            num_layers=1,  # 层数固定为1
                            cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                            norm_num_groups=resnet_groups,  # 归一化组数
                        )
                    )
            # 将注意力模型列表转换为nn.ModuleList
            self.attentions = nn.ModuleList(attentions)
            # 将残差块列表转换为nn.ModuleList
            self.resnets = nn.ModuleList(resnets)
    
            # 检查是否添加下采样层
            if add_downsample:
                # 创建下采样层列表
                self.downsamplers = nn.ModuleList(
                    [
                        Downsample2D(
                            out_channels,  # 输出通道数
                            use_conv=True,  # 是否使用卷积
                            out_channels=out_channels,  # 输出通道数
                            padding=downsample_padding,  # 填充
                            name="op"  # 操作名称
                        )
                    ]
                )
            else:
                # 如果不添加下采样层，则设置为None
                self.downsamplers = None
    
            # 设置梯度检查点开关为关闭
            self.gradient_checkpointing = False
    # 定义前向传播函数，接收多个参数
        def forward(
            self,
            # 隐藏状态张量，表示模型的内部状态
            hidden_states: torch.Tensor,
            # 可选的时间嵌入张量，用于控制生成的时间步
            temb: Optional[torch.Tensor] = None,
            # 可选的编码器隐藏状态张量，表示编码器的输出
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选的注意力掩码，用于屏蔽输入中不需要关注的部分
            attention_mask: Optional[torch.Tensor] = None,
            # 可选的交叉注意力参数字典，用于传递其他配置
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的编码器注意力掩码，控制编码器的注意力机制
            encoder_attention_mask: Optional[torch.Tensor] = None,
            # 可选的附加残差张量，作为额外的信息传递
            additional_residuals: Optional[torch.Tensor] = None,
    # 返回类型为元组，包含张量和一个张量元组
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # 检查交叉注意力参数是否不为空
        if cross_attention_kwargs is not None:
            # 如果"scale"参数存在，则发出警告，说明其已被弃用
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    
        # 初始化输出状态为空元组
        output_states = ()
    
        # 将残差网络和注意力层配对
        blocks = list(zip(self.resnets, self.attentions))
    
        # 遍历每一对残差网络和注意力层
        for i, (resnet, attn) in enumerate(blocks):
            # 如果在训练中并且启用了梯度检查点
            if self.training and self.gradient_checkpointing:
    
                # 定义创建自定义前向传播的函数
                def create_custom_forward(module, return_dict=None):
                    # 定义自定义前向传播逻辑
                    def custom_forward(*inputs):
                        # 根据是否返回字典调用模块
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
    
                    return custom_forward
    
                # 根据 PyTorch 版本设置检查点参数
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                # 使用检查点机制计算隐藏状态
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                # 通过注意力层处理隐藏状态
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                # 直接通过残差网络处理隐藏状态
                hidden_states = resnet(hidden_states, temb)
                # 通过注意力层处理隐藏状态
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
    
            # 如果是最后一对块并且有额外的残差
            if i == len(blocks) - 1 and additional_residuals is not None:
                # 将额外的残差添加到隐藏状态
                hidden_states = hidden_states + additional_residuals
    
            # 更新输出状态，添加当前隐藏状态
            output_states = output_states + (hidden_states,)
    
        # 如果下采样器不为空
        if self.downsamplers is not None:
            # 遍历每个下采样器
            for downsampler in self.downsamplers:
                # 使用下采样器处理隐藏状态
                hidden_states = downsampler(hidden_states)
    
            # 更新输出状态，添加当前隐藏状态
            output_states = output_states + (hidden_states,)
    
        # 返回最终的隐藏状态和输出状态
        return hidden_states, output_states
# 定义一个二维向下块，继承自 nn.Module
class DownBlock2D(nn.Module):
    # 初始化方法，定义各个参数及其默认值
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # ResNet 层数
        resnet_eps: float = 1e-6,  # ResNet 中的 epsilon
        resnet_time_scale_shift: str = "default",  # 时间缩放偏移设置
        resnet_act_fn: str = "swish",  # ResNet 的激活函数
        resnet_groups: int = 32,  # ResNet 的分组数
        resnet_pre_norm: bool = True,  # 是否使用预归一化
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_downsample: bool = True,  # 是否添加下采样
        downsample_padding: int = 1,  # 下采样填充
    ):
        # 调用父类构造函数
        super().__init__()
        # 初始化空的 ResNet 块列表
        resnets = []

        # 根据层数循环创建 ResNet 块
        for i in range(num_layers):
            # 确定当前层的输入通道数
            in_channels = in_channels if i == 0 else out_channels
            # 添加 ResNet 块到列表
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,  # 当前层的输入通道数
                    out_channels=out_channels,  # 当前层的输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # epsilon 参数
                    groups=resnet_groups,  # 分组数
                    dropout=dropout,  # dropout 概率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化
                    non_linearity=resnet_act_fn,  # 激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 预归一化标志
                )
            )

        # 将 ResNet 块列表转换为 ModuleList
        self.resnets = nn.ModuleList(resnets)

        # 根据标志决定是否添加下采样层
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(  # 创建下采样层
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            # 如果不添加下采样，则将其设置为 None
            self.downsamplers = None

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    # 定义前向传播方法
    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # 检查位置参数是否大于0，或关键字参数中的 "scale" 是否不为 None
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 定义弃用消息，提示用户 "scale" 参数已弃用，未来将引发错误
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用弃用函数，记录 "scale" 参数的弃用
            deprecate("scale", "1.0.0", deprecation_message)

        # 初始化输出状态元组
        output_states = ()

        # 遍历自定义的 ResNet 模块列表
        for resnet in self.resnets:
            # 如果在训练模式下并且启用了梯度检查点
            if self.training and self.gradient_checkpointing:

                # 定义用于创建自定义前向传播的函数
                def create_custom_forward(module):
                    # 定义自定义前向传播函数
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                # 检查 PyTorch 版本是否大于等于 1.11.0
                if is_torch_version(">=", "1.11.0"):
                    # 使用检查点技术来计算隐藏状态，防止内存泄漏
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    # 在较早的版本中调用检查点计算隐藏状态
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                # 在非训练模式下直接调用 ResNet 模块以计算隐藏状态
                hidden_states = resnet(hidden_states, temb)

            # 将计算出的隐藏状态添加到输出状态元组中
            output_states = output_states + (hidden_states,)

        # 如果存在下采样器
        if self.downsamplers is not None:
            # 遍历下采样器并计算隐藏状态
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            # 将下采样后的隐藏状态添加到输出状态元组中
            output_states = output_states + (hidden_states,)

        # 返回当前的隐藏状态和输出状态元组
        return hidden_states, output_states
# 定义一个 2D 下采样编码块的类，继承自 nn.Module
class DownEncoderBlock2D(nn.Module):
    # 初始化方法，设置各类参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        dropout: float = 0.0,  # dropout 概率，默认 0
        num_layers: int = 1,  # 层数，默认 1
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 值，默认 1e-6
        resnet_time_scale_shift: str = "default",  # 时间尺度偏移方式，默认值为 "default"
        resnet_act_fn: str = "swish",  # ResNet 使用的激活函数，默认是 "swish"
        resnet_groups: int = 32,  # ResNet 的组数，默认 32
        resnet_pre_norm: bool = True,  # 是否在前面进行归一化，默认 True
        output_scale_factor: float = 1.0,  # 输出缩放因子，默认 1.0
        add_downsample: bool = True,  # 是否添加下采样层，默认 True
        downsample_padding: int = 1,  # 下采样的填充大小，默认 1
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化一个空的列表，用于存放 ResNet 块
        resnets = []

        # 根据层数创建 ResNet 块
        for i in range(num_layers):
            # 如果是第一层，使用输入通道数；否则使用输出通道数
            in_channels = in_channels if i == 0 else out_channels
            # 根据时间尺度偏移方式选择相应的 ResNet 块
            if resnet_time_scale_shift == "spatial":
                # 创建一个带条件归一化的 ResNet 块，并添加到 resnets 列表
                resnets.append(
                    ResnetBlockCondNorm2D(
                        in_channels=in_channels,  # 输入通道数
                        out_channels=out_channels,  # 输出通道数
                        temb_channels=None,  # 时间嵌入通道数，默认 None
                        eps=resnet_eps,  # epsilon 值
                        groups=resnet_groups,  # 组数
                        dropout=dropout,  # dropout 概率
                        time_embedding_norm="spatial",  # 时间嵌入的归一化方式
                        non_linearity=resnet_act_fn,  # 激活函数
                        output_scale_factor=output_scale_factor,  # 输出缩放因子
                    )
                )
            else:
                # 创建一个标准的 ResNet 块，并添加到 resnets 列表
                resnets.append(
                    ResnetBlock2D(
                        in_channels=in_channels,  # 输入通道数
                        out_channels=out_channels,  # 输出通道数
                        temb_channels=None,  # 时间嵌入通道数，默认 None
                        eps=resnet_eps,  # epsilon 值
                        groups=resnet_groups,  # 组数
                        dropout=dropout,  # dropout 概率
                        time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入的归一化方式
                        non_linearity=resnet_act_fn,  # 激活函数
                        output_scale_factor=output_scale_factor,  # 输出缩放因子
                        pre_norm=resnet_pre_norm,  # 是否前归一化
                    )
                )

        # 将 ResNet 块列表转为 nn.ModuleList，便于管理
        self.resnets = nn.ModuleList(resnets)

        # 根据 add_downsample 参数决定是否添加下采样层
        if add_downsample:
            # 创建下采样层，并添加到 nn.ModuleList
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,  # 输入通道数
                        use_conv=True,  # 是否使用卷积进行下采样
                        out_channels=out_channels,  # 输出通道数
                        padding=downsample_padding,  # 填充大小
                        name="op"  # 下采样层的名称
                    )
                ]
            )
        else:
            # 如果不添加下采样层，设置为 None
            self.downsamplers = None
    # 定义前向传播函数，接受隐藏状态和可选参数
        def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # 检查是否传入多余的参数或已弃用的 `scale` 参数
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                # 设置弃用信息，提醒用户移除 `scale` 参数
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                # 调用弃用函数，记录警告信息
                deprecate("scale", "1.0.0", deprecation_message)
    
            # 遍历每个 ResNet 层，更新隐藏状态
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states, temb=None)
    
            # 如果存在下采样层，则逐个应用下采样
            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)
    
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个二维注意力下采样编码器块的类，继承自 nn.Module
class AttnDownEncoderBlock2D(nn.Module):
    # 初始化方法，接收多个参数用于配置编码器块
    def __init__(
        # 输入通道数
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 丢弃率，控制神经元随机失活的比例
        dropout: float = 0.0,
        # 编码器块的层数
        num_layers: int = 1,
        # ResNet 的小常量，用于防止除零错误
        resnet_eps: float = 1e-6,
        # ResNet 的时间尺度偏移，默认配置
        resnet_time_scale_shift: str = "default",
        # ResNet 使用的激活函数类型，默认为 swish
        resnet_act_fn: str = "swish",
        # ResNet 中的分组数
        resnet_groups: int = 32,
        # 是否在前面进行归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头的维度
        attention_head_dim: int = 1,
        # 输出缩放因子，默认为 1.0
        output_scale_factor: float = 1.0,
        # 是否添加下采样层，默认为 True
        add_downsample: bool = True,
        # 下采样的填充大小，默认为 1
        downsample_padding: int = 1,
    ):
        # 调用父类构造函数
        super().__init__()
        # 初始化空列表以存储残差块
        resnets = []
        # 初始化空列表以存储注意力模块
        attentions = []

        # 检查是否传入注意力头维度
        if attention_head_dim is None:
            # 记录警告信息，默认设置注意力头维度为输出通道数
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            # 将注意力头维度设置为输出通道数
            attention_head_dim = out_channels

        # 遍历层数以构建残差块和注意力模块
        for i in range(num_layers):
            # 第一层输入通道为 in_channels，其余层为 out_channels
            in_channels = in_channels if i == 0 else out_channels
            # 根据时间缩放偏移类型构建不同类型的残差块
            if resnet_time_scale_shift == "spatial":
                # 添加条件归一化的残差块到列表
                resnets.append(
                    ResnetBlockCondNorm2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        temb_channels=None,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm="spatial",
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                    )
                )
            else:
                # 添加普通残差块到列表
                resnets.append(
                    ResnetBlock2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        temb_channels=None,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    )
                )
            # 添加注意力模块到列表
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        # 将注意力模块列表转换为 nn.ModuleList
        self.attentions = nn.ModuleList(attentions)
        # 将残差块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

        # 根据标志决定是否添加下采样层
        if add_downsample:
            # 创建下采样模块并添加到列表
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            # 如果不添加下采样层，将其设置为 None
            self.downsamplers = None
    # 定义前向传播方法，接收隐藏状态和其他参数
        def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # 检查是否有额外的参数或已弃用的 scale 参数
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                # 构建弃用警告信息
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                # 调用 deprecate 函数显示弃用警告
                deprecate("scale", "1.0.0", deprecation_message)
    
            # 遍历自定义的 ResNet 和注意力层进行处理
            for resnet, attn in zip(self.resnets, self.attentions):
                # 通过 ResNet 层处理隐藏状态
                hidden_states = resnet(hidden_states, temb=None)
                # 通过注意力层处理更新后的隐藏状态
                hidden_states = attn(hidden_states)
    
            # 如果有下采样层，则依次处理隐藏状态
            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    # 通过下采样层处理隐藏状态
                    hidden_states = downsampler(hidden_states)
    
            # 返回处理后的隐藏状态
            return hidden_states
# 定义一个名为 AttnSkipDownBlock2D 的类，继承自 nn.Module
class AttnSkipDownBlock2D(nn.Module):
    # 初始化方法，定义类的构造函数
    def __init__(
        # 输入通道数，整型
        in_channels: int,
        # 输出通道数，整型
        out_channels: int,
        # 嵌入通道数，整型
        temb_channels: int,
        # dropout 率，浮点型，默认为 0.0
        dropout: float = 0.0,
        # 网络层数，整型，默认为 1
        num_layers: int = 1,
        # ResNet 的 epsilon 值，浮点型，默认为 1e-6
        resnet_eps: float = 1e-6,
        # ResNet 的时间尺度偏移方式，字符串，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 的激活函数类型，字符串，默认为 "swish"
        resnet_act_fn: str = "swish",
        # 是否在前面进行规范化，布尔值，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头的维度，整型，默认为 1
        attention_head_dim: int = 1,
        # 输出缩放因子，浮点型，默认为平方根2
        output_scale_factor: float = np.sqrt(2.0),
        # 是否添加下采样层，布尔值，默认为 True
        add_downsample: bool = True,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化一个空的模块列表用于存储注意力层
        self.attentions = nn.ModuleList([])
        # 初始化一个空的模块列表用于存储残差块
        self.resnets = nn.ModuleList([])

        # 检查 attention_head_dim 是否为 None
        if attention_head_dim is None:
            # 如果为 None，记录警告信息，并将其设置为输出通道数
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        # 根据层数创建残差块和注意力层
        for i in range(num_layers):
            # 设置当前层的输入通道数，如果是第一层则使用 in_channels，否则使用 out_channels
            in_channels = in_channels if i == 0 else out_channels
            # 添加一个 ResnetBlock2D 到 resnets 列表中
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # 小常数以防止除零
                    groups=min(in_channels // 4, 32),  # 分组数
                    groups_out=min(out_channels // 4, 32),  # 输出分组数
                    dropout=dropout,  # dropout 概率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化方式
                    non_linearity=resnet_act_fn,  # 非线性激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 是否在残差块前进行归一化
                )
            )
            # 添加一个 Attention 层到 attentions 列表中
            self.attentions.append(
                Attention(
                    out_channels,  # 输出通道数
                    heads=out_channels // attention_head_dim,  # 注意力头的数量
                    dim_head=attention_head_dim,  # 每个注意力头的维度
                    rescale_output_factor=output_scale_factor,  # 输出缩放因子
                    eps=resnet_eps,  # 小常数以防止除零
                    norm_num_groups=32,  # 归一化分组数
                    residual_connection=True,  # 是否使用残差连接
                    bias=True,  # 是否使用偏置
                    upcast_softmax=True,  # 是否使用上溢出 softmax
                    _from_deprecated_attn_block=True,  # 是否来自过时的注意力块
                )
            )

        # 检查是否需要添加下采样层
        if add_downsample:
            # 创建一个 ResnetBlock2D 作为下采样层
            self.resnet_down = ResnetBlock2D(
                in_channels=out_channels,  # 输入通道数
                out_channels=out_channels,  # 输出通道数
                temb_channels=temb_channels,  # 时间嵌入通道数
                eps=resnet_eps,  # 小常数以防止除零
                groups=min(out_channels // 4, 32),  # 分组数
                dropout=dropout,  # dropout 概率
                time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化方式
                non_linearity=resnet_act_fn,  # 非线性激活函数
                output_scale_factor=output_scale_factor,  # 输出缩放因子
                pre_norm=resnet_pre_norm,  # 是否在残差块前进行归一化
                use_in_shortcut=True,  # 是否在快捷连接中使用
                down=True,  # 是否进行下采样
                kernel="fir",  # 卷积核类型
            )
            # 创建下采样模块列表
            self.downsamplers = nn.ModuleList([FirDownsample2D(out_channels, out_channels=out_channels)])
            # 创建跳跃连接卷积层
            self.skip_conv = nn.Conv2d(3, out_channels, kernel_size=(1, 1), stride=(1, 1))
        else:
            # 如果不添加下采样层，则将相关属性设置为 None
            self.resnet_down = None
            self.downsamplers = None
            self.skip_conv = None

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入
        skip_sample: Optional[torch.Tensor] = None,  # 可选的跳跃样本
        *args,  # 额外的位置参数
        **kwargs,  # 额外的关键字参数
    # 定义返回类型为元组，包含张量和多个张量的元组
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        # 检查传入的参数是否存在，或 kwargs 中的 scale 是否不为 None
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 定义弃用信息，说明 scale 参数将被忽略并将来会引发错误
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用弃用函数，记录 scale 参数的弃用信息
            deprecate("scale", "1.0.0", deprecation_message)
    
        # 初始化输出状态为一个空元组
        output_states = ()
    
        # 遍历 resnet 和 attention 的组合
        for resnet, attn in zip(self.resnets, self.attentions):
            # 使用 resnet 处理隐藏状态和时间嵌入
            hidden_states = resnet(hidden_states, temb)
            # 使用 attention 处理更新后的隐藏状态
            hidden_states = attn(hidden_states)
            # 将当前隐藏状态添加到输出状态元组中
            output_states += (hidden_states,)
    
        # 检查是否存在下采样器
        if self.downsamplers is not None:
            # 使用下采样网络处理隐藏状态
            hidden_states = self.resnet_down(hidden_states, temb)
            # 遍历每个下采样器并处理跳跃样本
            for downsampler in self.downsamplers:
                skip_sample = downsampler(skip_sample)
    
            # 结合跳跃样本和隐藏状态，更新隐藏状态
            hidden_states = self.skip_conv(skip_sample) + hidden_states
    
            # 将当前隐藏状态添加到输出状态元组中
            output_states += (hidden_states,)
    
        # 返回更新后的隐藏状态、输出状态元组和跳跃样本
        return hidden_states, output_states, skip_sample
# 定义一个二维跳过块的类，继承自 nn.Module
class SkipDownBlock2D(nn.Module):
    # 初始化方法，设置输入和输出通道等参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # 层数
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 参数
        resnet_time_scale_shift: str = "default",  # 时间缩放偏移方式
        resnet_act_fn: str = "swish",  # ResNet 激活函数
        resnet_pre_norm: bool = True,  # 是否在前面进行归一化
        output_scale_factor: float = np.sqrt(2.0),  # 输出缩放因子
        add_downsample: bool = True,  # 是否添加下采样层
        downsample_padding: int = 1,  # 下采样时的填充
    ):
        super().__init__()  # 调用父类构造函数
        self.resnets = nn.ModuleList([])  # 初始化 ResNet 块列表

        # 循环创建每一层的 ResNet 块
        for i in range(num_layers):
            # 第一层使用输入通道，后续层使用输出通道
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(  # 添加 ResNet 块
                    in_channels=in_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # epsilon 参数
                    groups=min(in_channels // 4, 32),  # 输入通道组数
                    groups_out=min(out_channels // 4, 32),  # 输出通道组数
                    dropout=dropout,  # dropout 概率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化
                    non_linearity=resnet_act_fn,  # 激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 前归一化
                )
            )

        # 如果需要添加下采样层
        if add_downsample:
            self.resnet_down = ResnetBlock2D(  # 创建下采样 ResNet 块
                in_channels=out_channels,  # 输入通道数
                out_channels=out_channels,  # 输出通道数
                temb_channels=temb_channels,  # 时间嵌入通道数
                eps=resnet_eps,  # epsilon 参数
                groups=min(out_channels // 4, 32),  # 输出通道组数
                dropout=dropout,  # dropout 概率
                time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化
                non_linearity=resnet_act_fn,  # 激活函数
                output_scale_factor=output_scale_factor,  # 输出缩放因子
                pre_norm=resnet_pre_norm,  # 前归一化
                use_in_shortcut=True,  # 使用短接
                down=True,  # 启用下采样
                kernel="fir",  # 指定卷积核类型
            )
            # 创建下采样模块列表
            self.downsamplers = nn.ModuleList([FirDownsample2D(out_channels, out_channels=out_channels)])
            # 创建跳过连接卷积层
            self.skip_conv = nn.Conv2d(3, out_channels, kernel_size=(1, 1), stride=(1, 1))
        else:  # 如果不添加下采样层
            self.resnet_down = None  # 不使用下采样 ResNet 块
            self.downsamplers = None  # 不使用下采样模块列表
            self.skip_conv = None  # 不使用跳过连接卷积层

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入
        skip_sample: Optional[torch.Tensor] = None,  # 可选的跳过样本
        *args,  # 可变位置参数
        **kwargs,  # 可变关键字参数
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:  # 定义返回类型为元组，包含一个张量、一个张量元组和另一个张量
        if len(args) > 0 or kwargs.get("scale", None) is not None:  # 检查是否传入了位置参数或名为“scale”的关键字参数
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."  # 定义废弃警告信息
            deprecate("scale", "1.0.0", deprecation_message)  # 调用 deprecate 函数，记录“scale”参数的废弃信息

        output_states = ()  # 初始化输出状态为一个空元组

        for resnet in self.resnets:  # 遍历自定义的 ResNet 模型列表
            hidden_states = resnet(hidden_states, temb)  # 将当前的隐藏状态和时间嵌入传递给 ResNet，获取更新后的隐藏状态
            output_states += (hidden_states,)  # 将当前的隐藏状态添加到输出状态元组中

        if self.downsamplers is not None:  # 检查是否存在下采样模块
            hidden_states = self.resnet_down(hidden_states, temb)  # 使用 ResNet 下采样隐藏状态和时间嵌入
            for downsampler in self.downsamplers:  # 遍历下采样模块
                skip_sample = downsampler(skip_sample)  # 对跳过连接样本进行下采样

            hidden_states = self.skip_conv(skip_sample) + hidden_states  # 通过跳过卷积处理下采样样本，并与当前的隐藏状态相加

            output_states += (hidden_states,)  # 将更新后的隐藏状态添加到输出状态元组中

        return hidden_states, output_states, skip_sample  # 返回更新后的隐藏状态、输出状态元组和跳过样本
# 定义一个 2D ResNet 下采样块，继承自 nn.Module
class ResnetDownsampleBlock2D(nn.Module):
    # 初始化函数，定义各层参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # ResNet 层数
        resnet_eps: float = 1e-6,  # ResNet 中的 epsilon 值
        resnet_time_scale_shift: str = "default",  # 时间缩放偏移方式
        resnet_act_fn: str = "swish",  # 激活函数
        resnet_groups: int = 32,  # 组数
        resnet_pre_norm: bool = True,  # 是否使用预归一化
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_downsample: bool = True,  # 是否添加下采样层
        skip_time_act: bool = False,  # 是否跳过时间激活
    ):
        # 调用父类构造函数
        super().__init__()
        resnets = []  # 初始化 ResNet 层列表

        # 根据指定的层数构建 ResNet 块
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels  # 确定输入通道数
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # epsilon 值
                    groups=resnet_groups,  # 组数
                    dropout=dropout,  # dropout 概率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化方式
                    non_linearity=resnet_act_fn,  # 激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 是否使用预归一化
                    skip_time_act=skip_time_act,  # 是否跳过时间激活
                )
            )

        self.resnets = nn.ModuleList(resnets)  # 将 ResNet 层转换为模块列表

        # 如果需要，添加下采样层
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,  # 输入通道数
                        out_channels=out_channels,  # 输出通道数
                        temb_channels=temb_channels,  # 时间嵌入通道数
                        eps=resnet_eps,  # epsilon 值
                        groups=resnet_groups,  # 组数
                        dropout=dropout,  # dropout 概率
                        time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化方式
                        non_linearity=resnet_act_fn,  # 激活函数
                        output_scale_factor=output_scale_factor,  # 输出缩放因子
                        pre_norm=resnet_pre_norm,  # 是否使用预归一化
                        skip_time_act=skip_time_act,  # 是否跳过时间激活
                        down=True,  # 指定为下采样层
                    )
                ]
            )
        else:
            self.downsamplers = None  # 如果不需要下采样层，则为 None

        self.gradient_checkpointing = False  # 初始化梯度检查点为 False

    # 定义前向传播函数
    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs  # 前向传播输入
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # 检查参数是否存在，或者是否提供了已弃用的 `scale` 参数
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 定义弃用消息，告知用户 `scale` 参数已弃用并将在未来的版本中引发错误
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用 deprecate 函数记录弃用信息
            deprecate("scale", "1.0.0", deprecation_message)

        # 初始化输出状态元组
        output_states = ()

        # 遍历所有的 ResNet 模型
        for resnet in self.resnets:
            # 如果处于训练模式并启用梯度检查点
            if self.training and self.gradient_checkpointing:
                
                # 定义一个函数用于创建自定义前向传播
                def create_custom_forward(module):
                    # 定义自定义前向传播函数，调用传入的模块
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                # 检查 PyTorch 版本是否大于等于 1.11.0
                if is_torch_version(">=", "1.11.0"):
                    # 使用梯度检查点机制计算隐藏状态，禁用重入
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    # 在较旧版本的 PyTorch 中使用梯度检查点机制计算隐藏状态
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                # 在非训练模式下直接调用 ResNet 计算隐藏状态
                hidden_states = resnet(hidden_states, temb)

            # 将当前的隐藏状态添加到输出状态元组中
            output_states = output_states + (hidden_states,)

        # 检查是否存在下采样器
        if self.downsamplers is not None:
            # 遍历所有的下采样器
            for downsampler in self.downsamplers:
                # 使用下采样器计算隐藏状态
                hidden_states = downsampler(hidden_states, temb)

            # 将当前的隐藏状态添加到输出状态元组中
            output_states = output_states + (hidden_states,)

        # 返回最终的隐藏状态和输出状态元组
        return hidden_states, output_states
# 定义一个简单的二维交叉注意力下采样块类，继承自 nn.Module
class SimpleCrossAttnDownBlock2D(nn.Module):
    # 初始化方法，设置输入、输出通道等参数
    def __init__(
        # 输入通道数
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # dropout 概率，默认值为 0.0
        dropout: float = 0.0,
        # 层数，默认值为 1
        num_layers: int = 1,
        # ResNet 中的 epsilon 值，默认值为 1e-6
        resnet_eps: float = 1e-6,
        # ResNet 的时间尺度偏移设置，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 的激活函数类型，默认为 "swish"
        resnet_act_fn: str = "swish",
        # ResNet 的分组数，默认为 32
        resnet_groups: int = 32,
        # 是否在 ResNet 中使用预归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头的维度，默认为 1
        attention_head_dim: int = 1,
        # 交叉注意力的维度，默认为 1280
        cross_attention_dim: int = 1280,
        # 输出缩放因子，默认为 1.0
        output_scale_factor: float = 1.0,
        # 是否添加下采样层，默认为 True
        add_downsample: bool = True,
        # 是否跳过时间激活，默认为 False
        skip_time_act: bool = False,
        # 是否仅使用交叉注意力，默认为 False
        only_cross_attention: bool = False,
        # 交叉注意力的归一化方式，默认为 None
        cross_attention_norm: Optional[str] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化是否有交叉注意力标志
        self.has_cross_attention = True

        # 初始化残差网络和注意力模块的列表
        resnets = []
        attentions = []

        # 设置注意力头的维度
        self.attention_head_dim = attention_head_dim
        # 计算注意力头的数量
        self.num_heads = out_channels // self.attention_head_dim

        # 根据层数创建残差块
        for i in range(num_layers):
            # 设置输入通道，第一层使用给定的输入通道，其余层使用输出通道
            in_channels = in_channels if i == 0 else out_channels
            # 将残差块添加到列表中
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # 残差网络中的 epsilon 值
                    groups=resnet_groups,  # 分组数量
                    dropout=dropout,  # dropout 概率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入规范化
                    non_linearity=resnet_act_fn,  # 非线性激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 是否预归一化
                    skip_time_act=skip_time_act,  # 是否跳过时间激活
                )
            )

            # 根据是否有缩放点积注意力创建处理器
            processor = (
                AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            )

            # 将注意力模块添加到列表中
            attentions.append(
                Attention(
                    query_dim=out_channels,  # 查询维度
                    cross_attention_dim=out_channels,  # 交叉注意力维度
                    heads=self.num_heads,  # 注意力头数量
                    dim_head=attention_head_dim,  # 每个头的维度
                    added_kv_proj_dim=cross_attention_dim,  # 额外的键值投影维度
                    norm_num_groups=resnet_groups,  # 规范化的组数量
                    bias=True,  # 是否使用偏置
                    upcast_softmax=True,  # 是否上调 softmax
                    only_cross_attention=only_cross_attention,  # 是否仅使用交叉注意力
                    cross_attention_norm=cross_attention_norm,  # 交叉注意力的规范化
                    processor=processor,  # 使用的处理器
                )
            )
        # 将注意力模块列表转换为可训练模块
        self.attentions = nn.ModuleList(attentions)
        # 将残差块列表转换为可训练模块
        self.resnets = nn.ModuleList(resnets)

        # 如果需要添加下采样
        if add_downsample:
            # 创建下采样的残差块
            self.downsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,  # 输入通道数
                        out_channels=out_channels,  # 输出通道数
                        temb_channels=temb_channels,  # 时间嵌入通道数
                        eps=resnet_eps,  # 残差网络中的 epsilon 值
                        groups=resnet_groups,  # 分组数量
                        dropout=dropout,  # dropout 概率
                        time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入规范化
                        non_linearity=resnet_act_fn,  # 非线性激活函数
                        output_scale_factor=output_scale_factor,  # 输出缩放因子
                        pre_norm=resnet_pre_norm,  # 是否预归一化
                        skip_time_act=skip_time_act,  # 是否跳过时间激活
                        down=True,  # 表示这是下采样
                    )
                ]
            )
        else:
            # 如果不需要下采样，将下采样设置为 None
            self.downsamplers = None

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False
    # 定义一个前向传播方法，接受多个输入参数
        def forward(
            self,
            # 输入的隐藏状态张量
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
    # 返回值类型为元组，包含一个张量和一个张量元组
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # 如果未提供 cross_attention_kwargs，则使用空字典
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        # 检查是否传入了 scale 参数，若有则发出弃用警告
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    
        # 初始化输出状态为一个空元组
        output_states = ()
    
        # 检查 attention_mask 是否为 None
        if attention_mask is None:
            # 如果 encoder_hidden_states 已定义，则进行交叉注意力，使用交叉注意力掩码
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            # 如果已定义 attention_mask，则直接使用，不检查 encoder_attention_mask
            # 这为 UnCLIP 兼容性提供支持
            # TODO: UnCLIP 应该通过 encoder_attention_mask 参数表达交叉注意力掩码
            mask = attention_mask
    
        # 遍历 ResNet 和注意力层
        for resnet, attn in zip(self.resnets, self.attentions):
            # 在训练中且开启了梯度检查点
            if self.training and self.gradient_checkpointing:
                # 定义一个自定义前向传播函数
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        # 根据 return_dict 决定返回方式
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
    
                    return custom_forward
    
                # 使用检查点进行前向传播，节省内存
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                # 执行注意力层
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=mask,
                    **cross_attention_kwargs,
                )
            else:
                # 否则直接使用 ResNet 进行前向传播
                hidden_states = resnet(hidden_states, temb)
    
                # 执行注意力层
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=mask,
                    **cross_attention_kwargs,
                )
    
            # 将当前隐藏状态添加到输出状态元组中
            output_states = output_states + (hidden_states,)
    
        # 如果存在下采样层
        if self.downsamplers is not None:
            # 遍历所有下采样层
            for downsampler in self.downsamplers:
                # 执行下采样
                hidden_states = downsampler(hidden_states, temb)
    
            # 将下采样后的隐藏状态添加到输出状态元组中
            output_states = output_states + (hidden_states,)
    
        # 返回最终的隐藏状态和输出状态元组
        return hidden_states, output_states
# 定义一个二维下采样的神经网络模块，继承自 nn.Module
class KDownBlock2D(nn.Module):
    # 初始化方法，定义输入输出通道数、时间嵌入通道、dropout 概率等参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout 概率，默认为 0
        num_layers: int = 4,  # 残差层的数量，默认为 4
        resnet_eps: float = 1e-5,  # 残差层中的 epsilon 值，防止除零错误
        resnet_act_fn: str = "gelu",  # 残差层使用的激活函数，默认为 GELU
        resnet_group_size: int = 32,  # 残差层中组的大小
        add_downsample: bool = False,  # 是否添加下采样层的标志
    ):
        # 调用父类的初始化方法
        super().__init__()
        resnets = []  # 初始化一个空列表用于存储残差块

        # 根据层数构建残差块
        for i in range(num_layers):
            # 第一层使用输入通道，其他层使用输出通道
            in_channels = in_channels if i == 0 else out_channels
            # 计算组的数量
            groups = in_channels // resnet_group_size
            # 计算输出组的数量
            groups_out = out_channels // resnet_group_size

            # 创建残差块并添加到列表中
            resnets.append(
                ResnetBlockCondNorm2D(
                    in_channels=in_channels,  # 当前层的输入通道数
                    out_channels=out_channels,  # 当前层的输出通道数
                    dropout=dropout,  # 当前层的 dropout 概率
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    groups=groups,  # 当前层的组数量
                    groups_out=groups_out,  # 输出层的组数量
                    eps=resnet_eps,  # 残差层中的 epsilon 值
                    non_linearity=resnet_act_fn,  # 残差层的激活函数
                    time_embedding_norm="ada_group",  # 时间嵌入的归一化方式
                    conv_shortcut_bias=False,  # 卷积快捷连接是否使用偏置
                )
            )

        # 将残差块列表转换为 nn.ModuleList，以便于参数管理
        self.resnets = nn.ModuleList(resnets)

        # 根据标志决定是否添加下采样层
        if add_downsample:
            # 如果需要，创建一个下采样层并添加到列表中
            self.downsamplers = nn.ModuleList([KDownsample2D()])
        else:
            # 如果不需要下采样，设置为 None
            self.downsamplers = None

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接收隐藏状态和时间嵌入（可选）
    def forward(
        self, hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        *args, **kwargs  # 其他可选参数
    # 函数返回一个包含张量和元组的元组
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # 检查是否有额外参数或“scale”关键字参数不为 None
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 定义关于“scale”参数的弃用消息
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用弃用函数记录“scale”的弃用
            deprecate("scale", "1.0.0", deprecation_message)
    
        # 初始化输出状态为一个空元组
        output_states = ()
    
        # 遍历所有的 ResNet 模块
        for resnet in self.resnets:
            # 如果处于训练模式且启用了梯度检查点
            if self.training and self.gradient_checkpointing:
    
                # 定义一个创建自定义前向传播的函数
                def create_custom_forward(module):
                    # 定义自定义前向传播函数，接受任意输入并返回模块的输出
                    def custom_forward(*inputs):
                        return module(*inputs)
    
                    return custom_forward
    
                # 如果 PyTorch 版本大于等于 1.11.0
                if is_torch_version(">=", "1.11.0"):
                    # 使用检查点功能计算隐藏状态，传递自定义前向函数
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    # 否则使用检查点功能计算隐藏状态
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                # 如果不满足条件，则直接通过 ResNet 模块计算隐藏状态
                hidden_states = resnet(hidden_states, temb)
    
            # 将当前隐藏状态添加到输出状态元组中
            output_states += (hidden_states,)
    
        # 如果存在下采样器
        if self.downsamplers is not None:
            # 遍历每个下采样器
            for downsampler in self.downsamplers:
                # 通过下采样器计算隐藏状态
                hidden_states = downsampler(hidden_states)
    
        # 返回最终的隐藏状态和输出状态
        return hidden_states, output_states
# 定义一个名为 KCrossAttnDownBlock2D 的类，继承自 nn.Module
class KCrossAttnDownBlock2D(nn.Module):
    # 初始化方法，接受多个参数以设置模型的结构
    def __init__(
        self,
        in_channels: int,               # 输入通道数
        out_channels: int,              # 输出通道数
        temb_channels: int,             # 时间嵌入通道数
        cross_attention_dim: int,       # 跨注意力维度
        dropout: float = 0.0,           # dropout 概率，默认为 0
        num_layers: int = 4,            # 层数，默认为 4
        resnet_group_size: int = 32,    # ResNet 组的大小，默认为 32
        add_downsample: bool = True,    # 是否添加下采样，默认为 True
        attention_head_dim: int = 64,    # 注意力头维度，默认为 64
        add_self_attention: bool = False, # 是否添加自注意力，默认为 False
        resnet_eps: float = 1e-5,       # ResNet 的 epsilon 值，默认为 1e-5
        resnet_act_fn: str = "gelu",    # ResNet 的激活函数类型，默认为 "gelu"
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化空列表以存放 ResNet 块
        resnets = []
        # 初始化空列表以存放注意力块
        attentions = []

        # 设置是否包含跨注意力标志
        self.has_cross_attention = True

        # 创建指定数量的层
        for i in range(num_layers):
            # 第一层的输入通道数为 in_channels，之后的层使用 out_channels
            in_channels = in_channels if i == 0 else out_channels
            # 计算组数
            groups = in_channels // resnet_group_size
            groups_out = out_channels // resnet_group_size

            # 将 ResnetBlockCondNorm2D 添加到 resnets 列表
            resnets.append(
                ResnetBlockCondNorm2D(
                    in_channels=in_channels,        # 输入通道数
                    out_channels=out_channels,      # 输出通道数
                    dropout=dropout,                # dropout 概率
                    temb_channels=temb_channels,    # 时间嵌入通道数
                    groups=groups,                  # 组数
                    groups_out=groups_out,          # 输出组数
                    eps=resnet_eps,                 # epsilon 值
                    non_linearity=resnet_act_fn,    # 激活函数
                    time_embedding_norm="ada_group", # 时间嵌入归一化类型
                    conv_shortcut_bias=False,       # 是否使用卷积快捷连接偏置
                )
            )
            # 将 KAttentionBlock 添加到 attentions 列表
            attentions.append(
                KAttentionBlock(
                    out_channels,                     # 输出通道数
                    out_channels // attention_head_dim, # 注意力头数量
                    attention_head_dim,              # 注意力头维度
                    cross_attention_dim=cross_attention_dim, # 跨注意力维度
                    temb_channels=temb_channels,     # 时间嵌入通道数
                    attention_bias=True,             # 是否使用注意力偏置
                    add_self_attention=add_self_attention, # 是否添加自注意力
                    cross_attention_norm="layer_norm", # 跨注意力归一化类型
                    group_size=resnet_group_size,    # 组大小
                )
            )

        # 将 resnets 列表转换为 nn.ModuleList，以便可以在模型中使用
        self.resnets = nn.ModuleList(resnets)
        # 将 attentions 列表转换为 nn.ModuleList
        self.attentions = nn.ModuleList(attentions)

        # 根据参数决定是否添加下采样层
        if add_downsample:
            # 添加下采样模块
            self.downsamplers = nn.ModuleList([KDownsample2D()])
        else:
            # 如果不添加下采样，设置为 None
            self.downsamplers = None

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    # 前向传播方法，定义输入和输出
    def forward(
        self,
        hidden_states: torch.Tensor,         # 隐藏状态输入
        temb: Optional[torch.Tensor] = None, # 可选的时间嵌入
        encoder_hidden_states: Optional[torch.Tensor] = None, # 可选的编码器隐藏状态
        attention_mask: Optional[torch.Tensor] = None, # 可选的注意力掩码
        cross_attention_kwargs: Optional[Dict[str, Any]] = None, # 可选的跨注意力参数
        encoder_attention_mask: Optional[torch.Tensor] = None, # 可选的编码器注意力掩码
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # 如果没有传入 cross_attention_kwargs，则初始化为空字典
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        # 检查 cross_attention_kwargs 中是否存在 "scale" 参数，并发出警告
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # 初始化输出状态为一个空元组
        output_states = ()

        # 遍历 resnets 和 attentions 的对应元素
        for resnet, attn in zip(self.resnets, self.attentions):
            # 如果处于训练模式且开启了梯度检查点
            if self.training and self.gradient_checkpointing:

                # 创建自定义前向传播函数
                def create_custom_forward(module, return_dict=None):
                    # 定义自定义前向传播逻辑
                    def custom_forward(*inputs):
                        # 如果指定了返回字典，则返回包含字典的结果
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            # 否则返回普通结果
                            return module(*inputs)

                    return custom_forward

                # 设置检查点参数，针对 PyTorch 版本进行不同处理
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                # 使用检查点机制计算隐藏状态
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),  # 传入自定义前向函数
                    hidden_states,  # 输入隐藏状态
                    temb,  # 传入时间嵌入
                    **ckpt_kwargs,  # 传入检查点参数
                )
                # 使用注意力机制更新隐藏状态
                hidden_states = attn(
                    hidden_states,  # 输入隐藏状态
                    encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
                    emb=temb,  # 时间嵌入
                    attention_mask=attention_mask,  # 注意力掩码
                    cross_attention_kwargs=cross_attention_kwargs,  # 交叉注意力参数
                    encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码
                )
            else:
                # 如果不是训练模式或没有使用梯度检查点，直接通过 ResNet 更新隐藏状态
                hidden_states = resnet(hidden_states, temb)
                # 使用注意力机制更新隐藏状态
                hidden_states = attn(
                    hidden_states,  # 输入隐藏状态
                    encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
                    emb=temb,  # 时间嵌入
                    attention_mask=attention_mask,  # 注意力掩码
                    cross_attention_kwargs=cross_attention_kwargs,  # 交叉注意力参数
                    encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码
                )

            # 如果没有下采样层，输出状态添加 None
            if self.downsamplers is None:
                output_states += (None,)
            else:
                # 否则将当前隐藏状态添加到输出状态
                output_states += (hidden_states,)

        # 如果存在下采样层，则依次对隐藏状态进行下采样
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        # 返回最终的隐藏状态和输出状态
        return hidden_states, output_states
# 定义一个名为 AttnUpBlock2D 的类，继承自 nn.Module
class AttnUpBlock2D(nn.Module):
    # 初始化方法，接受多个参数以配置该模块
    def __init__(
        # 输入通道数
        self,
        in_channels: int,
        # 前一层输出通道数
        prev_output_channel: int,
        # 输出通道数
        out_channels: int,
        # 嵌入通道数
        temb_channels: int,
        # 分辨率索引，默认为 None
        resolution_idx: int = None,
        # dropout 率，默认为 0.0
        dropout: float = 0.0,
        # 层数，默认为 1
        num_layers: int = 1,
        # ResNet 中的小常数，避免除零
        resnet_eps: float = 1e-6,
        # ResNet 时间缩放偏移，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 激活函数，默认为 "swish"
        resnet_act_fn: str = "swish",
        # ResNet 组数，默认为 32
        resnet_groups: int = 32,
        # 是否在 ResNet 中使用预归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头维度，默认为 1
        attention_head_dim: int = 1,
        # 输出缩放因子，默认为 1.0
        output_scale_factor: float = 1.0,
        # 上采样类型，默认为 "conv"
        upsample_type: str = "conv",
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化空列表用于存储 ResNet 块
        resnets = []
        # 初始化空列表用于存储注意力层
        attentions = []

        # 设置上采样类型
        self.upsample_type = upsample_type

        # 如果没有传入注意力头维度
        if attention_head_dim is None:
            # 记录警告，建议使用默认的头维度
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            # 将注意力头维度设置为输出通道数
            attention_head_dim = out_channels

        # 遍历每一层
        for i in range(num_layers):
            # 设置残差跳过通道数，最后一层使用输入通道
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 设置当前 ResNet 块的输入通道数
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 创建 ResNet 块并添加到列表
            resnets.append(
                ResnetBlock2D(
                    # 输入通道数为当前 ResNet 的输入通道加上跳过的通道
                    in_channels=resnet_in_channels + res_skip_channels,
                    # 输出通道数
                    out_channels=out_channels,
                    # 时间嵌入通道数
                    temb_channels=temb_channels,
                    # 余弦相似性小常数
                    eps=resnet_eps,
                    # 分组数
                    groups=resnet_groups,
                    # dropout 概率
                    dropout=dropout,
                    # 时间嵌入的归一化方式
                    time_embedding_norm=resnet_time_scale_shift,
                    # 非线性激活函数
                    non_linearity=resnet_act_fn,
                    # 输出缩放因子
                    output_scale_factor=output_scale_factor,
                    # 是否进行预归一化
                    pre_norm=resnet_pre_norm,
                )
            )
            # 创建注意力层并添加到列表
            attentions.append(
                Attention(
                    # 输出通道数
                    out_channels,
                    # 注意力头的数量
                    heads=out_channels // attention_head_dim,
                    # 每个头的维度
                    dim_head=attention_head_dim,
                    # 输出重缩放因子
                    rescale_output_factor=output_scale_factor,
                    # 余弦相似性小常数
                    eps=resnet_eps,
                    # 归一化的组数
                    norm_num_groups=resnet_groups,
                    # 是否使用残差连接
                    residual_connection=True,
                    # 是否使用偏置
                    bias=True,
                    # 是否上升 softmax 的精度
                    upcast_softmax=True,
                    # 是否从已弃用的注意力块中获取
                    _from_deprecated_attn_block=True,
                )
            )

        # 将注意力层列表转换为 nn.ModuleList，以便于管理
        self.attentions = nn.ModuleList(attentions)
        # 将 ResNet 块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

        # 根据上采样类型选择上采样方法
        if upsample_type == "conv":
            # 使用卷积上采样，并创建 ModuleList
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        elif upsample_type == "resnet":
            # 使用 ResNet 块进行上采样，并创建 ModuleList
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        # 输入通道数
                        in_channels=out_channels,
                        # 输出通道数
                        out_channels=out_channels,
                        # 时间嵌入通道数
                        temb_channels=temb_channels,
                        # 余弦相似性小常数
                        eps=resnet_eps,
                        # 分组数
                        groups=resnet_groups,
                        # dropout 概率
                        dropout=dropout,
                        # 时间嵌入的归一化方式
                        time_embedding_norm=resnet_time_scale_shift,
                        # 非线性激活函数
                        non_linearity=resnet_act_fn,
                        # 输出缩放因子
                        output_scale_factor=output_scale_factor,
                        # 是否进行预归一化
                        pre_norm=resnet_pre_norm,
                        # 表示这是一个上采样的块
                        up=True,
                    )
                ]
            )
        else:
            # 如果上采样类型无效，则设为 None
            self.upsamplers = None

        # 存储当前分辨率索引
        self.resolution_idx = resolution_idx
    # 定义前向传播函数，接收隐藏状态及其他参数
    def forward(
            self,
            hidden_states: torch.Tensor,  # 当前的隐藏状态张量
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],  # 之前的隐藏状态元组
            temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
            upsample_size: Optional[int] = None,  # 可选的上采样大小
            *args,  # 额外的位置参数
            **kwargs,  # 额外的关键字参数
        ) -> torch.Tensor:  # 返回类型为张量
            # 检查是否传入了多余的参数或已弃用的 scale 参数
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                # 设置弃用警告信息
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                # 调用弃用函数发出警告
                deprecate("scale", "1.0.0", deprecation_message)
    
            # 遍历每一对残差网络和注意力层
            for resnet, attn in zip(self.resnets, self.attentions):
                # 从元组中弹出最后一个残差隐藏状态
                res_hidden_states = res_hidden_states_tuple[-1]
                # 更新残差隐藏状态元组，去掉最后一个元素
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                # 将当前隐藏状态和残差隐藏状态在维度1上拼接
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
    
                # 将拼接后的隐藏状态传入残差网络
                hidden_states = resnet(hidden_states, temb)
                # 将输出的隐藏状态传入注意力层
                hidden_states = attn(hidden_states)
    
            # 检查是否存在上采样器
            if self.upsamplers is not None:
                # 遍历每个上采样器
                for upsampler in self.upsamplers:
                    # 根据上采样类型选择处理方式
                    if self.upsample_type == "resnet":
                        # 将隐藏状态传入上采样器并提供时间嵌入
                        hidden_states = upsampler(hidden_states, temb=temb)
                    else:
                        # 将隐藏状态传入上采样器
                        hidden_states = upsampler(hidden_states)
    
            # 返回处理后的隐藏状态
            return hidden_states
# 定义一个名为 CrossAttnUpBlock2D 的类，继承自 nn.Module
class CrossAttnUpBlock2D(nn.Module):
    # 初始化方法，定义该类的构造函数
    def __init__(
        # 输入通道数
        self,
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 前一层输出通道数
        prev_output_channel: int,
        # 时间嵌入通道数
        temb_channels: int,
        # 可选的分辨率索引
        resolution_idx: Optional[int] = None,
        # Dropout 概率
        dropout: float = 0.0,
        # 层数
        num_layers: int = 1,
        # 每个块的 Transformer 层数，可以是单个整数或元组
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        # ResNet 的 epsilon 值，避免除零错误
        resnet_eps: float = 1e-6,
        # ResNet 的时间尺度偏移参数
        resnet_time_scale_shift: str = "default",
        # ResNet 的激活函数类型
        resnet_act_fn: str = "swish",
        # ResNet 的组数
        resnet_groups: int = 32,
        # 是否使用预归一化
        resnet_pre_norm: bool = True,
        # 注意力头的数量
        num_attention_heads: int = 1,
        # 跨注意力的维度
        cross_attention_dim: int = 1280,
        # 输出缩放因子
        output_scale_factor: float = 1.0,
        # 是否添加上采样层
        add_upsample: bool = True,
        # 是否使用双重跨注意力
        dual_cross_attention: bool = False,
        # 是否使用线性投影
        use_linear_projection: bool = False,
        # 是否仅使用跨注意力
        only_cross_attention: bool = False,
        # 是否提升注意力计算精度
        upcast_attention: bool = False,
        # 注意力类型
        attention_type: str = "default",
    # 继承父类的初始化方法
        ):
            super().__init__()
            # 初始化空列表用于存储 ResNet 和注意力模块
            resnets = []
            attentions = []
    
            # 设置是否有交叉注意力的标志
            self.has_cross_attention = True
            # 设置注意力头的数量
            self.num_attention_heads = num_attention_heads
    
            # 如果输入的是整数，则将其转换为包含多个相同值的列表
            if isinstance(transformer_layers_per_block, int):
                transformer_layers_per_block = [transformer_layers_per_block] * num_layers
    
            # 遍历每一层，构建 ResNet 和注意力模块
            for i in range(num_layers):
                # 设置残差跳跃通道数量
                res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
                # 设置 ResNet 输入通道
                resnet_in_channels = prev_output_channel if i == 0 else out_channels
    
                # 将 ResNet 模块添加到列表中
                resnets.append(
                    ResnetBlock2D(
                        in_channels=resnet_in_channels + res_skip_channels,  # 输入通道数
                        out_channels=out_channels,  # 输出通道数
                        temb_channels=temb_channels,  # 时间嵌入通道数
                        eps=resnet_eps,  # 小常数用于数值稳定性
                        groups=resnet_groups,  # 分组数
                        dropout=dropout,  # Dropout 率
                        time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入的归一化方式
                        non_linearity=resnet_act_fn,  # 非线性激活函数
                        output_scale_factor=output_scale_factor,  # 输出缩放因子
                        pre_norm=resnet_pre_norm,  # 是否进行预归一化
                    )
                )
                # 根据是否启用双重交叉注意力，选择不同的注意力模块
                if not dual_cross_attention:
                    attentions.append(
                        Transformer2DModel(
                            num_attention_heads,  # 注意力头数量
                            out_channels // num_attention_heads,  # 每个头的输出通道数
                            in_channels=out_channels,  # 输入通道数
                            num_layers=transformer_layers_per_block[i],  # 当前层的变换器层数
                            cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                            norm_num_groups=resnet_groups,  # 归一化分组数
                            use_linear_projection=use_linear_projection,  # 是否使用线性投影
                            only_cross_attention=only_cross_attention,  # 是否仅使用交叉注意力
                            upcast_attention=upcast_attention,  # 是否上调注意力精度
                            attention_type=attention_type,  # 注意力类型
                        )
                    )
                else:
                    attentions.append(
                        DualTransformer2DModel(
                            num_attention_heads,  # 注意力头数量
                            out_channels // num_attention_heads,  # 每个头的输出通道数
                            in_channels=out_channels,  # 输入通道数
                            num_layers=1,  # 仅使用一层
                            cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                            norm_num_groups=resnet_groups,  # 归一化分组数
                        )
                    )
            # 将注意力模块和 ResNet 模块转换为 nn.ModuleList 以便于管理
            self.attentions = nn.ModuleList(attentions)
            self.resnets = nn.ModuleList(resnets)
    
            # 根据是否添加上采样层初始化上采样模块
            if add_upsample:
                self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
            else:
                self.upsamplers = None
    
            # 初始化梯度检查点标志
            self.gradient_checkpointing = False
            # 设置分辨率索引
            self.resolution_idx = resolution_idx
    # 定义前向传播函数，接收多个参数
        def forward(
            self,
            # 当前隐藏状态的张量
            hidden_states: torch.Tensor,
            # 元组，包含残差隐藏状态的张量
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            # 可选的时间嵌入张量
            temb: Optional[torch.Tensor] = None,
            # 可选的编码器隐藏状态张量
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选的跨注意力参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的上采样大小
            upsample_size: Optional[int] = None,
            # 可选的注意力掩码张量
            attention_mask: Optional[torch.Tensor] = None,
            # 可选的编码器注意力掩码张量
            encoder_attention_mask: Optional[torch.Tensor] = None,
# 定义一个名为 UpBlock2D 的类，继承自 nn.Module
class UpBlock2D(nn.Module):
    # 初始化方法，接受多个参数来构造 UpBlock2D 对象
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        prev_output_channel: int,  # 前一层输出的通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        resolution_idx: Optional[int] = None,  # 分辨率索引，默认为 None
        dropout: float = 0.0,  # dropout 概率，默认为 0.0
        num_layers: int = 1,  # 层数，默认为 1
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 值，默认为 1e-6
        resnet_time_scale_shift: str = "default",  # ResNet 的时间缩放偏移，默认为 "default"
        resnet_act_fn: str = "swish",  # ResNet 的激活函数，默认为 "swish"
        resnet_groups: int = 32,  # ResNet 的组数，默认为 32
        resnet_pre_norm: bool = True,  # 是否进行预归一化，默认为 True
        output_scale_factor: float = 1.0,  # 输出缩放因子，默认为 1.0
        add_upsample: bool = True,  # 是否添加上采样层，默认为 True
    ):
        # 调用父类的初始化方法
        super().__init__()
        resnets = []  # 初始化一个空列表用于存储 ResNet 块

        # 根据 num_layers 创建 ResNet 块
        for i in range(num_layers):
            # 设置跳过通道数，如果是最后一层，则使用 in_channels，否则使用 out_channels
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 设置 ResNet 输入通道数，第一层使用 prev_output_channel，其余层使用 out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 将 ResNet 块添加到列表中
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,  # 输入通道数加上跳过通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # epsilon 值
                    groups=resnet_groups,  # 组数
                    dropout=dropout,  # dropout 概率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化
                    non_linearity=resnet_act_fn,  # 非线性激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 预归一化
                )
            )

        # 将 ResNet 块列表转换为 nn.ModuleList 以便于管理
        self.resnets = nn.ModuleList(resnets)

        # 如果需要添加上采样层，则初始化上采样模块
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None  # 不添加上采样层，设置为 None

        self.gradient_checkpointing = False  # 初始化梯度检查点标志为 False
        self.resolution_idx = resolution_idx  # 保存分辨率索引

    # 定义前向传播方法，接受输入的隐藏状态及其他参数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],  # 之前层的隐藏状态元组
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        upsample_size: Optional[int] = None,  # 可选的上采样大小
        *args,  # 其他位置参数
        **kwargs,  # 其他关键字参数
    ) -> torch.Tensor:
        # 检查是否传入参数或 'scale' 参数
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 设置弃用消息，提示用户 'scale' 参数已被弃用
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用 deprecate 函数记录弃用
            deprecate("scale", "1.0.0", deprecation_message)

        # 检查 FreeU 是否启用
        is_freeu_enabled = (
            getattr(self, "s1", None)  # 获取属性 s1
            and getattr(self, "s2", None)  # 获取属性 s2
            and getattr(self, "b1", None)  # 获取属性 b1
            and getattr(self, "b2", None)  # 获取属性 b2
        )

        # 遍历所有 ResNet 模型
        for resnet in self.resnets:
            # 弹出最后的 ResNet 隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]  # 更新元组，去掉最后一个元素

            # 如果 FreeU 被启用，则仅对前两个阶段进行操作
            if is_freeu_enabled:
                # 调用 apply_freeu 函数处理隐藏状态
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,  # 当前分辨率索引
                    hidden_states,  # 当前隐藏状态
                    res_hidden_states,  # ResNet 隐藏状态
                    s1=self.s1,  # s1 属性
                    s2=self.s2,  # s2 属性
                    b1=self.b1,  # b1 属性
                    b2=self.b2,  # b2 属性
                )

            # 连接当前隐藏状态和 ResNet 隐藏状态
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # 如果处于训练模式且启用梯度检查点
            if self.training and self.gradient_checkpointing:
                # 创建自定义前向传播函数
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)  # 调用模块的前向传播

                    return custom_forward

                # 根据 PyTorch 版本选择检查点方式
                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),  # 使用自定义前向函数
                        hidden_states,  # 当前隐藏状态
                        temb,  # 时间嵌入
                        use_reentrant=False  # 不使用可重入检查点
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),  # 使用自定义前向函数
                        hidden_states,  # 当前隐藏状态
                        temb  # 时间嵌入
                    )
            else:
                # 如果不启用检查点，直接调用 ResNet
                hidden_states = resnet(hidden_states, temb)

        # 如果存在上采样器，则遍历进行上采样
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)  # 调用上采样器

        # 返回最终的隐藏状态
        return hidden_states
# 定义一个 2D 上采样解码块类，继承自 nn.Module
class UpDecoderBlock2D(nn.Module):
    # 初始化方法，接受多个参数用于构造解码块
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        resolution_idx: Optional[int] = None,  # 分辨率索引，默认为 None
        dropout: float = 0.0,  # dropout 概率，默认为 0
        num_layers: int = 1,  # 层数，默认为 1
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 值，默认为 1e-6
        resnet_time_scale_shift: str = "default",  # 时间尺度偏移类型，默认为 "default"
        resnet_act_fn: str = "swish",  # ResNet 的激活函数，默认为 "swish"
        resnet_groups: int = 32,  # ResNet 的分组数，默认为 32
        resnet_pre_norm: bool = True,  # 是否在 ResNet 中预先归一化，默认为 True
        output_scale_factor: float = 1.0,  # 输出缩放因子，默认为 1.0
        add_upsample: bool = True,  # 是否添加上采样层，默认为 True
        temb_channels: Optional[int] = None,  # 时间嵌入通道数，默认为 None
    ):
        # 调用父类构造方法
        super().__init__()
        # 初始化一个空的 ResNet 列表
        resnets = []

        # 根据层数创建相应数量的 ResNet 层
        for i in range(num_layers):
            # 第一个层使用输入通道数，其余层使用输出通道数
            input_channels = in_channels if i == 0 else out_channels

            # 根据时间尺度偏移类型创建不同的 ResNet 块
            if resnet_time_scale_shift == "spatial":
                resnets.append(
                    ResnetBlockCondNorm2D(  # 添加条件归一化的 ResNet 块
                        in_channels=input_channels,  # 输入通道数
                        out_channels=out_channels,  # 输出通道数
                        temb_channels=temb_channels,  # 时间嵌入通道数
                        eps=resnet_eps,  # epsilon 值
                        groups=resnet_groups,  # 分组数
                        dropout=dropout,  # dropout 概率
                        time_embedding_norm="spatial",  # 时间嵌入归一化类型
                        non_linearity=resnet_act_fn,  # 激活函数
                        output_scale_factor=output_scale_factor,  # 输出缩放因子
                    )
                )
            else:
                resnets.append(
                    ResnetBlock2D(  # 添加普通的 ResNet 块
                        in_channels=input_channels,  # 输入通道数
                        out_channels=out_channels,  # 输出通道数
                        temb_channels=temb_channels,  # 时间嵌入通道数
                        eps=resnet_eps,  # epsilon 值
                        groups=resnet_groups,  # 分组数
                        dropout=dropout,  # dropout 概率
                        time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化类型
                        non_linearity=resnet_act_fn,  # 激活函数
                        output_scale_factor=output_scale_factor,  # 输出缩放因子
                        pre_norm=resnet_pre_norm,  # 是否预先归一化
                    )
                )

        # 将创建的 ResNet 块存储在 ModuleList 中，以便于管理
        self.resnets = nn.ModuleList(resnets)

        # 根据是否添加上采样层初始化上采样层列表
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None  # 如果不添加，则设为 None

        # 存储分辨率索引
        self.resolution_idx = resolution_idx

    # 定义前向传播方法
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 遍历所有 ResNet 层进行前向传播
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)  # 更新隐藏状态

        # 如果存在上采样层，则遍历进行上采样
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)  # 更新隐藏状态

        # 返回最终的隐藏状态
        return hidden_states


# 定义一个注意力上采样解码块类，继承自 nn.Module
class AttnUpDecoderBlock2D(nn.Module):
    # 初始化类的构造函数，定义各参数
        def __init__(
            # 输入通道数，决定输入数据的特征维度
            self,
            in_channels: int,
            # 输出通道数，决定输出数据的特征维度
            out_channels: int,
            # 分辨率索引，选择特定分辨率（可选）
            resolution_idx: Optional[int] = None,
            # dropout比率，用于防止过拟合，默认为0
            dropout: float = 0.0,
            # 网络层数，决定模型的深度，默认为1
            num_layers: int = 1,
            # ResNet的epsilon值，防止分母为0的情况，默认为1e-6
            resnet_eps: float = 1e-6,
            # ResNet的时间尺度偏移设置，默认为"default"
            resnet_time_scale_shift: str = "default",
            # ResNet的激活函数类型，默认为"swish"
            resnet_act_fn: str = "swish",
            # ResNet中的组数，影响计算和模型复杂度，默认为32
            resnet_groups: int = 32,
            # 是否在ResNet中使用预归一化，默认为True
            resnet_pre_norm: bool = True,
            # 注意力头的维度，影响注意力机制的计算，默认为1
            attention_head_dim: int = 1,
            # 输出缩放因子，调整输出的大小，默认为1.0
            output_scale_factor: float = 1.0,
            # 是否添加上采样层，影响模型的结构，默认为True
            add_upsample: bool = True,
            # 时间嵌入通道数（可选），用于特定的时间信息表示
            temb_channels: Optional[int] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化存储残差块的列表
        resnets = []
        # 初始化存储注意力层的列表
        attentions = []

        # 如果未指定注意力头维度，则发出警告并使用输出通道数作为默认值
        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `out_channels`: {out_channels}."
            )
            # 将注意力头维度设置为输出通道数
            attention_head_dim = out_channels

        # 遍历每一层以构建残差块和注意力层
        for i in range(num_layers):
            # 如果是第一层，则输入通道为输入通道数，否则为输出通道数
            input_channels = in_channels if i == 0 else out_channels

            # 如果时间尺度偏移为"spatial"，则使用条件归一化的残差块
            if resnet_time_scale_shift == "spatial":
                resnets.append(
                    ResnetBlockCondNorm2D(
                        # 输入通道数
                        in_channels=input_channels,
                        # 输出通道数
                        out_channels=out_channels,
                        # 时间嵌入通道数
                        temb_channels=temb_channels,
                        # 残差块的epsilon参数
                        eps=resnet_eps,
                        # 组归一化的组数
                        groups=resnet_groups,
                        # dropout比例
                        dropout=dropout,
                        # 时间嵌入的归一化方式
                        time_embedding_norm="spatial",
                        # 非线性激活函数
                        non_linearity=resnet_act_fn,
                        # 输出缩放因子
                        output_scale_factor=output_scale_factor,
                    )
                )
            else:
                # 否则使用普通的2D残差块
                resnets.append(
                    ResnetBlock2D(
                        # 输入通道数
                        in_channels=input_channels,
                        # 输出通道数
                        out_channels=out_channels,
                        # 时间嵌入通道数
                        temb_channels=temb_channels,
                        # 残差块的epsilon参数
                        eps=resnet_eps,
                        # 组归一化的组数
                        groups=resnet_groups,
                        # dropout比例
                        dropout=dropout,
                        # 时间嵌入的归一化方式
                        time_embedding_norm=resnet_time_scale_shift,
                        # 非线性激活函数
                        non_linearity=resnet_act_fn,
                        # 输出缩放因子
                        output_scale_factor=output_scale_factor,
                        # 是否使用预归一化
                        pre_norm=resnet_pre_norm,
                    )
                )

            # 添加注意力层
            attentions.append(
                Attention(
                    # 输出通道数
                    out_channels,
                    # 计算注意力头数
                    heads=out_channels // attention_head_dim,
                    # 注意力头维度
                    dim_head=attention_head_dim,
                    # 输出缩放因子
                    rescale_output_factor=output_scale_factor,
                    # epsilon参数
                    eps=resnet_eps,
                    # 组归一化的组数（如果不是空间归一化）
                    norm_num_groups=resnet_groups if resnet_time_scale_shift != "spatial" else None,
                    # 空间归一化维度（如果是空间归一化）
                    spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                    # 是否使用残差连接
                    residual_connection=True,
                    # 是否使用偏置
                    bias=True,
                    # 是否使用上采样的softmax
                    upcast_softmax=True,
                    # 从过时的注意力块创建
                    _from_deprecated_attn_block=True,
                )
            )

        # 将注意力层转换为模块列表
        self.attentions = nn.ModuleList(attentions)
        # 将残差块转换为模块列表
        self.resnets = nn.ModuleList(resnets)

        # 如果需要添加上采样层
        if add_upsample:
            # 创建上采样层的模块列表
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            # 如果不需要上采样，则将其设置为None
            self.upsamplers = None

        # 设置分辨率索引
        self.resolution_idx = resolution_idx
    # 定义前向传播函数，接收隐藏状态和可选的时间嵌入，返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 遍历残差网络和注意力模块的组合
        for resnet, attn in zip(self.resnets, self.attentions):
            # 将当前隐藏状态输入到残差网络中，可能包含时间嵌入
            hidden_states = resnet(hidden_states, temb=temb)
            # 将残差网络的输出输入到注意力模块中，可能包含时间嵌入
            hidden_states = attn(hidden_states, temb=temb)
    
        # 如果上采样模块不为空，则执行上采样操作
        if self.upsamplers is not None:
            # 遍历所有上采样模块
            for upsampler in self.upsamplers:
                # 将当前隐藏状态输入到上采样模块中
                hidden_states = upsampler(hidden_states)
    
        # 返回最终处理后的隐藏状态
        return hidden_states
# 定义一个名为 AttnSkipUpBlock2D 的类，继承自 nn.Module
class AttnSkipUpBlock2D(nn.Module):
    # 初始化方法，定义该类的属性
    def __init__(
        # 输入通道数
        in_channels: int,
        # 前一个输出通道数
        prev_output_channel: int,
        # 输出通道数
        out_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # 可选的分辨率索引
        resolution_idx: Optional[int] = None,
        # dropout 概率
        dropout: float = 0.0,
        # 层数
        num_layers: int = 1,
        # ResNet 的 epsilon 值
        resnet_eps: float = 1e-6,
        # ResNet 时间缩放偏移设置
        resnet_time_scale_shift: str = "default",
        # ResNet 激活函数类型
        resnet_act_fn: str = "swish",
        # 是否使用 ResNet 预归一化
        resnet_pre_norm: bool = True,
        # 注意力头的维度
        attention_head_dim: int = 1,
        # 输出缩放因子
        output_scale_factor: float = np.sqrt(2.0),
        # 是否添加上采样
        add_upsample: bool = True,
    ):
        # 初始化父类
        super().__init__()
        # 此处应有具体的初始化代码（如层的定义），略去

    # 定义前向传播方法
    def forward(
        # 隐藏状态输入
        hidden_states: torch.Tensor,
        # 之前的隐藏状态元组
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        # 可选的时间嵌入
        temb: Optional[torch.Tensor] = None,
        # 可选的跳跃样本
        skip_sample=None,
        # 额外参数
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 检查 args 和 kwargs 是否包含 scale 参数
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 定义弃用信息
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用弃用函数
            deprecate("scale", "1.0.0", deprecation_message)

        # 遍历 ResNet 层
        for resnet in self.resnets:
            # 从元组中提取最近的隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            # 更新隐藏状态元组，去掉最近的隐藏状态
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            # 将当前隐藏状态与提取的隐藏状态拼接在一起
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # 通过 ResNet 层处理隐藏状态
            hidden_states = resnet(hidden_states, temb)

        # 通过注意力层处理隐藏状态
        hidden_states = self.attentions[0](hidden_states)

        # 检查跳跃样本是否为 None
        if skip_sample is not None:
            # 如果不是，则通过上采样层处理跳跃样本
            skip_sample = self.upsampler(skip_sample)
        else:
            # 如果是，则将跳跃样本设为 0
            skip_sample = 0

        # 检查 ResNet 上采样层是否存在
        if self.resnet_up is not None:
            # 通过跳跃归一化层处理隐藏状态
            skip_sample_states = self.skip_norm(hidden_states)
            # 应用激活函数
            skip_sample_states = self.act(skip_sample_states)
            # 通过跳跃卷积层处理状态
            skip_sample_states = self.skip_conv(skip_sample_states)

            # 更新跳跃样本
            skip_sample = skip_sample + skip_sample_states

            # 通过 ResNet 上采样层处理隐藏状态
            hidden_states = self.resnet_up(hidden_states, temb)

        # 返回处理后的隐藏状态和跳跃样本
        return hidden_states, skip_sample


# 定义一个名为 SkipUpBlock2D 的类，继承自 nn.Module
class SkipUpBlock2D(nn.Module):
    # 初始化方法，定义该类的属性
    def __init__(
        # 输入通道数
        in_channels: int,
        # 前一个输出通道数
        prev_output_channel: int,
        # 输出通道数
        out_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # 可选的分辨率索引
        resolution_idx: Optional[int] = None,
        # dropout 概率
        dropout: float = 0.0,
        # 层数
        num_layers: int = 1,
        # ResNet 的 epsilon 值
        resnet_eps: float = 1e-6,
        # ResNet 时间缩放偏移设置
        resnet_time_scale_shift: str = "default",
        # ResNet 激活函数类型
        resnet_act_fn: str = "swish",
        # 是否使用 ResNet 预归一化
        resnet_pre_norm: bool = True,
        # 输出缩放因子
        output_scale_factor: float = np.sqrt(2.0),
        # 是否添加上采样
        add_upsample: bool = True,
        # 上采样填充大小
        upsample_padding: int = 1,
    ):
        # 初始化父类
        super().__init__()
        # 此处应有具体的初始化代码（如层的定义），略去
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个空的 ModuleList 用于存储 ResnetBlock2D 层
        self.resnets = nn.ModuleList([])

        # 根据 num_layers 的数量来添加 ResnetBlock2D 层
        for i in range(num_layers):
            # 计算跳过通道数，如果是最后一层则使用 in_channels，否则使用 out_channels
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 确定当前 ResNet 块的输入通道数
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 向 resnets 列表中添加一个新的 ResnetBlock2D 实例
            self.resnets.append(
                ResnetBlock2D(
                    # 输入通道数为当前层输入通道加跳过通道数
                    in_channels=resnet_in_channels + res_skip_channels,
                    # 输出通道数
                    out_channels=out_channels,
                    # 时间嵌入通道数
                    temb_channels=temb_channels,
                    # 归一化的 epsilon 值
                    eps=resnet_eps,
                    # 分组数为输入通道数的一部分，最多为 32
                    groups=min((resnet_in_channels + res_skip_channels) // 4, 32),
                    # 输出分组数，同样最多为 32
                    groups_out=min(out_channels // 4, 32),
                    # dropout 概率
                    dropout=dropout,
                    # 时间嵌入的归一化方式
                    time_embedding_norm=resnet_time_scale_shift,
                    # 激活函数类型
                    non_linearity=resnet_act_fn,
                    # 输出缩放因子
                    output_scale_factor=output_scale_factor,
                    # 是否使用预归一化
                    pre_norm=resnet_pre_norm,
                )
            )

        # 初始化上采样层
        self.upsampler = FirUpsample2D(in_channels, out_channels=out_channels)
        # 如果需要添加上采样层
        if add_upsample:
            # 添加一个上采样的 ResnetBlock2D
            self.resnet_up = ResnetBlock2D(
                # 输入通道数
                in_channels=out_channels,
                # 输出通道数
                out_channels=out_channels,
                # 时间嵌入通道数
                temb_channels=temb_channels,
                # 归一化的 epsilon 值
                eps=resnet_eps,
                # 分组数，最多为 32
                groups=min(out_channels // 4, 32),
                # 输出分组数，最多为 32
                groups_out=min(out_channels // 4, 32),
                # dropout 概率
                dropout=dropout,
                # 时间嵌入的归一化方式
                time_embedding_norm=resnet_time_scale_shift,
                # 激活函数类型
                non_linearity=resnet_act_fn,
                # 输出缩放因子
                output_scale_factor=output_scale_factor,
                # 是否使用预归一化
                pre_norm=resnet_pre_norm,
                # 是否在快捷路径中使用
                use_in_shortcut=True,
                # 标记为上采样
                up=True,
                # 使用 FIR 卷积核
                kernel="fir",
            )
            # 定义跳过连接的卷积层，将输出通道数映射到 3 通道
            self.skip_conv = nn.Conv2d(out_channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # 定义跳过连接的归一化层
            self.skip_norm = torch.nn.GroupNorm(
                num_groups=min(out_channels // 4, 32), num_channels=out_channels, eps=resnet_eps, affine=True
            )
            # 定义激活函数为 SiLU
            self.act = nn.SiLU()
        else:
            # 如果不添加上采样层，则将相关属性设为 None
            self.resnet_up = None
            self.skip_conv = None
            self.skip_norm = None
            self.act = None

        # 保存分辨率索引
        self.resolution_idx = resolution_idx

    # 定义前向传播方法
    def forward(
        # 定义前向传播输入参数
        hidden_states: torch.Tensor,
        # 存储残差隐藏状态的元组
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        # 可选的时间嵌入张量
        temb: Optional[torch.Tensor] = None,
        # 跳过采样的可选参数
        skip_sample=None,
        # 可变位置参数
        *args,
        # 可变关键字参数
        **kwargs,
    # 函数返回两个张量，表示隐藏状态和跳过的样本
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 检查参数长度或关键字参数 "scale" 是否存在
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 创建弃用消息，提醒用户 "scale" 参数将被忽略
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用弃用函数，传递参数信息
            deprecate("scale", "1.0.0", deprecation_message)
    
        # 遍历自定义的 ResNet 模块
        for resnet in self.resnets:
            # 从隐藏状态元组中弹出最后一个 ResNet 隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            # 更新隐藏状态元组，去掉最后一个元素
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            # 将当前的隐藏状态与 ResNet 隐藏状态连接
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
    
            # 通过 ResNet 模块更新隐藏状态
            hidden_states = resnet(hidden_states, temb)
    
        # 检查跳过样本是否存在
        if skip_sample is not None:
            # 如果存在，使用上采样器处理跳过样本
            skip_sample = self.upsampler(skip_sample)
        else:
            # 否则，将跳过样本初始化为 0
            skip_sample = 0
    
        # 检查是否存在 ResNet 上采样模块
        if self.resnet_up is not None:
            # 对隐藏状态应用归一化
            skip_sample_states = self.skip_norm(hidden_states)
            # 对归一化结果应用激活函数
            skip_sample_states = self.act(skip_sample_states)
            # 对激活结果应用卷积操作
            skip_sample_states = self.skip_conv(skip_sample_states)
    
            # 将跳过样本与处理后的状态相加
            skip_sample = skip_sample + skip_sample_states
    
            # 通过 ResNet 上采样模块更新隐藏状态
            hidden_states = self.resnet_up(hidden_states, temb)
    
        # 返回最终的隐藏状态和跳过样本
        return hidden_states, skip_sample
# 定义一个 2D 上采样的 ResNet 块，继承自 nn.Module
class ResnetUpsampleBlock2D(nn.Module):
    # 初始化方法，设置网络参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        prev_output_channel: int,  # 前一层输出的通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入的通道数
        resolution_idx: Optional[int] = None,  # 分辨率索引（可选）
        dropout: float = 0.0,  # dropout 比例
        num_layers: int = 1,  # ResNet 层数
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 值
        resnet_time_scale_shift: str = "default",  # 时间缩放偏移方式
        resnet_act_fn: str = "swish",  # 激活函数类型
        resnet_groups: int = 32,  # 组数
        resnet_pre_norm: bool = True,  # 是否使用预归一化
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_upsample: bool = True,  # 是否添加上采样层
        skip_time_act: bool = False,  # 是否跳过时间激活
    ):
        # 调用父类构造函数
        super().__init__()
        # 初始化一个空的 ResNet 列表
        resnets = []

        # 遍历层数，创建每一层的 ResNet 块
        for i in range(num_layers):
            # 确定跳过通道数，如果是最后一层，则使用输入通道数，否则使用输出通道数
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 确定当前层的输入通道数
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 将 ResNet 块添加到列表中
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # epsilon 值
                    groups=resnet_groups,  # 组数
                    dropout=dropout,  # dropout 比例
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化方式
                    non_linearity=resnet_act_fn,  # 非线性激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 是否使用预归一化
                    skip_time_act=skip_time_act,  # 是否跳过时间激活
                )
            )

        # 将 ResNet 列表转换为 nn.ModuleList，便于 PyTorch 管理
        self.resnets = nn.ModuleList(resnets)

        # 如果需要添加上采样层，则创建上采样模块
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,  # 输入通道数
                        out_channels=out_channels,  # 输出通道数
                        temb_channels=temb_channels,  # 时间嵌入通道数
                        eps=resnet_eps,  # epsilon 值
                        groups=resnet_groups,  # 组数
                        dropout=dropout,  # dropout 比例
                        time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化方式
                        non_linearity=resnet_act_fn,  # 非线性激活函数
                        output_scale_factor=output_scale_factor,  # 输出缩放因子
                        pre_norm=resnet_pre_norm,  # 是否使用预归一化
                        skip_time_act=skip_time_act,  # 是否跳过时间激活
                        up=True,  # 标记为上采样块
                    )
                ]
            )
        else:
            # 如果不需要上采样，则将其设为 None
            self.upsamplers = None

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False
        # 设置分辨率索引
        self.resolution_idx = resolution_idx

    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],  # 额外的隐藏状态元组
        temb: Optional[torch.Tensor] = None,  # 时间嵌入（可选）
        upsample_size: Optional[int] = None,  # 上采样大小（可选）
        *args,  # 额外的位置参数
        **kwargs,  # 额外的关键字参数
    ) -> torch.Tensor:  # 定义一个返回 torch.Tensor 的函数
        # 检查参数列表是否包含参数或 "scale" 是否不为 None
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 定义弃用信息，说明 "scale" 参数将被忽略
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用 deprecate 函数以记录 "scale" 参数的弃用
            deprecate("scale", "1.0.0", deprecation_message)

        # 遍历存储的 ResNet 模型列表
        for resnet in self.resnets:
            # 从隐藏状态元组中弹出最后一个 ResNet 隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            # 更新隐藏状态元组，去掉最后一个元素
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            # 将当前的隐藏状态与 ResNet 隐藏状态在指定维度上拼接
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # 如果处于训练模式且开启了梯度检查点
            if self.training and self.gradient_checkpointing:

                # 定义一个创建自定义前向传播函数的内部函数
                def create_custom_forward(module):
                    # 定义自定义前向传播函数，调用模块
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                # 检查 PyTorch 版本是否大于等于 1.11.0
                if is_torch_version(">=", "1.11.0"):
                    # 使用梯度检查点功能进行前向传播
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    # 使用梯度检查点功能进行前向传播，不使用可重入选项
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                # 直接调用 ResNet 模型进行前向传播
                hidden_states = resnet(hidden_states, temb)

        # 如果存在上采样器
        if self.upsamplers is not None:
            # 遍历每个上采样器
            for upsampler in self.upsamplers:
                # 调用上采样器进行上采样处理
                hidden_states = upsampler(hidden_states, temb)

        # 返回最终的隐藏状态
        return hidden_states
# 定义一个简单的二维交叉注意力上采样模块，继承自 nn.Module
class SimpleCrossAttnUpBlock2D(nn.Module):
    # 初始化函数，接受多个参数来配置模块
    def __init__(
        # 输入通道数
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 上一个输出通道数
        prev_output_channel: int,
        # 时间嵌入通道数
        temb_channels: int,
        # 可选的分辨率索引
        resolution_idx: Optional[int] = None,
        # Dropout 概率
        dropout: float = 0.0,
        # 层数
        num_layers: int = 1,
        # ResNet 的 epsilon 值
        resnet_eps: float = 1e-6,
        # ResNet 时间缩放偏移
        resnet_time_scale_shift: str = "default",
        # ResNet 激活函数
        resnet_act_fn: str = "swish",
        # ResNet 中的组数
        resnet_groups: int = 32,
        # 是否在 ResNet 中使用预归一化
        resnet_pre_norm: bool = True,
        # 注意力头的维度
        attention_head_dim: int = 1,
        # 交叉注意力的维度
        cross_attention_dim: int = 1280,
        # 输出缩放因子
        output_scale_factor: float = 1.0,
        # 是否添加上采样
        add_upsample: bool = True,
        # 是否跳过时间激活
        skip_time_act: bool = False,
        # 是否仅使用交叉注意力
        only_cross_attention: bool = False,
        # 可选的交叉注意力归一化方式
        cross_attention_norm: Optional[str] = None,
    # 初始化函数
        ):
            # 调用父类的初始化方法
            super().__init__()
            # 创建一个空列表用于存储 ResNet 模块
            resnets = []
            # 创建一个空列表用于存储 Attention 模块
            attentions = []
    
            # 设置是否使用交叉注意力
            self.has_cross_attention = True
            # 设置每个注意力头的维度
            self.attention_head_dim = attention_head_dim
    
            # 计算注意力头的数量
            self.num_heads = out_channels // self.attention_head_dim
    
            # 遍历每一层以构建 ResNet 模块
            for i in range(num_layers):
                # 设置跳跃连接通道数
                res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
                # 设置当前 ResNet 输入通道数
                resnet_in_channels = prev_output_channel if i == 0 else out_channels
    
                # 添加 ResNet 块到列表
                resnets.append(
                    ResnetBlock2D(
                        # 设置输入通道数为 ResNet 输入通道加上跳跃连接通道
                        in_channels=resnet_in_channels + res_skip_channels,
                        # 设置输出通道数
                        out_channels=out_channels,
                        # 设置时间嵌入通道数
                        temb_channels=temb_channels,
                        # 设置小常数用于数值稳定性
                        eps=resnet_eps,
                        # 设置分组数
                        groups=resnet_groups,
                        # 设置 dropout 比例
                        dropout=dropout,
                        # 设置时间嵌入的归一化方式
                        time_embedding_norm=resnet_time_scale_shift,
                        # 设置激活函数
                        non_linearity=resnet_act_fn,
                        # 设置输出缩放因子
                        output_scale_factor=output_scale_factor,
                        # 设置是否预归一化
                        pre_norm=resnet_pre_norm,
                        # 设置是否跳过时间激活
                        skip_time_act=skip_time_act,
                    )
                )
    
                # 根据是否支持缩放点积注意力选择处理器
                processor = (
                    AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
                )
    
                # 添加 Attention 模块到列表
                attentions.append(
                    Attention(
                        # 设置查询维度
                        query_dim=out_channels,
                        # 设置交叉注意力维度
                        cross_attention_dim=out_channels,
                        # 设置头的数量
                        heads=self.num_heads,
                        # 设置每个头的维度
                        dim_head=self.attention_head_dim,
                        # 设置额外的 KV 投影维度
                        added_kv_proj_dim=cross_attention_dim,
                        # 设置归一化的组数
                        norm_num_groups=resnet_groups,
                        # 设置是否使用偏置
                        bias=True,
                        # 设置是否上溯 softmax
                        upcast_softmax=True,
                        # 设置是否仅使用交叉注意力
                        only_cross_attention=only_cross_attention,
                        # 设置交叉注意力的归一化方式
                        cross_attention_norm=cross_attention_norm,
                        # 设置处理器
                        processor=processor,
                    )
                )
            # 将 Attention 模块列表转换为 ModuleList
            self.attentions = nn.ModuleList(attentions)
            # 将 ResNet 模块列表转换为 ModuleList
            self.resnets = nn.ModuleList(resnets)
    
            # 如果需要添加上采样模块
            if add_upsample:
                # 创建一个上采样的 ResNet 模块列表
                self.upsamplers = nn.ModuleList(
                    [
                        ResnetBlock2D(
                            # 设置上采样的输入和输出通道数
                            in_channels=out_channels,
                            out_channels=out_channels,
                            # 设置时间嵌入通道数
                            temb_channels=temb_channels,
                            # 设置小常数用于数值稳定性
                            eps=resnet_eps,
                            # 设置分组数
                            groups=resnet_groups,
                            # 设置 dropout 比例
                            dropout=dropout,
                            # 设置时间嵌入的归一化方式
                            time_embedding_norm=resnet_time_scale_shift,
                            # 设置激活函数
                            non_linearity=resnet_act_fn,
                            # 设置输出缩放因子
                            output_scale_factor=output_scale_factor,
                            # 设置是否预归一化
                            pre_norm=resnet_pre_norm,
                            # 设置是否跳过时间激活
                            skip_time_act=skip_time_act,
                            # 设置为上采样模式
                            up=True,
                        )
                    ]
                )
            else:
                # 如果不需要上采样，则将上采样模块设置为 None
                self.upsamplers = None
    
            # 初始化梯度检查点设置为 False
            self.gradient_checkpointing = False
            # 设置分辨率索引
            self.resolution_idx = resolution_idx
    # 定义前向传播方法，接收多个输入参数
        def forward(
            self,
            # 当前隐藏状态的张量
            hidden_states: torch.Tensor,
            # 包含残差隐藏状态的元组
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            # 可选的时间嵌入张量
            temb: Optional[torch.Tensor] = None,
            # 可选的编码器隐藏状态张量
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选的上采样大小
            upsample_size: Optional[int] = None,
            # 可选的注意力掩码张量
            attention_mask: Optional[torch.Tensor] = None,
            # 可选的跨注意力参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的编码器注意力掩码张量
            encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # 定义函数的返回类型为 torch.Tensor
        # 如果 cross_attention_kwargs 为 None，则初始化为空字典
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        # 如果 cross_attention_kwargs 中的 "scale" 存在，发出警告，提示已弃用
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # 如果 attention_mask 为 None
        if attention_mask is None:
            # 如果 encoder_hidden_states 已定义，则进行交叉注意力，使用 encoder_attention_mask
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            # 如果 attention_mask 已定义，不检查 encoder_attention_mask
            # 这样做是为了兼容 UnCLIP，后者使用 'attention_mask' 参数作为交叉注意力掩码
            # TODO: UnCLIP 应该通过 encoder_attention_mask 参数表达交叉注意力掩码，而不是通过 attention_mask
            #       那样可以简化整个 if/else 语句块
            mask = attention_mask  # 使用提供的 attention_mask

        # 遍历 self.resnets 和 self.attentions 的元素
        for resnet, attn in zip(self.resnets, self.attentions):
            # 获取最后一项的残差隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            # 更新 res_hidden_states_tuple，去掉最后一项
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            # 将当前的 hidden_states 和残差隐藏状态在维度 1 上连接
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # 如果处于训练模式并且开启了梯度检查点
            if self.training and self.gradient_checkpointing:

                # 定义创建自定义前向传播函数的内部函数
                def create_custom_forward(module, return_dict=None):
                    # 定义自定义前向传播函数
                    def custom_forward(*inputs):
                        # 如果 return_dict 不为 None，则返回字典形式的结果
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)  # 否则直接返回结果

                    return custom_forward  # 返回自定义前向传播函数

                # 使用检查点机制进行前向传播，节省内存
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                # 进行注意力计算
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=mask,
                    **cross_attention_kwargs,
                )
            else:
                # 直接进行前向传播计算
                hidden_states = resnet(hidden_states, temb)

                # 进行注意力计算
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=mask,
                    **cross_attention_kwargs,
                )

        # 如果存在上采样器
        if self.upsamplers is not None:
            # 遍历所有上采样器
            for upsampler in self.upsamplers:
                # 进行上采样操作
                hidden_states = upsampler(hidden_states, temb)

        # 返回最终的隐藏状态
        return hidden_states
# 定义一个名为 KUpBlock2D 的神经网络模块，继承自 nn.Module
class KUpBlock2D(nn.Module):
    # 初始化函数，设置网络层的参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        resolution_idx: int,  # 分辨率索引
        dropout: float = 0.0,  # dropout 比例，默认 0
        num_layers: int = 5,  # 网络层数，默认 5
        resnet_eps: float = 1e-5,  # ResNet 中的 epsilon 值
        resnet_act_fn: str = "gelu",  # ResNet 中使用的激活函数，默认 "gelu"
        resnet_group_size: Optional[int] = 32,  # ResNet 中的组大小，默认 32
        add_upsample: bool = True,  # 是否添加上采样层，默认 True
    ):
        # 调用父类初始化函数
        super().__init__()
        # 创建一个空的列表，用于存放 ResNet 模块
        resnets = []
        # 定义输入通道的数量，设置为输出通道的两倍
        k_in_channels = 2 * out_channels
        # 定义输出通道的数量
        k_out_channels = in_channels
        # 减少层数，以适应后续的循环
        num_layers = num_layers - 1

        # 创建指定层数的 ResNet 模块
        for i in range(num_layers):
            # 第一层的输入通道为 k_in_channels，其余层为 out_channels
            in_channels = k_in_channels if i == 0 else out_channels
            # 计算组的数量
            groups = in_channels // resnet_group_size
            # 计算输出组的数量
            groups_out = out_channels // resnet_group_size

            # 将 ResNet 模块添加到列表中
            resnets.append(
                ResnetBlockCondNorm2D(
                    in_channels=in_channels,  # 输入通道数
                    out_channels=k_out_channels if (i == num_layers - 1) else out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # epsilon 值
                    groups=groups,  # 输入组数量
                    groups_out=groups_out,  # 输出组数量
                    dropout=dropout,  # dropout 比例
                    non_linearity=resnet_act_fn,  # 激活函数
                    time_embedding_norm="ada_group",  # 时间嵌入规范化方式
                    conv_shortcut_bias=False,  # 是否使用卷积快捷连接的偏置
                )
            )

        # 将 ResNet 模块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

        # 根据是否添加上采样层来初始化上采样模块
        if add_upsample:
            # 如果添加上采样，创建上采样层列表
            self.upsamplers = nn.ModuleList([KUpsample2D()])
        else:
            # 如果不添加上采样，设置为 None
            self.upsamplers = None

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False
        # 存储分辨率索引
        self.resolution_idx = resolution_idx

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],  # 传入的隐藏状态元组
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        upsample_size: Optional[int] = None,  # 可选的上采样大小
        *args,  # 额外的位置参数
        **kwargs,  # 额外的关键字参数
    # 定义返回值为 torch.Tensor 的函数结束部分
        ) -> torch.Tensor:
            # 检查传入参数是否包含 args 或者 kwargs 中的 "scale" 参数
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                # 定义弃用的提示信息，告知用户应删除 "scale" 参数
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                # 调用 deprecate 函数记录弃用信息
                deprecate("scale", "1.0.0", deprecation_message)
    
            # 取 res_hidden_states_tuple 的最后一个元素
            res_hidden_states_tuple = res_hidden_states_tuple[-1]
            # 如果 res_hidden_states_tuple 不为 None，则将其与 hidden_states 拼接
            if res_hidden_states_tuple is not None:
                hidden_states = torch.cat([hidden_states, res_hidden_states_tuple], dim=1)
    
            # 遍历 self.resnets 列表中的每个 resnet 模块
            for resnet in self.resnets:
                # 如果处于训练模式且开启梯度检查点功能
                if self.training and self.gradient_checkpointing:
    
                    # 定义一个创建自定义前向函数的内部函数
                    def create_custom_forward(module):
                        # 定义自定义前向函数，调用模块处理输入
                        def custom_forward(*inputs):
                            return module(*inputs)
    
                        return custom_forward
    
                    # 检查 PyTorch 版本是否大于等于 1.11.0
                    if is_torch_version(">=", "1.11.0"):
                        # 使用检查点功能进行前向传播，避免计算图保存
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                        )
                    else:
                        # 使用检查点功能进行前向传播
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb
                        )
                else:
                    # 在非训练模式下直接通过 resnet 处理 hidden_states
                    hidden_states = resnet(hidden_states, temb)
    
            # 如果存在 upsamplers，则遍历每个 upsampler
            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    # 通过 upsampler 处理 hidden_states
                    hidden_states = upsampler(hidden_states)
    
            # 返回处理后的 hidden_states
            return hidden_states
# 定义一个 KCrossAttnUpBlock2D 类，继承自 nn.Module
class KCrossAttnUpBlock2D(nn.Module):
    # 初始化方法，定义该类的属性
    def __init__(
        # 输入通道数
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 额外的嵌入通道数
        temb_channels: int,
        # 当前分辨率索引
        resolution_idx: int,
        # dropout 概率，默认为 0.0
        dropout: float = 0.0,
        # 残差网络的层数，默认为 4
        num_layers: int = 4,
        # 残差网络的 epsilon 值，默认为 1e-5
        resnet_eps: float = 1e-5,
        # 残差网络的激活函数类型，默认为 "gelu"
        resnet_act_fn: str = "gelu",
        # 残差网络的分组大小，默认为 32
        resnet_group_size: int = 32,
        # 注意力的维度，默认为 1
        attention_head_dim: int = 1,  # attention dim_head
        # 交叉注意力的维度，默认为 768
        cross_attention_dim: int = 768,
        # 是否添加上采样，默认为 True
        add_upsample: bool = True,
        # 是否上溢注意力，默认为 False
        upcast_attention: bool = False,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化一个空列表，用于存储 ResNet 块
        resnets = []
        # 初始化一个空列表，用于存储注意力块
        attentions = []

        # 判断是否为第一个块：输入、输出和时间嵌入通道是否相等
        is_first_block = in_channels == out_channels == temb_channels
        # 判断是否为中间块：输入和输出通道是否不相等
        is_middle_block = in_channels != out_channels
        # 如果是第一个块，设置为 True 以添加自注意力
        add_self_attention = True if is_first_block else False

        # 设置跨注意力的标志为 True
        self.has_cross_attention = True
        # 存储注意力头的维度
        self.attention_head_dim = attention_head_dim

        # 定义当前块的输入通道，若是第一个块则使用输出通道，否则使用两倍的输出通道
        k_in_channels = out_channels if is_first_block else 2 * out_channels
        # 当前块的输出通道为输入通道
        k_out_channels = in_channels

        # 减少层数以计算循环中的层数
        num_layers = num_layers - 1

        # 根据层数循环创建 ResNet 块和注意力块
        for i in range(num_layers):
            # 第一个层使用 k_in_channels，后续层使用 out_channels
            in_channels = k_in_channels if i == 0 else out_channels
            # 计算组数，以便在 ResNet 中分组
            groups = in_channels // resnet_group_size
            groups_out = out_channels // resnet_group_size

            # 判断是否为中间块并且是最后一层，设置卷积的输出通道
            if is_middle_block and (i == num_layers - 1):
                conv_2d_out_channels = k_out_channels
            else:
                # 如果不是，设置为 None
                conv_2d_out_channels = None

            # 创建并添加 ResNet 块到 resnets 列表
            resnets.append(
                ResnetBlockCondNorm2D(
                    # 输入通道
                    in_channels=in_channels,
                    # 输出通道
                    out_channels=out_channels,
                    # 卷积输出通道
                    conv_2d_out_channels=conv_2d_out_channels,
                    # 时间嵌入通道
                    temb_channels=temb_channels,
                    # 设定 epsilon 值
                    eps=resnet_eps,
                    # 输入组数
                    groups=groups,
                    # 输出组数
                    groups_out=groups_out,
                    # dropout 概率
                    dropout=dropout,
                    # 非线性激活函数
                    non_linearity=resnet_act_fn,
                    # 时间嵌入的归一化方式
                    time_embedding_norm="ada_group",
                    # 是否使用卷积快捷连接的偏置
                    conv_shortcut_bias=False,
                )
            )
            # 创建并添加注意力块到 attentions 列表
            attentions.append(
                KAttentionBlock(
                    # 最后一个层使用 k_out_channels，否则使用 out_channels
                    k_out_channels if (i == num_layers - 1) else out_channels,
                    # 最后一个层注意力维度
                    k_out_channels // attention_head_dim
                    if (i == num_layers - 1)
                    else out_channels // attention_head_dim,
                    # 注意力头的维度
                    attention_head_dim,
                    # 跨注意力维度
                    cross_attention_dim=cross_attention_dim,
                    # 时间嵌入通道
                    temb_channels=temb_channels,
                    # 是否添加注意力偏置
                    attention_bias=True,
                    # 是否添加自注意力
                    add_self_attention=add_self_attention,
                    # 跨注意力归一化方式
                    cross_attention_norm="layer_norm",
                    # 是否上溯注意力
                    upcast_attention=upcast_attention,
                )
            )

        # 将 ResNet 块列表转为 PyTorch 的 ModuleList
        self.resnets = nn.ModuleList(resnets)
        # 将注意力块列表转为 PyTorch 的 ModuleList
        self.attentions = nn.ModuleList(attentions)

        # 如果需要上采样，则创建一个包含上采样块的 ModuleList
        if add_upsample:
            self.upsamplers = nn.ModuleList([KUpsample2D()])
        else:
            # 否则将上采样块设置为 None
            self.upsamplers = None

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False
        # 存储当前分辨率的索引
        self.resolution_idx = resolution_idx
    # 定义前向传播函数，接受多个输入参数，返回处理后的张量
        def forward(
            self,
            hidden_states: torch.Tensor,  # 输入的隐藏状态张量
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],  # 先前隐藏状态的元组
            temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
            encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 交叉注意力的可选参数
            upsample_size: Optional[int] = None,  # 可选的上采样尺寸
            attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
            encoder_attention_mask: Optional[torch.Tensor] = None,  # 可选的编码器注意力掩码
        ) -> torch.Tensor:  # 函数返回类型为张量
            res_hidden_states_tuple = res_hidden_states_tuple[-1]  # 获取最后一个隐藏状态
            if res_hidden_states_tuple is not None:  # 检查是否存在先前的隐藏状态
                hidden_states = torch.cat([hidden_states, res_hidden_states_tuple], dim=1)  # 拼接当前和先前的隐藏状态
    
            for resnet, attn in zip(self.resnets, self.attentions):  # 遍历每个残差网络和注意力层
                if self.training and self.gradient_checkpointing:  # 检查是否在训练模式且使用梯度检查点
    
                    def create_custom_forward(module, return_dict=None):  # 定义创建自定义前向函数的内部函数
                        def custom_forward(*inputs):  # 自定义前向函数
                            if return_dict is not None:  # 检查是否需要返回字典
                                return module(*inputs, return_dict=return_dict)  # 使用返回字典的方式调用模块
                            else:
                                return module(*inputs)  # 普通调用模块
    
                        return custom_forward  # 返回自定义前向函数
    
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}  # 根据Torch版本设置检查点参数
                    hidden_states = torch.utils.checkpoint.checkpoint(  # 使用检查点进行前向传播以节省内存
                        create_custom_forward(resnet),  # 创建自定义前向函数
                        hidden_states,  # 输入隐藏状态
                        temb,  # 输入时间嵌入
                        **ckpt_kwargs,  # 传递检查点参数
                    )
                    hidden_states = attn(  # 通过注意力层处理隐藏状态
                        hidden_states,  # 输入隐藏状态
                        encoder_hidden_states=encoder_hidden_states,  # 输入编码器隐藏状态
                        emb=temb,  # 输入时间嵌入
                        attention_mask=attention_mask,  # 输入注意力掩码
                        cross_attention_kwargs=cross_attention_kwargs,  # 交叉注意力参数
                        encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码
                    )
                else:  # 如果不使用梯度检查点
                    hidden_states = resnet(hidden_states, temb)  # 直接通过残差网络处理隐藏状态
                    hidden_states = attn(  # 通过注意力层处理隐藏状态
                        hidden_states,  # 输入隐藏状态
                        encoder_hidden_states=encoder_hidden_states,  # 输入编码器隐藏状态
                        emb=temb,  # 输入时间嵌入
                        attention_mask=attention_mask,  # 输入注意力掩码
                        cross_attention_kwargs=cross_attention_kwargs,  # 交叉注意力参数
                        encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码
                    )
    
            if self.upsamplers is not None:  # 检查是否有上采样层
                for upsampler in self.upsamplers:  # 遍历每个上采样层
                    hidden_states = upsampler(hidden_states)  # 通过上采样层处理隐藏状态
    
            return hidden_states  # 返回处理后的隐藏状态
# 可以潜在地更名为 `No-feed-forward` 注意力
class KAttentionBlock(nn.Module):
    r"""
    基本的 Transformer 块。

    参数：
        dim (`int`): 输入和输出的通道数。
        num_attention_heads (`int`): 用于多头注意力的头数。
        attention_head_dim (`int`): 每个头的通道数。
        dropout (`float`, *可选*, 默认为 0.0): 使用的丢弃概率。
        cross_attention_dim (`int`, *可选*): 用于交叉注意力的 encoder_hidden_states 向量的大小。
        attention_bias (`bool`, *可选*, 默认为 `False`):
            配置注意力层是否应该包含偏置参数。
        upcast_attention (`bool`, *可选*, 默认为 `False`):
            设置为 `True` 以将注意力计算上调为 `float32`。
        temb_channels (`int`, *可选*, 默认为 768):
            令牌嵌入中的通道数。
        add_self_attention (`bool`, *可选*, 默认为 `False`):
            设置为 `True` 以将自注意力添加到块中。
        cross_attention_norm (`str`, *可选*, 默认为 `None`):
            用于交叉注意力的规范化类型。可以是 `None`、`layer_norm` 或 `group_norm`。
        group_size (`int`, *可选*, 默认为 32):
            用于组规范化将通道分成的组数。
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        upcast_attention: bool = False,
        temb_channels: int = 768,  # 用于 ada_group_norm
        add_self_attention: bool = False,
        cross_attention_norm: Optional[str] = None,
        group_size: int = 32,
    ):
        # 调用父类构造函数，初始化 nn.Module
        super().__init__()
        # 设置是否添加自注意力的标志
        self.add_self_attention = add_self_attention

        # 1. 自注意力
        if add_self_attention:
            # 初始化自注意力的归一化层
            self.norm1 = AdaGroupNorm(temb_channels, dim, max(1, dim // group_size))
            # 初始化自注意力机制
            self.attn1 = Attention(
                query_dim=dim,  # 查询向量的维度
                heads=num_attention_heads,  # 注意力头数
                dim_head=attention_head_dim,  # 每个头的维度
                dropout=dropout,  # 丢弃率
                bias=attention_bias,  # 是否使用偏置
                cross_attention_dim=None,  # 交叉注意力维度
                cross_attention_norm=None,  # 交叉注意力的归一化
            )

        # 2. 交叉注意力
        # 初始化交叉注意力的归一化层
        self.norm2 = AdaGroupNorm(temb_channels, dim, max(1, dim // group_size))
        # 初始化交叉注意力机制
        self.attn2 = Attention(
            query_dim=dim,  # 查询向量的维度
            cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
            heads=num_attention_heads,  # 注意力头数
            dim_head=attention_head_dim,  # 每个头的维度
            dropout=dropout,  # 丢弃率
            bias=attention_bias,  # 是否使用偏置
            upcast_attention=upcast_attention,  # 是否上调注意力计算
            cross_attention_norm=cross_attention_norm,  # 交叉注意力的归一化
        )
    # 将隐藏状态转换为 3D 张量，包含 batch size, height*weight 和通道数
    def _to_3d(self, hidden_states: torch.Tensor, height: int, weight: int) -> torch.Tensor:
        # 重新排列维度，并调整形状为 (batch size, height*weight, -1)
        return hidden_states.permute(0, 2, 3, 1).reshape(hidden_states.shape[0], height * weight, -1)

    # 将隐藏状态转换为 4D 张量，包含 batch size, 通道数, height 和 weight
    def _to_4d(self, hidden_states: torch.Tensor, height: int, weight: int) -> torch.Tensor:
        # 重新排列维度，并调整形状为 (batch size, -1, height, weight)
        return hidden_states.permute(0, 2, 1).reshape(hidden_states.shape[0], -1, height, weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # TODO: 将 emb 标记为非可选 (self.norm2 需要它)。
        #       需要评估对位置参数接口更改的影响。
        emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 如果 cross_attention_kwargs 为空，则初始化为一个空字典
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        # 检查 "scale" 参数是否存在，如果存在则发出警告
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # 1. 自注意力
        if self.add_self_attention:
            # 使用 norm1 对隐藏状态进行归一化处理
            norm_hidden_states = self.norm1(hidden_states, emb)

            # 获取归一化后状态的高度和宽度
            height, weight = norm_hidden_states.shape[2:]
            # 将归一化后的隐藏状态转换为 3D 张量
            norm_hidden_states = self._to_3d(norm_hidden_states, height, weight)

            # 执行自注意力操作
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            # 将自注意力输出转换为 4D 张量
            attn_output = self._to_4d(attn_output, height, weight)

            # 将自注意力输出与原始隐藏状态相加
            hidden_states = attn_output + hidden_states

        # 2. 交叉注意力或无交叉注意力
        # 使用 norm2 对隐藏状态进行归一化处理
        norm_hidden_states = self.norm2(hidden_states, emb)

        # 获取归一化后状态的高度和宽度
        height, weight = norm_hidden_states.shape[2:]
        # 将归一化后的隐藏状态转换为 3D 张量
        norm_hidden_states = self._to_3d(norm_hidden_states, height, weight)
        # 执行交叉注意力操作
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask if encoder_hidden_states is None else encoder_attention_mask,
            **cross_attention_kwargs,
        )
        # 将交叉注意力输出转换为 4D 张量
        attn_output = self._to_4d(attn_output, height, weight)

        # 将交叉注意力输出与隐藏状态相加
        hidden_states = attn_output + hidden_states

        # 返回最终的隐藏状态
        return hidden_states
```