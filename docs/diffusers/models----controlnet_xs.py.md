# `.\diffusers\models\controlnet_xs.py`

```
# 版权信息，声明该文件归 HuggingFace 团队所有，所有权利保留
# 
# 根据 Apache 许可证第 2.0 版（"许可证"）进行授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件是按“原样”基础进行分发，
# 不提供任何形式的担保或条件，无论是明示或暗示的。
# 请参阅许可证以获取有关权限和限制的特定信息。
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from math import gcd  # 从 math 模块导入 gcd 函数，用于计算最大公约数
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的类型

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 工具，用于保存内存
from torch import Tensor, nn  # 从 torch 模块导入 Tensor 类和 nn 模块

from ..configuration_utils import ConfigMixin, register_to_config  # 从上层模块导入配置相关的类和函数
from ..utils import BaseOutput, is_torch_version, logging  # 从上层模块导入工具类和函数
from ..utils.torch_utils import apply_freeu  # 从上层模块导入特定的 PyTorch 工具函数
from .attention_processor import (  # 从当前包导入注意力处理器相关的类
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from .controlnet import ControlNetConditioningEmbedding  # 从当前包导入 ControlNet 的条件嵌入类
from .embeddings import TimestepEmbedding, Timesteps  # 从当前包导入时间步嵌入相关的类
from .modeling_utils import ModelMixin  # 从当前包导入模型混合类
from .unets.unet_2d_blocks import (  # 从当前包导入 2D U-Net 模块相关的类
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    Downsample2D,
    ResnetBlock2D,
    Transformer2DModel,
    UNetMidBlock2DCrossAttn,
    Upsample2D,
)
from .unets.unet_2d_condition import UNet2DConditionModel  # 从当前包导入带条件的 2D U-Net 模型类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


@dataclass  # 将该类声明为数据类
class ControlNetXSOutput(BaseOutput):  # 定义 ControlNetXSOutput 类，继承自 BaseOutput
    """
    [`UNetControlNetXSModel`] 的输出。

    参数：
        sample (`Tensor`，形状为 `(batch_size, num_channels, height, width)`):
            `UNetControlNetXSModel` 的输出。与 `ControlNetOutput` 不同，此输出不是要与基础模型输出相加，而是已经是最终输出。
    """

    sample: Tensor = None  # 定义一个可选的 Tensor 属性 sample，默认为 None


class DownBlockControlNetXSAdapter(nn.Module):  # 定义 DownBlockControlNetXSAdapter 类，继承自 nn.Module
    """与基础模型的对应组件一起形成 `ControlNetXSCrossAttnDownBlock2D` 的组件"""

    def __init__(  # 定义初始化方法
        self,
        resnets: nn.ModuleList,  # 传入一个 ResNet 组件的模块列表
        base_to_ctrl: nn.ModuleList,  # 传入基础模型到 ControlNet 的模块列表
        ctrl_to_base: nn.ModuleList,  # 传入 ControlNet 到基础模型的模块列表
        attentions: Optional[nn.ModuleList] = None,  # 可选的注意力模块列表，默认为 None
        downsampler: Optional[nn.Conv2d] = None,  # 可选的下采样模块，默认为 None
    ):
        super().__init__()  # 调用父类的初始化方法
        self.resnets = resnets  # 保存 ResNet 组件列表
        self.base_to_ctrl = base_to_ctrl  # 保存基础模型到 ControlNet 的模块列表
        self.ctrl_to_base = ctrl_to_base  # 保存 ControlNet 到基础模型的模块列表
        self.attentions = attentions  # 保存注意力模块列表
        self.downsamplers = downsampler  # 保存下采样模块


class MidBlockControlNetXSAdapter(nn.Module):  # 定义 MidBlockControlNetXSAdapter 类，继承自 nn.Module
    """与基础模型的对应组件一起形成 `ControlNetXSCrossAttnMidBlock2D` 的组件"""
    # 初始化类的构造函数
        def __init__(self, midblock: UNetMidBlock2DCrossAttn, base_to_ctrl: nn.ModuleList, ctrl_to_base: nn.ModuleList):
            # 调用父类的构造函数
            super().__init__()
            # 将传入的 midblock 参数赋值给实例变量 midblock
            self.midblock = midblock
            # 将传入的 base_to_ctrl 参数赋值给实例变量 base_to_ctrl
            self.base_to_ctrl = base_to_ctrl
            # 将传入的 ctrl_to_base 参数赋值给实例变量 ctrl_to_base
            self.ctrl_to_base = ctrl_to_base
# 定义一个名为 UpBlockControlNetXSAdapter 的类，继承自 nn.Module
class UpBlockControlNetXSAdapter(nn.Module):
    """与基础模型的相应组件一起组成 `ControlNetXSCrossAttnUpBlock2D`"""

    # 初始化方法，接受一个控制到基础的模块列表
    def __init__(self, ctrl_to_base: nn.ModuleList):
        super().__init__()  # 调用父类的初始化方法
        self.ctrl_to_base = ctrl_to_base  # 将传入的控制到基础模块列表保存为实例变量


# 定义一个函数，获取下行块适配器
def get_down_block_adapter(
    base_in_channels: int,  # 基础输入通道数
    base_out_channels: int,  # 基础输出通道数
    ctrl_in_channels: int,  # 控制输入通道数
    ctrl_out_channels: int,  # 控制输出通道数
    temb_channels: int,  # 时间嵌入通道数
    max_norm_num_groups: Optional[int] = 32,  # 最大归一化组数
    has_crossattn=True,  # 是否使用交叉注意力
    transformer_layers_per_block: Optional[Union[int, Tuple[int]]] = 1,  # 每个块的变换器层数
    num_attention_heads: Optional[int] = 1,  # 注意力头数量
    cross_attention_dim: Optional[int] = 1024,  # 交叉注意力维度
    add_downsample: bool = True,  # 是否添加下采样
    upcast_attention: Optional[bool] = False,  # 是否上调注意力
    use_linear_projection: Optional[bool] = True,  # 是否使用线性投影
):
    num_layers = 2  # 仅支持 sd + sdxl

    resnets = []  # 存储 ResNet 块的列表
    attentions = []  # 存储注意力模型的列表
    ctrl_to_base = []  # 存储控制到基础的卷积层列表
    base_to_ctrl = []  # 存储基础到控制的卷积层列表

    # 如果传入的是整数，则将其转换为与层数相同的列表
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * num_layers

    # 遍历每层以构建网络结构
    for i in range(num_layers):
        # 第一层使用基础输入通道数，后续层使用基础输出通道数
        base_in_channels = base_in_channels if i == 0 else base_out_channels
        # 第一层使用控制输入通道数，后续层使用控制输出通道数
        ctrl_in_channels = ctrl_in_channels if i == 0 else ctrl_out_channels

        # 在应用 ResNet/注意力之前，从基础到控制的通道信息进行连接
        # 连接不需要更改通道数量
        base_to_ctrl.append(make_zero_conv(base_in_channels, base_in_channels))

        resnets.append(
            ResnetBlock2D(
                in_channels=ctrl_in_channels + base_in_channels,  # 从基础连接到控制的信息
                out_channels=ctrl_out_channels,  # 控制输出通道数
                temb_channels=temb_channels,  # 时间嵌入通道数
                groups=find_largest_factor(ctrl_in_channels + base_in_channels, max_factor=max_norm_num_groups),  # 计算组数
                groups_out=find_largest_factor(ctrl_out_channels, max_factor=max_norm_num_groups),  # 计算输出组数
                eps=1e-5,  # 小常数以避免除零
            )
        )

        # 如果需要交叉注意力，则添加对应的模型
        if has_crossattn:
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,  # 注意力头数量
                    ctrl_out_channels // num_attention_heads,  # 每个头的通道数
                    in_channels=ctrl_out_channels,  # 输入通道数
                    num_layers=transformer_layers_per_block[i],  # 当前块的变换器层数
                    cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                    use_linear_projection=use_linear_projection,  # 是否使用线性投影
                    upcast_attention=upcast_attention,  # 是否上调注意力
                    norm_num_groups=find_largest_factor(ctrl_out_channels, max_factor=max_norm_num_groups),  # 计算归一化组数
                )
            )

        # 在应用 ResNet/注意力之后，从控制到基础的通道信息进行相加
        # 相加需要更改通道数量
        ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))  # 添加控制到基础的卷积层
    # 判断是否需要进行下采样
    if add_downsample:
        # 在应用下采样器之前，将 base 的信息与 control 的信息连接
        # 连接操作不需要改变通道数量
        base_to_ctrl.append(make_zero_conv(base_out_channels, base_out_channels))

        # 创建下采样器对象，输入通道为控制通道和基础通道之和，使用卷积，输出通道为控制通道数量，命名为 "op"
        downsamplers = Downsample2D(
            ctrl_out_channels + base_out_channels, use_conv=True, out_channels=ctrl_out_channels, name="op"
        )

        # 在应用下采样器之后，将控制的数据信息添加到基础数据中
        # 添加操作需要改变通道数量
        ctrl_to_base.append(make_zero_conv(ctrl_out_channels, base_out_channels))
    else:
        # 如果不需要下采样，则将 downsamplers 设置为 None
        downsamplers = None

    # 创建下块控制网络适配器，传入残差网络和连接的控制基础模块
    down_block_components = DownBlockControlNetXSAdapter(
        resnets=nn.ModuleList(resnets),
        base_to_ctrl=nn.ModuleList(base_to_ctrl),
        ctrl_to_base=nn.ModuleList(ctrl_to_base),
    )

    # 如果存在交叉注意力，则将注意力模块添加到下块组件中
    if has_crossattn:
        down_block_components.attentions = nn.ModuleList(attentions)
    # 如果下采样器不为 None，则将下采样器添加到下块组件中
    if downsamplers is not None:
        down_block_components.downsamplers = downsamplers

    # 返回下块组件
    return down_block_components
# 定义一个函数，用于获取中间块适配器，接受多个参数以配置其行为
def get_mid_block_adapter(
    # 基础通道数
    base_channels: int,
    # 控制通道数
    ctrl_channels: int,
    # 可选的时间嵌入通道数
    temb_channels: Optional[int] = None,
    # 最大归一化组数量，默认为32
    max_norm_num_groups: Optional[int] = 32,
    # 每个块的变换层数，默认为1
    transformer_layers_per_block: int = 1,
    # 可选的注意力头数量，默认为1
    num_attention_heads: Optional[int] = 1,
    # 可选的交叉注意力维度，默认为1024
    cross_attention_dim: Optional[int] = 1024,
    # 是否提升注意力精度，默认为False
    upcast_attention: bool = False,
    # 是否使用线性投影，默认为True
    use_linear_projection: bool = True,
):
    # 在中间块应用之前，从基础通道到控制通道的信息进行拼接
    # 拼接不需要改变通道数
    base_to_ctrl = make_zero_conv(base_channels, base_channels)

    # 创建一个中间块对象，使用交叉注意力
    midblock = UNetMidBlock2DCrossAttn(
        # 设置每个块的变换层数
        transformer_layers_per_block=transformer_layers_per_block,
        # 输入通道为控制通道和基础通道的和
        in_channels=ctrl_channels + base_channels,
        # 输出通道为控制通道数
        out_channels=ctrl_channels,
        # 时间嵌入通道数
        temb_channels=temb_channels,
        # 归一化组数量必须能够同时整除输入和输出通道数
        resnet_groups=find_largest_factor(gcd(ctrl_channels, ctrl_channels + base_channels), max_norm_num_groups),
        # 交叉注意力的维度
        cross_attention_dim=cross_attention_dim,
        # 注意力头的数量
        num_attention_heads=num_attention_heads,
        # 是否使用线性投影
        use_linear_projection=use_linear_projection,
        # 是否提升注意力精度
        upcast_attention=upcast_attention,
    )

    # 在中间块应用之后，从控制通道到基础通道的信息进行相加
    # 相加需要改变通道数
    ctrl_to_base = make_zero_conv(ctrl_channels, base_channels)

    # 返回一个中间块控制适配器的实例，包含拼接层、中间块和相加层
    return MidBlockControlNetXSAdapter(base_to_ctrl=base_to_ctrl, midblock=midblock, ctrl_to_base=ctrl_to_base)


# 定义一个函数，用于获取上块适配器，接受输出通道数、前一层输出通道数和控制跳跃通道
def get_up_block_adapter(
    # 输出通道数
    out_channels: int,
    # 前一层的输出通道数
    prev_output_channel: int,
    # 控制跳跃通道列表
    ctrl_skip_channels: List[int],
):
    # 初始化控制到基础的卷积层列表
    ctrl_to_base = []
    # 设置层数为3，仅支持 sd 和 sdxl
    num_layers = 3  
    # 循环构建每一层的控制到基础卷积层
    for i in range(num_layers):
        # 第一层使用前一层输出通道，其他层使用输出通道
        resnet_in_channels = prev_output_channel if i == 0 else out_channels
        # 将控制跳跃通道与当前输入通道连接
        ctrl_to_base.append(make_zero_conv(ctrl_skip_channels[i], resnet_in_channels))

    # 返回一个上块控制适配器的实例，使用nn.ModuleList管理控制到基础卷积层
    return UpBlockControlNetXSAdapter(ctrl_to_base=nn.ModuleList(ctrl_to_base))


# 定义一个控制网络适配器类，继承自ModelMixin和ConfigMixin
class ControlNetXSAdapter(ModelMixin, ConfigMixin):
    r"""
    控制网络适配器模型。使用时，将其传递给 `UNetControlNetXSModel`（以及一个
    `UNet2DConditionModel` 基础模型）。

    该模型继承自[`ModelMixin`]和[`ConfigMixin`]。请查看超类文档，了解其通用
    方法（例如下载或保存）。

    与`UNetControlNetXSModel`一样，`ControlNetXSAdapter`与StableDiffusion和StableDiffusion-XL兼容。其
    默认参数与StableDiffusion兼容。
    # 参数部分说明
    Parameters:
        # conditioning_channels: 条件输入的通道数（例如：一张图像），默认值为3
        conditioning_channels (`int`, defaults to 3):
            # 条件图像的通道顺序。若为 `bgr`，则转换为 `rgb`
            conditioning_channel_order (`str`, defaults to `"rgb"`):
            # `controlnet_cond_embedding` 层中每个块的输出通道的元组，默认值为 (16, 32, 96, 256)
            conditioning_embedding_out_channels (`tuple[int]`, defaults to `(16, 32, 96, 256)`):
            # time_embedding_mix: 如果为0，则仅使用控制适配器的时间嵌入；如果为1，则仅使用基础 UNet 的时间嵌入；否则，两者结合
            time_embedding_mix (`float`, defaults to 1.0):
            # learn_time_embedding: 是否应学习时间嵌入，若是则 `UNetControlNetXSModel` 会结合基础模型和控制适配器的时间嵌入，若否则只使用基础模型的时间嵌入
            learn_time_embedding (`bool`, defaults to `False`):
            # num_attention_heads: 注意力头的数量，默认值为 [4]
            num_attention_heads (`list[int]`, defaults to `[4]`):
            # block_out_channels: 每个块的输出通道的元组，默认值为 [4, 8, 16, 16]
            block_out_channels (`list[int]`, defaults to `[4, 8, 16, 16]`):
            # base_block_out_channels: 基础 UNet 中每个块的输出通道的元组，默认值为 [320, 640, 1280, 1280]
            base_block_out_channels (`list[int]`, defaults to `[320, 640, 1280, 1280]`):
            # cross_attention_dim: 跨注意力特征的维度，默认值为 1024
            cross_attention_dim (`int`, defaults to 1024):
            # down_block_types: 要使用的下采样块的元组，默认值为 ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]
            down_block_types (`list[str]`, defaults to `["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]`):
            # sample_size: 输入/输出样本的高度和宽度，默认值为 96
            sample_size (`int`, defaults to 96):
            # transformer_layers_per_block: 每个块的变换器块数量，默认值为 1，仅与某些块相关
            transformer_layers_per_block (`Union[int, Tuple[int]]`, defaults to 1):
            # upcast_attention: 是否应始终提升注意力计算的精度，默认值为 True
            upcast_attention (`bool`, defaults to `True`):
            # max_norm_num_groups: 分组归一化中的最大组数，默认值为 32，实际数量为不大于 max_norm_num_groups 的相应通道的最大除数
            max_norm_num_groups (`int`, defaults to 32):
    # 注释部分结束
    """
    
    # 注册到配置中
    @register_to_config
    # 初始化方法，设置 ControlNetXSAdapter 的基本参数
        def __init__(
            # 条件通道数，默认为 3
            self,
            conditioning_channels: int = 3,
            # 条件通道的颜色顺序，默认为 RGB
            conditioning_channel_order: str = "rgb",
            # 输出通道数的元组，定义各层的输出通道
            conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
            # 时间嵌入混合因子，默认为 1.0
            time_embedding_mix: float = 1.0,
            # 是否学习时间嵌入，默认为 False
            learn_time_embedding: bool = False,
            # 注意力头数，默认为 4，可以是整数或整数元组
            num_attention_heads: Union[int, Tuple[int]] = 4,
            # 块输出通道的元组，定义每个块的输出通道
            block_out_channels: Tuple[int] = (4, 8, 16, 16),
            # 基础块输出通道的元组
            base_block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            # 交叉注意力维度，默认为 1024
            cross_attention_dim: int = 1024,
            # 各层的块类型元组
            down_block_types: Tuple[str] = (
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            # 采样大小，默认为 96
            sample_size: Optional[int] = 96,
            # 每个块的变换器层数，可以是整数或整数元组
            transformer_layers_per_block: Union[int, Tuple[int]] = 1,
            # 是否上溢注意力，默认为 True
            upcast_attention: bool = True,
            # 最大归一化组数，默认为 32
            max_norm_num_groups: int = 32,
            # 是否使用线性投影，默认为 True
            use_linear_projection: bool = True,
        # 类方法，从 UNet 创建 ControlNetXSAdapter
        @classmethod
        def from_unet(
            cls,
            # 传入的 UNet2DConditionModel 对象
            unet: UNet2DConditionModel,
            # 尺寸比例，默认为 None
            size_ratio: Optional[float] = None,
            # 可选的块输出通道列表
            block_out_channels: Optional[List[int]] = None,
            # 可选的注意力头数列表
            num_attention_heads: Optional[List[int]] = None,
            # 是否学习时间嵌入，默认为 False
            learn_time_embedding: bool = False,
            # 时间嵌入混合因子，默认为 1.0
            time_embedding_mix: int = 1.0,
            # 条件通道数，默认为 3
            conditioning_channels: int = 3,
            # 条件通道的颜色顺序，默认为 RGB
            conditioning_channel_order: str = "rgb",
            # 输出通道数的元组，默认为 (16, 32, 96, 256)
            conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        # 前向传播方法，处理输入参数
        def forward(self, *args, **kwargs):
            # 抛出错误，指示不能单独运行 ControlNetXSAdapter
            raise ValueError(
                "A ControlNetXSAdapter cannot be run by itself. Use it together with a UNet2DConditionModel to instantiate a UNetControlNetXSModel."
            )
# 定义一个 UNet 融合 ControlNet-XS 适配器的模型类
class UNetControlNetXSModel(ModelMixin, ConfigMixin):
    r"""
    A UNet fused with a ControlNet-XS adapter model

    此模型继承自 [`ModelMixin`] 和 [`ConfigMixin`]。有关所有模型实现的通用方法（如下载或保存），请检查超类文档。

    `UNetControlNetXSModel` 与 StableDiffusion 和 StableDiffusion-XL 兼容。其默认参数与 StableDiffusion 兼容。

    它的参数要么传递给底层的 `UNet2DConditionModel`，要么与 `ControlNetXSAdapter` 完全相同。有关详细信息，请参阅它们的文档。
    """

    # 启用梯度检查点支持
    _supports_gradient_checkpointing = True

    # 注册到配置的方法
    @register_to_config
    def __init__(
        self,
        # unet 配置
        # 样本尺寸，默认值为 96
        sample_size: Optional[int] = 96,
        # 下采样块类型的元组
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        # 上采样块类型的元组
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        # 每个块的输出通道数
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        # 归一化的组数，默认为 32
        norm_num_groups: Optional[int] = 32,
        # 交叉注意力维度，默认为 1024
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        # 每个块的变换器层数，默认为 1
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        # 注意力头的数量，默认为 8
        num_attention_heads: Union[int, Tuple[int]] = 8,
        # 附加嵌入类型，默认为 None
        addition_embed_type: Optional[str] = None,
        # 附加时间嵌入维度，默认为 None
        addition_time_embed_dim: Optional[int] = None,
        # 是否上溯注意力，默认为 True
        upcast_attention: bool = True,
        # 是否使用线性投影，默认为 True
        use_linear_projection: bool = True,
        # 时间条件投影维度，默认为 None
        time_cond_proj_dim: Optional[int] = None,
        # 类别嵌入输入维度，默认为 None
        projection_class_embeddings_input_dim: Optional[int] = None,
        # 附加控制网配置
        # 时间嵌入混合系数，默认为 1.0
        time_embedding_mix: float = 1.0,
        # 控制条件通道数，默认为 3
        ctrl_conditioning_channels: int = 3,
        # 控制条件嵌入输出通道的元组
        ctrl_conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
        # 控制条件通道顺序，默认为 "rgb"
        ctrl_conditioning_channel_order: str = "rgb",
        # 是否学习时间嵌入，默认为 False
        ctrl_learn_time_embedding: bool = False,
        # 控制块输出通道的元组
        ctrl_block_out_channels: Tuple[int] = (4, 8, 16, 16),
        # 控制注意力头的数量，默认为 4
        ctrl_num_attention_heads: Union[int, Tuple[int]] = 4,
        # 控制最大归一化组数，默认为 32
        ctrl_max_norm_num_groups: int = 32,
    # 定义类方法，从 UNet 创建模型
    @classmethod
    def from_unet(
        cls,
        # UNet2DConditionModel 实例
        unet: UNet2DConditionModel,
        # 可选的 ControlNetXSAdapter 实例
        controlnet: Optional[ControlNetXSAdapter] = None,
        # 可选的大小比例
        size_ratio: Optional[float] = None,
        # 可选的控制块输出通道列表
        ctrl_block_out_channels: Optional[List[float]] = None,
        # 可选的时间嵌入混合系数
        time_embedding_mix: Optional[float] = None,
        # 可选的控制额外参数字典
        ctrl_optional_kwargs: Optional[Dict] = None,
    # 冻结 UNet2DConditionModel 基本部分的权重，其他部分可用于微调
    def freeze_unet_params(self) -> None:
        """Freeze the weights of the parts belonging to the base UNet2DConditionModel, and leave everything else unfrozen for fine
        tuning."""
        # 将所有参数的梯度计算设置为可用
        for param in self.parameters():
            param.requires_grad = True
    
        # 解冻 ControlNetXSAdapter 相关部分
        base_parts = [
            "base_time_proj",
            "base_time_embedding",
            "base_add_time_proj",
            "base_add_embedding",
            "base_conv_in",
            "base_conv_norm_out",
            "base_conv_act",
            "base_conv_out",
        ]
        # 获取存在的基本部分的属性，过滤掉 None
        base_parts = [getattr(self, part) for part in base_parts if getattr(self, part) is not None]
        # 冻结基本部分的所有参数
        for part in base_parts:
            for param in part.parameters():
                param.requires_grad = False
    
        # 冻结每个下采样块的基本参数
        for d in self.down_blocks:
            d.freeze_base_params()
        # 冻结中间块的基本参数
        self.mid_block.freeze_base_params()
        # 冻结每个上采样块的基本参数
        for u in self.up_blocks:
            u.freeze_base_params()
    
    # 设置模块的梯度检查点功能
    def _set_gradient_checkpointing(self, module, value=False):
        # 如果模块具有梯度检查点属性，则设置其值
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
    
    @property
    # 从 UNet2DConditionModel 中复制的属性，用于获取注意力处理器
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # 用于递归设置处理器的字典
        processors = {}
    
        # 递归添加处理器的辅助函数
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 如果模块具有获取处理器的方法，则将其添加到字典中
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
    
            # 遍历模块的所有子模块，递归调用处理器添加函数
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
            return processors
    
        # 遍历当前对象的所有子模块，并调用处理器添加函数
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
    
        # 返回所有处理器的字典
        return processors
    
    # 从 UNet2DConditionModel 中复制的设置注意力处理器的方法
    # 定义设置注意力处理器的方法，参数为单个处理器或处理器字典
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。
    
        参数：
            processor（`dict` 或 `AttentionProcessor`）： 
                实例化的处理器类或将作为处理器设置的处理器类字典
                对于**所有** `Attention` 层。
    
                如果 `processor` 是字典，键需要定义对应交叉注意力处理器的路径。
                当设置可训练的注意力处理器时，强烈推荐这样做。
    
        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())
    
        # 如果传入的处理器为字典且数量不匹配，则抛出异常
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器字典，但处理器数量 {len(processor)} 与"
                f" 注意力层数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
            )
    
        # 定义递归设置注意力处理器的函数
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块具有 set_processor 方法，则设置处理器
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
    
            # 遍历模块的子模块，递归调用处理器设置函数
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
        # 遍历当前对象的子模块，调用递归设置函数
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)
    
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor 复制的
    def set_default_attn_processor(self):
        """
        禁用自定义注意力处理器并设置默认的注意力实现。
        """
        # 如果所有处理器都是添加的 KV 注意力处理器，则设置处理器为 AttnAddedKVProcessor
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        # 如果所有处理器都是交叉注意力处理器，则设置处理器为 AttnProcessor
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            # 否则抛出异常，提示无法设置默认处理器
            raise ValueError(
                f"当注意力处理器类型为 {next(iter(self.attn_processors.values()))} 时，无法调用 `set_default_attn_processor`"
            )
    
        # 调用设置注意力处理器的方法
        self.set_attn_processor(processor)
    
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.enable_freeu 复制的
    # 定义启用 FreeU 机制的方法，接收四个浮点数参数
        def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
            # 文档字符串，描述该方法的用途和参数含义
            r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.
    
            The suffixes after the scaling factors represent the stage blocks where they are being applied.
    
            Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
            are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.
    
            Args:
                s1 (`float`):
                    Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                    mitigate the "oversmoothing effect" in the enhanced denoising process.
                s2 (`float`):
                    Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                    mitigate the "oversmoothing effect" in the enhanced denoising process.
                b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
                b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
            """
            # 遍历上采样模块并为每个模块设置对应的 scaling 因子
            for i, upsample_block in enumerate(self.up_blocks):
                # 设置上采样块的 s1 属性为传入的 s1 值
                setattr(upsample_block, "s1", s1)
                # 设置上采样块的 s2 属性为传入的 s2 值
                setattr(upsample_block, "s2", s2)
                # 设置上采样块的 b1 属性为传入的 b1 值
                setattr(upsample_block, "b1", b1)
                # 设置上采样块的 b2 属性为传入的 b2 值
                setattr(upsample_block, "b2", b2)
    
        # 定义禁用 FreeU 机制的方法
        def disable_freeu(self):
            """Disables the FreeU mechanism."""
            # 定义 FreeU 机制中需要清除的键集合
            freeu_keys = {"s1", "s2", "b1", "b2"}
            # 遍历上采样模块
            for i, upsample_block in enumerate(self.up_blocks):
                # 遍历需要清除的键
                for k in freeu_keys:
                    # 检查模块是否具有该属性或属性值非 None
                    if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                        # 将属性值设置为 None，禁用 FreeU
                        setattr(upsample_block, k, None)
    
        # 定义融合 QKV 投影的方法
        def fuse_qkv_projections(self):
            """
            Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
            are fused. For cross-attention modules, key and value projection matrices are fused.
    
            <Tip warning={true}>
    
            This API is 🧪 experimental.
    
            </Tip>
            """
            # 初始化原始注意力处理器为 None
            self.original_attn_processors = None
    
            # 遍历注意力处理器
            for _, attn_processor in self.attn_processors.items():
                # 检查是否有添加的 KV 投影，不支持融合
                if "Added" in str(attn_processor.__class__.__name__):
                    raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")
    
            # 记录原始注意力处理器
            self.original_attn_processors = self.attn_processors
    
            # 遍历所有模块
            for module in self.modules():
                # 检查模块是否为注意力模块
                if isinstance(module, Attention):
                    # 执行投影融合
                    module.fuse_projections(fuse=True)
    
            # 设置注意力处理器为融合后的处理器
            self.set_attn_processor(FusedAttnProcessor2_0())
    
        # 此部分代码未提供，可能是禁用 QKV 投影的方法
    # 定义一个方法，用于禁用已启用的融合 QKV 投影
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        # 如果原始的注意力处理器不为 None，则设置为原始处理器
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    # 定义前向传播方法，接收多个输入参数
    def forward(
        self,
        sample: Tensor,  # 输入样本，类型为 Tensor
        timestep: Union[torch.Tensor, float, int],  # 时间步，支持多种类型
        encoder_hidden_states: torch.Tensor,  # 编码器的隐藏状态，类型为 Tensor
        controlnet_cond: Optional[torch.Tensor] = None,  # 可选的控制网络条件，类型为 Tensor
        conditioning_scale: Optional[float] = 1.0,  # 条件缩放因子，默认为 1.0
        class_labels: Optional[torch.Tensor] = None,  # 可选的类标签，类型为 Tensor
        timestep_cond: Optional[torch.Tensor] = None,  # 可选的时间步条件，类型为 Tensor
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码，类型为 Tensor
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的交叉注意力参数，类型为字典
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,  # 可选的附加条件参数，类型为字典
        return_dict: bool = True,  # 是否返回字典格式的结果，默认为 True
        apply_control: bool = True,  # 是否应用控制逻辑，默认为 True
# 定义一个名为 ControlNetXSCrossAttnDownBlock2D 的类，继承自 nn.Module
class ControlNetXSCrossAttnDownBlock2D(nn.Module):
    # 初始化方法，定义类的属性和参数
    def __init__(
        self,
        base_in_channels: int,  # 基础输入通道数
        base_out_channels: int,  # 基础输出通道数
        ctrl_in_channels: int,  # 控制输入通道数
        ctrl_out_channels: int,  # 控制输出通道数
        temb_channels: int,  # 时间嵌入通道数
        norm_num_groups: int = 32,  # 规范化组数
        ctrl_max_norm_num_groups: int = 32,  # 控制最大规范化组数
        has_crossattn=True,  # 是否包含交叉注意力机制
        transformer_layers_per_block: Optional[Union[int, Tuple[int]]] = 1,  # 每个块的变换器层数
        base_num_attention_heads: Optional[int] = 1,  # 基础注意力头数
        ctrl_num_attention_heads: Optional[int] = 1,  # 控制注意力头数
        cross_attention_dim: Optional[int] = 1024,  # 交叉注意力维度
        add_downsample: bool = True,  # 是否添加下采样
        upcast_attention: Optional[bool] = False,  # 是否上升注意力
        use_linear_projection: Optional[bool] = True,  # 是否使用线性投影
    @classmethod
    # 定义一个类方法，用于冻结基础模型的参数
    def freeze_base_params(self) -> None:
        """冻结基础 UNet2DConditionModel 的权重，保持其他部分可调，以便微调。"""
        # 解冻所有参数
        for param in self.parameters():
            param.requires_grad = True

        # 冻结基础部分的参数
        base_parts = [self.base_resnets]  # 包含基础残差网络部分
        if isinstance(self.base_attentions, nn.ModuleList):  # 如果注意力部分是一个模块列表
            base_parts.append(self.base_attentions)  # 添加基础注意力部分
        if self.base_downsamplers is not None:  # 如果存在基础下采样部分
            base_parts.append(self.base_downsamplers)  # 添加基础下采样部分
        for part in base_parts:  # 遍历基础部分
            for param in part.parameters():  # 遍历参数
                param.requires_grad = False  # 冻结参数以防止更新

    # 定义前向传播方法
    def forward(
        self,
        hidden_states_base: Tensor,  # 基础隐藏状态
        temb: Tensor,  # 时间嵌入
        encoder_hidden_states: Optional[Tensor] = None,  # 编码器隐藏状态
        hidden_states_ctrl: Optional[Tensor] = None,  # 控制隐藏状态
        conditioning_scale: Optional[float] = 1.0,  # 条件缩放因子
        attention_mask: Optional[Tensor] = None,  # 注意力掩码
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 交叉注意力的关键字参数
        encoder_attention_mask: Optional[Tensor] = None,  # 编码器注意力掩码
        apply_control: bool = True,  # 是否应用控制
class ControlNetXSCrossAttnMidBlock2D(nn.Module):
    # 定义一个名为 ControlNetXSCrossAttnMidBlock2D 的类，继承自 nn.Module
    def __init__(
        self,
        base_channels: int,  # 基础通道数
        ctrl_channels: int,  # 控制通道数
        temb_channels: Optional[int] = None,  # 时间嵌入通道数（可选）
        norm_num_groups: int = 32,  # 规范化组数
        ctrl_max_norm_num_groups: int = 32,  # 控制最大规范化组数
        transformer_layers_per_block: int = 1,  # 每个块的变换器层数
        base_num_attention_heads: Optional[int] = 1,  # 基础注意力头数
        ctrl_num_attention_heads: Optional[int] = 1,  # 控制注意力头数
        cross_attention_dim: Optional[int] = 1024,  # 交叉注意力维度
        upcast_attention: bool = False,  # 是否上升注意力
        use_linear_projection: Optional[bool] = True,  # 是否使用线性投影
    ):
        # 调用父类的构造函数以初始化继承的属性和方法
        super().__init__()

        # 在中间块应用之前，从基础信息到控制信息的连接。
        # 连接不需要改变通道数量
        self.base_to_ctrl = make_zero_conv(base_channels, base_channels)

        # 创建基础中间块，使用交叉注意力机制
        self.base_midblock = UNetMidBlock2DCrossAttn(
            # 每个块中的变换器层数量
            transformer_layers_per_block=transformer_layers_per_block,
            # 输入通道数为基础通道数
            in_channels=base_channels,
            # 嵌入通道数
            temb_channels=temb_channels,
            # ResNet 组的数量
            resnet_groups=norm_num_groups,
            # 交叉注意力维度
            cross_attention_dim=cross_attention_dim,
            # 注意力头的数量
            num_attention_heads=base_num_attention_heads,
            # 是否使用线性投影
            use_linear_projection=use_linear_projection,
            # 是否上溯注意力
            upcast_attention=upcast_attention,
        )

        # 创建控制中间块，使用交叉注意力机制
        self.ctrl_midblock = UNetMidBlock2DCrossAttn(
            # 每个块中的变换器层数量
            transformer_layers_per_block=transformer_layers_per_block,
            # 输入通道数为控制通道数加基础通道数
            in_channels=ctrl_channels + base_channels,
            # 输出通道数为控制通道数
            out_channels=ctrl_channels,
            # 嵌入通道数
            temb_channels=temb_channels,
            # norm 组数量必须同时能被输入和输出通道数整除
            resnet_groups=find_largest_factor(
                # 计算控制通道与控制通道加基础通道的最大公约数
                gcd(ctrl_channels, ctrl_channels + base_channels), ctrl_max_norm_num_groups
            ),
            # 交叉注意力维度
            cross_attention_dim=cross_attention_dim,
            # 注意力头的数量
            num_attention_heads=ctrl_num_attention_heads,
            # 是否使用线性投影
            use_linear_projection=use_linear_projection,
            # 是否上溯注意力
            upcast_attention=upcast_attention,
        )

        # 在中间块应用之后，从控制信息到基础信息的相加
        # 相加需要改变通道数量
        self.ctrl_to_base = make_zero_conv(ctrl_channels, base_channels)

        # 初始化梯度检查点标志为假
        self.gradient_checkpointing = False

    @classmethod
    def from_modules(
        # 类方法，接受基础中间块和控制中间块作为参数
        cls,
        base_midblock: UNetMidBlock2DCrossAttn,
        ctrl_midblock: MidBlockControlNetXSAdapter,
    ):
        # 获取中间块的基准到控制的映射
        base_to_ctrl = ctrl_midblock.base_to_ctrl
        # 获取中间块的控制到基准的映射
        ctrl_to_base = ctrl_midblock.ctrl_to_base
        # 获取中间块的实例
        ctrl_midblock = ctrl_midblock.midblock

        # 获取第一个交叉注意力模块
        def get_first_cross_attention(midblock):
            # 返回中间块的第一个注意力模块的交叉注意力层
            return midblock.attentions[0].transformer_blocks[0].attn2

        # 获取控制到基准的输出通道数
        base_channels = ctrl_to_base.out_channels
        # 获取控制到基准的输入通道数
        ctrl_channels = ctrl_to_base.in_channels
        # 获取基准中间块的每个块的转换层数
        transformer_layers_per_block = len(base_midblock.attentions[0].transformer_blocks)
        # 获取基准中间块时间嵌入的输入特征数
        temb_channels = base_midblock.resnets[0].time_emb_proj.in_features
        # 获取基准中间块的归一化组数
        num_groups = base_midblock.resnets[0].norm1.num_groups
        # 获取控制中间块的归一化组数
        ctrl_num_groups = ctrl_midblock.resnets[0].norm1.num_groups
        # 获取基准中间块第一个交叉注意力模块的注意力头数
        base_num_attention_heads = get_first_cross_attention(base_midblock).heads
        # 获取控制中间块第一个交叉注意力模块的注意力头数
        ctrl_num_attention_heads = get_first_cross_attention(ctrl_midblock).heads
        # 获取基准中间块第一个交叉注意力模块的交叉注意力维度
        cross_attention_dim = get_first_cross_attention(base_midblock).cross_attention_dim
        # 获取基准中间块第一个交叉注意力模块的上采样注意力设置
        upcast_attention = get_first_cross_attention(base_midblock).upcast_attention
        # 获取基准中间块第一个注意力模块的线性投影使用情况
        use_linear_projection = base_midblock.attentions[0].use_linear_projection

        # 创建模型实例
        model = cls(
            # 传入基准通道数
            base_channels=base_channels,
            # 传入控制通道数
            ctrl_channels=ctrl_channels,
            # 传入时间嵌入通道数
            temb_channels=temb_channels,
            # 传入归一化组数
            norm_num_groups=num_groups,
            # 传入控制最大归一化组数
            ctrl_max_norm_num_groups=ctrl_num_groups,
            # 传入每块的转换层数
            transformer_layers_per_block=transformer_layers_per_block,
            # 传入基准注意力头数
            base_num_attention_heads=base_num_attention_heads,
            # 传入控制注意力头数
            ctrl_num_attention_heads=ctrl_num_attention_heads,
            # 传入交叉注意力维度
            cross_attention_dim=cross_attention_dim,
            # 传入上采样注意力设置
            upcast_attention=upcast_attention,
            # 传入线性投影使用情况
            use_linear_projection=use_linear_projection,
        )

        # 加载模型权重
        model.base_to_ctrl.load_state_dict(base_to_ctrl.state_dict())
        # 加载基准中间块的权重
        model.base_midblock.load_state_dict(base_midblock.state_dict())
        # 加载控制中间块的权重
        model.ctrl_midblock.load_state_dict(ctrl_midblock.state_dict())
        # 加载控制到基准的权重
        model.ctrl_to_base.load_state_dict(ctrl_to_base.state_dict())

        # 返回构建好的模型
        return model

    def freeze_base_params(self) -> None:
        """冻结属于基准 UNet2DConditionModel 的权重，保留其他部分以便进行微调。"""
        # 解冻所有参数
        for param in self.parameters():
            param.requires_grad = True

        # 冻结基准部分的参数
        for param in self.base_midblock.parameters():
            param.requires_grad = False

    def forward(
        self,
        # 基准的隐藏状态
        hidden_states_base: Tensor,
        # 时间嵌入
        temb: Tensor,
        # 编码器的隐藏状态
        encoder_hidden_states: Tensor,
        # 控制的隐藏状态（可选）
        hidden_states_ctrl: Optional[Tensor] = None,
        # 条件缩放因子（可选），默认为1.0
        conditioning_scale: Optional[float] = 1.0,
        # 交叉注意力的额外参数（可选）
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 注意力掩码（可选）
        attention_mask: Optional[Tensor] = None,
        # 编码器的注意力掩码（可选）
        encoder_attention_mask: Optional[Tensor] = None,
        # 是否应用控制（默认为True）
        apply_control: bool = True,
    # 返回一个包含两个张量的元组
    ) -> Tuple[Tensor, Tensor]:
        # 如果提供了交叉注意力的参数
        if cross_attention_kwargs is not None:
            # 检查是否有 scale 参数，并发出警告
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    
        # 设置基础隐藏状态
        h_base = hidden_states_base
        # 设置控制隐藏状态
        h_ctrl = hidden_states_ctrl
    
        # 创建一个包含多个参数的字典
        joint_args = {
            "temb": temb,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
            "cross_attention_kwargs": cross_attention_kwargs,
            "encoder_attention_mask": encoder_attention_mask,
        }
    
        # 如果应用控制，则连接基础和控制隐藏状态
        if apply_control:
            h_ctrl = torch.cat([h_ctrl, self.base_to_ctrl(h_base)], dim=1)  # concat base -> ctrl
        # 应用基础中间块到基础隐藏状态
        h_base = self.base_midblock(h_base, **joint_args)  # apply base mid block
        # 如果应用控制，则应用控制中间块
        if apply_control:
            h_ctrl = self.ctrl_midblock(h_ctrl, **joint_args)  # apply ctrl mid block
            # 将控制结果加到基础隐藏状态上，乘以条件缩放因子
            h_base = h_base + self.ctrl_to_base(h_ctrl) * conditioning_scale  # add ctrl -> base
    
        # 返回基础和控制的隐藏状态
        return h_base, h_ctrl
# 定义一个名为 ControlNetXSCrossAttnUpBlock2D 的神经网络模块，继承自 nn.Module
class ControlNetXSCrossAttnUpBlock2D(nn.Module):
    # 初始化方法，定义该模块的参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        prev_output_channel: int,  # 前一层的输出通道数
        ctrl_skip_channels: List[int],  # 控制跳跃连接的通道数列表
        temb_channels: int,  # 时间嵌入通道数
        norm_num_groups: int = 32,  # 归一化的组数，默认值为32
        resolution_idx: Optional[int] = None,  # 分辨率索引，可选
        has_crossattn=True,  # 是否包含交叉注意力机制，默认值为True
        transformer_layers_per_block: int = 1,  # 每个模块的变换器层数，默认值为1
        num_attention_heads: int = 1,  # 注意力头的数量，默认值为1
        cross_attention_dim: int = 1024,  # 交叉注意力的维度，默认值为1024
        add_upsample: bool = True,  # 是否添加上采样层，默认值为True
        upcast_attention: bool = False,  # 是否提升注意力计算精度，默认值为False
        use_linear_projection: Optional[bool] = True,  # 是否使用线性投影，默认值为True
    ):
        # 调用父类的初始化方法
        super().__init__()
        resnets = []  # 初始化一个空列表，用于存放 ResNet 模块
        attentions = []  # 初始化一个空列表，用于存放注意力模块
        ctrl_to_base = []  # 初始化一个空列表，用于存放控制到基础的卷积模块

        num_layers = 3  # 仅支持3层，适用于 sd 和 sdxl

        # 记录是否包含交叉注意力和注意力头的数量
        self.has_cross_attention = has_crossattn
        self.num_attention_heads = num_attention_heads

        # 如果 transformer_layers_per_block 是整数，则将其扩展为包含 num_layers 个相同值的列表
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # 遍历每一层
        for i in range(num_layers):
            # 确定当前层的跳跃连接通道数
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 确定当前层的输入通道数
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 创建从控制通道到基础通道的零卷积，并添加到列表中
            ctrl_to_base.append(make_zero_conv(ctrl_skip_channels[i], resnet_in_channels))

            # 添加 ResNet 模块到 resnets 列表
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    groups=norm_num_groups,  # 归一化组数
                )
            )

            # 如果包含交叉注意力，则添加 Transformer 模块到 attentions 列表
            if has_crossattn:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,  # 注意力头数量
                        out_channels // num_attention_heads,  # 每个头的输出通道数
                        in_channels=out_channels,  # 输入通道数
                        num_layers=transformer_layers_per_block[i],  # 当前层的变换器层数
                        cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                        use_linear_projection=use_linear_projection,  # 是否使用线性投影
                        upcast_attention=upcast_attention,  # 是否提升注意力计算精度
                        norm_num_groups=norm_num_groups,  # 归一化组数
                    )
                )

        # 将 ResNet 模块列表转换为 nn.ModuleList，以便在模型中管理
        self.resnets = nn.ModuleList(resnets)
        # 如果有交叉注意力，转换 attentions 列表为 nn.ModuleList，否则填充 None
        self.attentions = nn.ModuleList(attentions) if has_crossattn else [None] * num_layers
        # 将控制到基础的卷积模块列表转换为 nn.ModuleList
        self.ctrl_to_base = nn.ModuleList(ctrl_to_base)

        # 如果需要添加上采样层，初始化 Upsample2D 模块
        if add_upsample:
            self.upsamplers = Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.upsamplers = None  # 如果不需要上采样，则将其设置为 None

        self.gradient_checkpointing = False  # 初始化时禁用梯度检查点
        self.resolution_idx = resolution_idx  # 设置分辨率索引
    # 从模块创建模型的类方法
        def from_modules(cls, base_upblock: CrossAttnUpBlock2D, ctrl_upblock: UpBlockControlNetXSAdapter):
            # 获取控制到基础的跳跃连接
            ctrl_to_base_skip_connections = ctrl_upblock.ctrl_to_base
    
            # 获取参数
            # 获取第一个交叉注意力模块
            def get_first_cross_attention(block):
                return block.attentions[0].transformer_blocks[0].attn2
    
            # 获取基础上采样块的输出通道数
            out_channels = base_upblock.resnets[0].out_channels
            # 计算输入通道数
            in_channels = base_upblock.resnets[-1].in_channels - out_channels
            # 计算前一个输出通道数
            prev_output_channels = base_upblock.resnets[0].in_channels - out_channels
            # 获取控制跳跃连接的输入通道数
            ctrl_skip_channelss = [c.in_channels for c in ctrl_to_base_skip_connections]
            # 获取时间嵌入的输入特征数
            temb_channels = base_upblock.resnets[0].time_emb_proj.in_features
            # 获取归一化组数
            num_groups = base_upblock.resnets[0].norm1.num_groups
            # 获取分辨率索引
            resolution_idx = base_upblock.resolution_idx
            # 检查基础上采样块是否有注意力模块
            if hasattr(base_upblock, "attentions"):
                has_crossattn = True
                # 获取每个块的变换层数
                transformer_layers_per_block = len(base_upblock.attentions[0].transformer_blocks)
                # 获取注意力头数
                num_attention_heads = get_first_cross_attention(base_upblock).heads
                # 获取交叉注意力维度
                cross_attention_dim = get_first_cross_attention(base_upblock).cross_attention_dim
                # 获取上升注意力标志
                upcast_attention = get_first_cross_attention(base_upblock).upcast_attention
                # 获取是否使用线性投影
                use_linear_projection = base_upblock.attentions[0].use_linear_projection
            else:
                has_crossattn = False
                transformer_layers_per_block = None
                num_attention_heads = None
                cross_attention_dim = None
                upcast_attention = None
                use_linear_projection = None
            # 检查是否需要添加上采样
            add_upsample = base_upblock.upsamplers is not None
    
            # 创建模型
            model = cls(
                # 输入通道数
                in_channels=in_channels,
                # 输出通道数
                out_channels=out_channels,
                # 前一个输出通道
                prev_output_channel=prev_output_channels,
                # 控制跳跃连接的输入通道数
                ctrl_skip_channels=ctrl_skip_channelss,
                # 时间嵌入的通道数
                temb_channels=temb_channels,
                # 归一化的组数
                norm_num_groups=num_groups,
                # 分辨率索引
                resolution_idx=resolution_idx,
                # 是否有交叉注意力
                has_crossattn=has_crossattn,
                # 每个块的变换层数
                transformer_layers_per_block=transformer_layers_per_block,
                # 注意力头数
                num_attention_heads=num_attention_heads,
                # 交叉注意力维度
                cross_attention_dim=cross_attention_dim,
                # 是否添加上采样
                add_upsample=add_upsample,
                # 上升注意力标志
                upcast_attention=upcast_attention,
                # 是否使用线性投影
                use_linear_projection=use_linear_projection,
            )
    
            # 加载权重
            model.resnets.load_state_dict(base_upblock.resnets.state_dict())
            # 如果有交叉注意力，加载其权重
            if has_crossattn:
                model.attentions.load_state_dict(base_upblock.attentions.state_dict())
            # 如果需要添加上采样，加载其权重
            if add_upsample:
                model.upsamplers.load_state_dict(base_upblock.upsamplers[0].state_dict())
            # 加载控制到基础的跳跃连接权重
            model.ctrl_to_base.load_state_dict(ctrl_to_base_skip_connections.state_dict())
    
            # 返回创建的模型
            return model
    # 定义一个方法，用于冻结基础 UNet2DConditionModel 的参数
    def freeze_base_params(self) -> None:
        """冻结属于基础 UNet2DConditionModel 的权重，其他部分保持解冻以便微调。"""
        # 解冻所有参数，允许训练
        for param in self.parameters():
            param.requires_grad = True
    
        # 冻结基础部分的参数
        base_parts = [self.resnets]  # 将基础部分（resnets）添加到列表中
        # 检查 attentions 是否是 ModuleList 类型（可能包含 None）
        if isinstance(self.attentions, nn.ModuleList):
            base_parts.append(self.attentions)  # 如果是，则添加 attentions
        # 检查 upsamplers 是否不为 None
        if self.upsamplers is not None:
            base_parts.append(self.upsamplers)  # 如果存在，添加 upsamplers
        # 冻结基础部分的参数
        for part in base_parts:
            for param in part.parameters():
                param.requires_grad = False  # 设置参数为不可训练
    
    # 定义前向传播方法
    def forward(
        self,
        hidden_states: Tensor,  # 输入的隐藏状态
        res_hidden_states_tuple_base: Tuple[Tensor, ...],  # 基础残差隐藏状态元组
        res_hidden_states_tuple_ctrl: Tuple[Tensor, ...],  # 控制残差隐藏状态元组
        temb: Tensor,  # 时间嵌入
        encoder_hidden_states: Optional[Tensor] = None,  # 可选的编码器隐藏状态
        conditioning_scale: Optional[float] = 1.0,  # 可选的条件缩放因子，默认值为 1.0
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的交叉注意力参数
        attention_mask: Optional[Tensor] = None,  # 可选的注意力掩码
        upsample_size: Optional[int] = None,  # 可选的上采样大小
        encoder_attention_mask: Optional[Tensor] = None,  # 可选的编码器注意力掩码
        apply_control: bool = True,  # 是否应用控制，默认值为 True
    # 函数返回一个 Tensor 对象
    ) -> Tensor:
        # 检查交叉注意力参数是否存在
        if cross_attention_kwargs is not None:
            # 检查参数中是否包含 "scale"
            if cross_attention_kwargs.get("scale", None) is not None:
                # 记录警告信息，表示 "scale" 参数已弃用
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    
        # 判断 FreeU 是否启用，检查相关属性是否存在
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )
    
        # 定义创建自定义前向传播的方法
        def create_custom_forward(module, return_dict=None):
            # 定义自定义前向传播函数
            def custom_forward(*inputs):
                # 根据是否返回字典选择调用方式
                if return_dict is not None:
                    return module(*inputs, return_dict=return_dict)
                else:
                    return module(*inputs)
    
            return custom_forward
    
        # 定义条件应用 FreeU 的方法
        def maybe_apply_freeu_to_subblock(hidden_states, res_h_base):
            # FreeU: 仅在前两个阶段操作
            if is_freeu_enabled:
                # 应用 FreeU 操作
                return apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_h_base,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )
            else:
                # 如果未启用 FreeU，直接返回输入状态
                return hidden_states, res_h_base
    
        # 同时遍历多个列表
        for resnet, attn, c2b, res_h_base, res_h_ctrl in zip(
            self.resnets,
            self.attentions,
            self.ctrl_to_base,
            reversed(res_hidden_states_tuple_base),
            reversed(res_hidden_states_tuple_ctrl),
        ):
            # 如果应用控制，则调整隐藏状态
            if apply_control:
                hidden_states += c2b(res_h_ctrl) * conditioning_scale
    
            # 可能应用 FreeU 操作
            hidden_states, res_h_base = maybe_apply_freeu_to_subblock(hidden_states, res_h_base)
            # 将隐藏状态和基础状态沿维度 1 拼接
            hidden_states = torch.cat([hidden_states, res_h_base], dim=1)
    
            # 如果在训练并启用梯度检查点
            if self.training and self.gradient_checkpointing:
                # 根据 PyTorch 版本设置检查点参数
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                # 应用检查点以减少内存使用
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                # 直接使用残差网络处理隐藏状态
                hidden_states = resnet(hidden_states, temb)
    
            # 如果注意力模块不为 None，则进行注意力计算
            if attn is not None:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
    
        # 如果上采样器存在，应用上采样操作
        if self.upsamplers is not None:
            hidden_states = self.upsamplers(hidden_states, upsample_size)
    
        # 返回最终的隐藏状态
        return hidden_states
# 创建一个零卷积层的函数，接收输入和输出通道数
def make_zero_conv(in_channels, out_channels=None):
    # 使用 zero_module 函数初始化一个卷积层，并设置卷积核大小为1，填充为0
    return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))


# 初始化传入模块的参数为零的函数
def zero_module(module):
    # 遍历模块的所有参数
    for p in module.parameters():
        # 将每个参数初始化为零
        nn.init.zeros_(p)
    # 返回已初始化的模块
    return module


# 查找给定数字的最大因数的函数，最大因数不超过指定值
def find_largest_factor(number, max_factor):
    # 将最大因数设置为初始因数
    factor = max_factor
    # 如果最大因数大于或等于数字，直接返回数字
    if factor >= number:
        return number
    # 循环直到找到一个因数
    while factor != 0:
        # 计算数字与因数的余数
        residual = number % factor
        # 如果余数为零，则因数是有效的
        if residual == 0:
            return factor
        # 减小因数，继续查找
        factor -= 1
```