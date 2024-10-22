# `.\diffusers\models\adapter.py`

```py
# 版权所有 2022 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（"许可证"）进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，依据许可证分发的软件
# 是以“原样”基础提供的，不附带任何形式的担保或条件，
# 无论是明示或暗示的。
# 有关许可证下特定权限和限制的具体信息，请参见许可证。
import os  # 导入操作系统模块以处理文件和目录
from typing import Callable, List, Optional, Union  # 导入类型注解以增强代码可读性

import torch  # 导入 PyTorch 库以进行张量运算和深度学习
import torch.nn as nn  # 导入 PyTorch 的神经网络模块以构建模型

from ..configuration_utils import ConfigMixin, register_to_config  # 从父目录导入配置相关工具
from ..utils import logging  # 从父目录导入日志记录工具
from .modeling_utils import ModelMixin  # 从当前目录导入模型混合工具

logger = logging.get_logger(__name__)  # 创建一个记录器实例，用于日志记录

class MultiAdapter(ModelMixin):  # 定义 MultiAdapter 类，继承自 ModelMixin
    r"""  # 类文档字符串，描述类的功能和用途
    MultiAdapter 是一个包装模型，包含多个适配器模型，并根据
    用户分配的权重合并它们的输出。

    该模型继承自 [`ModelMixin`]。有关库实现的所有模型的通用方法的文档
    （例如下载或保存等），请查看超类文档。

    参数：
        adapters (`List[T2IAdapter]`, *可选*, 默认为 None):
            一个 `T2IAdapter` 模型实例的列表。
    """
    # 初始化 MultiAdapter 类，接受一组适配器
        def __init__(self, adapters: List["T2IAdapter"]):
            # 调用父类的初始化方法
            super(MultiAdapter, self).__init__()
    
            # 计算适配器的数量
            self.num_adapter = len(adapters)
            # 将适配器列表转换为 PyTorch 的 ModuleList
            self.adapters = nn.ModuleList(adapters)
    
            # 检查适配器数量，至少需要一个
            if len(adapters) == 0:
                raise ValueError("Expecting at least one adapter")
    
            # 检查适配器数量，如果只有一个，建议使用 T2IAdapter
            if len(adapters) == 1:
                raise ValueError("For a single adapter, please use the `T2IAdapter` class instead of `MultiAdapter`")
    
            # 获取第一个适配器的总缩放因子
            first_adapter_total_downscale_factor = adapters[0].total_downscale_factor
            # 获取第一个适配器的缩放因子
            first_adapter_downscale_factor = adapters[0].downscale_factor
            # 遍历剩余的适配器，检查它们的缩放因子
            for idx in range(1, len(adapters)):
                if (
                    adapters[idx].total_downscale_factor != first_adapter_total_downscale_factor
                    or adapters[idx].downscale_factor != first_adapter_downscale_factor
                ):
                    # 如果缩放因子不一致，抛出错误
                    raise ValueError(
                        f"Expecting all adapters to have the same downscaling behavior, but got:\n"
                        f"adapters[0].total_downscale_factor={first_adapter_total_downscale_factor}\n"
                        f"adapters[0].downscale_factor={first_adapter_downscale_factor}\n"
                        f"adapter[`{idx}`].total_downscale_factor={adapters[idx].total_downscale_factor}\n"
                        f"adapter[`{idx}`].downscale_factor={adapters[idx].downscale_factor}"
                    )
    
            # 设置 MultiAdapter 的总缩放因子
            self.total_downscale_factor = first_adapter_total_downscale_factor
            # 设置 MultiAdapter 的缩放因子
            self.downscale_factor = first_adapter_downscale_factor
    # 定义前向传播方法，接受输入张量和可选的适配器权重
    def forward(self, xs: torch.Tensor, adapter_weights: Optional[List[float]] = None) -> List[torch.Tensor]:
        r"""
        Args:
            xs (`torch.Tensor`):
                (batch, channel, height, width) 输入的图像张量，多个适配器模型沿维度 1 连接，
                `channel` 应等于 `num_adapter` * "图像的通道数"。
            adapter_weights (`List[float]`, *optional*, defaults to None):
                表示在将每个适配器的输出相加之前，要乘以的权重的浮点数列表。
        """
        # 如果没有提供适配器权重，则初始化为每个适配器权重相等的张量
        if adapter_weights is None:
            adapter_weights = torch.tensor([1 / self.num_adapter] * self.num_adapter)
        else:
            # 将提供的适配器权重转换为张量
            adapter_weights = torch.tensor(adapter_weights)

        # 初始化累计状态为 None，用于存储加权特征
        accume_state = None
        # 遍历输入张量、适配器权重和适配器模型
        for x, w, adapter in zip(xs, adapter_weights, self.adapters):
            # 使用适配器模型处理输入张量以提取特征
            features = adapter(x)
            # 如果累计状态为空，初始化它
            if accume_state is None:
                accume_state = features
                # 根据当前适配器的权重调整累计状态的特征
                for i in range(len(accume_state)):
                    accume_state[i] = w * accume_state[i]
            else:
                # 如果累计状态已经存在，将新特征加到累计状态中
                for i in range(len(features)):
                    accume_state[i] += w * features[i]
        # 返回加权后的累计状态
        return accume_state

    # 定义保存预训练模型的方法，接收多个参数
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ):
        """
        保存模型及其配置文件到指定目录，以便后续通过 `[`~models.adapter.MultiAdapter.from_pretrained`]` 类方法重新加载。

        参数：
            save_directory (`str` 或 `os.PathLike`):
                要保存的目录。如果目录不存在，则会创建。
            is_main_process (`bool`, *可选*, 默认为 `True`):
                调用此函数的进程是否为主进程。在分布式训练（如 TPU）中有用，仅在主进程上设置 `is_main_process=True` 以避免竞争条件。
            save_function (`Callable`):
                用于保存状态字典的函数。在分布式训练（如 TPU）时，可以用其他方法替换 `torch.save`。可通过环境变量 `DIFFUSERS_SAVE_MODE` 配置。
            safe_serialization (`bool`, *可选*, 默认为 `True`):
                是否使用 `safetensors` 保存模型，或使用传统的 PyTorch 方法（使用 `pickle`）。
            variant (`str`, *可选*):
                如果指定，权重将以 pytorch_model.<variant>.bin 格式保存。
        """
        # 初始化索引为 0
        idx = 0
        # 设置保存模型的路径
        model_path_to_save = save_directory
        # 遍历所有适配器
        for adapter in self.adapters:
            # 调用适配器的保存方法，将模型及其配置保存到指定路径
            adapter.save_pretrained(
                model_path_to_save,
                is_main_process=is_main_process,
                save_function=save_function,
                safe_serialization=safe_serialization,
                variant=variant,
            )

            # 索引加一，用于下一个模型路径
            idx += 1
            # 更新模型保存路径，添加索引
            model_path_to_save = model_path_to_save + f"_{idx}"

    # 定义类方法装饰器
    @classmethod
# 定义一个 T2IAdapter 类，继承自 ModelMixin 和 ConfigMixin
class T2IAdapter(ModelMixin, ConfigMixin):
    r"""
    一个简单的类似 ResNet 的模型，接受包含控制信号（如关键姿态和深度）的图像。该模型
    生成多个特征图，作为 [`UNet2DConditionModel`] 的额外条件。模型的架构遵循
    [Adapter](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L97)
    和
    [AdapterLight](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L235) 的原始实现。

    该模型继承自 [`ModelMixin`]。有关库为所有模型实现的通用方法（如下载或保存等），请查看超类文档。

    参数：
        in_channels (`int`, *可选*, 默认为 3):
            Adapter 输入的通道数（*控制图像*）。如果使用灰度图像作为 *控制图像*，请将此参数设置为 1。
        channels (`List[int]`, *可选*, 默认为 `(320, 640, 1280, 1280)`):
            每个下采样块输出隐藏状态的通道数。`len(block_out_channels)` 还将决定 Adapter 中下采样块的数量。
        num_res_blocks (`int`, *可选*, 默认为 2):
            每个下采样块中的 ResNet 块数。
        downscale_factor (`int`, *可选*, 默认为 8):
            决定 Adapter 总体下采样因子的因素。
        adapter_type (`str`, *可选*, 默认为 `full_adapter`):
            要使用的 Adapter 类型。选择 `full_adapter`、`full_adapter_xl` 或 `light_adapter`。
    """

    # 注册初始化函数到配置中
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,  # 设置输入通道数，默认为 3
        channels: List[int] = [320, 640, 1280, 1280],  # 设置每个下采样块的输出通道数，默认为给定列表
        num_res_blocks: int = 2,  # 设置每个下采样块中的 ResNet 块数，默认为 2
        downscale_factor: int = 8,  # 设置下采样因子，默认为 8
        adapter_type: str = "full_adapter",  # 设置 Adapter 类型，默认为 'full_adapter'
    ):
        super().__init__()  # 调用父类的初始化方法

        # 根据 adapter_type 的值实例化相应的 Adapter
        if adapter_type == "full_adapter":
            self.adapter = FullAdapter(in_channels, channels, num_res_blocks, downscale_factor)  # 实例化 FullAdapter
        elif adapter_type == "full_adapter_xl":
            self.adapter = FullAdapterXL(in_channels, channels, num_res_blocks, downscale_factor)  # 实例化 FullAdapterXL
        elif adapter_type == "light_adapter":
            self.adapter = LightAdapter(in_channels, channels, num_res_blocks, downscale_factor)  # 实例化 LightAdapter
        else:
            raise ValueError(  # 如果 adapter_type 不合法，抛出异常
                f"Unsupported adapter_type: '{adapter_type}'. Choose either 'full_adapter' or "
                "'full_adapter_xl' or 'light_adapter'."  # 提示支持的 Adapter 类型
            )
    # 定义前向传播函数，接收一个张量 x，并返回特征张量列表
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 文档字符串，描述该函数的功能和输出
        r"""
        该函数通过适配器模型处理输入张量 `x`，并返回特征张量列表，
        每个张量表示从输入中提取的不同尺度的信息。列表的长度由
        在初始化时指定的 `channels` 和 `num_res_blocks` 参数中的
        下采样块数量决定。
        """
        # 调用适配器的前向方法，处理输入张量并返回结果
        return self.adapter(x)
    
    # 定义属性 total_downscale_factor，返回适配器的总下采样因子
    @property
    def total_downscale_factor(self):
        # 返回适配器的总下采样因子
        return self.adapter.total_downscale_factor
    
    # 定义属性 downscale_factor，表示初始像素无序操作中的下采样因子
    @property
    def downscale_factor(self):
        # 文档字符串，描述下采样因子的作用和可能的异常情况
        """在 T2I-Adapter 的初始像素无序操作中应用的下采样因子。如果输入图像的维度
        不能被下采样因子整除，则会引发异常。
        """
        # 返回适配器无序操作中的下采样因子
        return self.adapter.unshuffle.downscale_factor
# 全适配器类
class FullAdapter(nn.Module):
    r"""
    详细信息请参见 [`T2IAdapter`]。
    """

    # 初始化方法，设置输入通道、通道列表、残差块数量和下采样因子
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
    ):
        # 调用父类初始化方法
        super().__init__()

        # 根据下采样因子计算输入通道数
        in_channels = in_channels * downscale_factor**2

        # 创建像素反混洗层
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        # 创建输入卷积层
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # 创建适配器块列表
        self.body = nn.ModuleList(
            [
                # 添加第一个适配器块
                AdapterBlock(channels[0], channels[0], num_res_blocks),
                # 添加后续适配器块，带下采样
                *[
                    AdapterBlock(channels[i - 1], channels[i], num_res_blocks, down=True)
                    for i in range(1, len(channels))
                ],
            ]
        )

        # 计算总的下采样因子
        self.total_downscale_factor = downscale_factor * 2 ** (len(channels) - 1)

    # 前向传播方法
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        该方法通过 FullAdapter 模型处理输入张量 `x`，执行像素反混洗、卷积和适配器块堆栈的操作。
        返回一个特征张量列表，每个张量在处理的不同阶段捕获信息。特征张量的数量由初始化时指定的下采样块数量决定。
        """
        # 反混洗输入张量
        x = self.unshuffle(x)
        # 通过输入卷积层处理
        x = self.conv_in(x)

        # 初始化特征列表
        features = []

        # 遍历适配器块并处理输入
        for block in self.body:
            x = block(x)
            # 将特征添加到列表中
            features.append(x)

        # 返回特征列表
        return features


# 全适配器 XL 类
class FullAdapterXL(nn.Module):
    r"""
    详细信息请参见 [`T2IAdapter`]。
    """

    # 初始化方法，设置输入通道、通道列表、残差块数量和下采样因子
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 16,
    ):
        # 调用父类初始化方法
        super().__init__()

        # 根据下采样因子计算输入通道数
        in_channels = in_channels * downscale_factor**2

        # 创建像素反混洗层
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        # 创建输入卷积层
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # 初始化适配器块列表
        self.body = []
        # 遍历通道列表，创建适配器块
        for i in range(len(channels)):
            if i == 1:
                # 为第二个通道添加适配器块
                self.body.append(AdapterBlock(channels[i - 1], channels[i], num_res_blocks))
            elif i == 2:
                # 为第三个通道添加带下采样的适配器块
                self.body.append(AdapterBlock(channels[i - 1], channels[i], num_res_blocks, down=True))
            else:
                # 为其他通道添加适配器块
                self.body.append(AdapterBlock(channels[i], channels[i], num_res_blocks))

        # 将适配器块列表转换为 ModuleList
        self.body = nn.ModuleList(self.body)
        # XL 只有一个下采样适配器块
        self.total_downscale_factor = downscale_factor * 2
    # 定义一个前向传播方法，输入为张量 x，返回特征张量的列表
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        该方法接受张量 x 作为输入，并通过 FullAdapterXL 模型处理。包括像素解混淆、应用卷积层，并将每个块追加到特征张量列表中。
        """
        # 对输入张量进行像素解混淆操作
        x = self.unshuffle(x)
        # 将解混淆后的张量通过输入卷积层
        x = self.conv_in(x)
    
        # 初始化一个空列表以存储特征张量
        features = []
    
        # 遍历模型主体中的每个块
        for block in self.body:
            # 将当前张量通过块处理
            x = block(x)
            # 将处理后的张量追加到特征列表中
            features.append(x)
    
        # 返回特征张量的列表
        return features
# AdapterBlock 类是一个辅助模型，包含多个类似 ResNet 的模块，用于 FullAdapter 和 FullAdapterXL 模型
class AdapterBlock(nn.Module):
    r"""
    AdapterBlock 是一个包含多个 ResNet 样式块的辅助模型。它在 `FullAdapter` 和
    `FullAdapterXL` 模型中使用。

    参数：
        in_channels (`int`):
            AdapterBlock 输入的通道数。
        out_channels (`int`):
            AdapterBlock 输出的通道数。
        num_res_blocks (`int`):
            AdapterBlock 中 ResNet 块的数量。
        down (`bool`, *可选*, 默认为 `False`):
            是否对 AdapterBlock 的输入进行下采样。
    """

    # 初始化 AdapterBlock 类，接收输入输出通道数及 ResNet 块数量
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False):
        super().__init__()  # 调用父类的构造函数

        self.downsample = None  # 初始化下采样层为 None
        if down:  # 如果需要下采样
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 创建平均池化层

        self.in_conv = None  # 初始化输入卷积层为 None
        if in_channels != out_channels:  # 如果输入通道与输出通道不相等
            self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 创建 1x1 卷积层

        # 创建一系列的 ResNet 块，数量由 num_res_blocks 指定
        self.resnets = nn.Sequential(
            *[AdapterResnetBlock(out_channels) for _ in range(num_res_blocks)],
        )

    # 定义前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        此方法接收张量 x 作为输入，执行下采样和卷积操作（如果指定了 self.downsample 和 self.in_conv）。
        然后，它对输入张量应用一系列残差块。
        """
        if self.downsample is not None:  # 如果存在下采样层
            x = self.downsample(x)  # 执行下采样操作

        if self.in_conv is not None:  # 如果存在输入卷积层
            x = self.in_conv(x)  # 执行卷积操作

        x = self.resnets(x)  # 将输入传递通过一系列 ResNet 块

        return x  # 返回处理后的张量


# AdapterResnetBlock 类是一个实现 ResNet 样式块的辅助模型
class AdapterResnetBlock(nn.Module):
    r"""
    `AdapterResnetBlock` 是一个实现 ResNet 样式块的辅助模型。

    参数：
        channels (`int`):
            AdapterResnetBlock 输入和输出的通道数。
    """

    # 初始化 AdapterResnetBlock 类，接收通道数
    def __init__(self, channels: int):
        super().__init__()  # 调用父类的构造函数
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 创建 3x3 卷积层
        self.act = nn.ReLU()  # 创建 ReLU 激活函数
        self.block2 = nn.Conv2d(channels, channels, kernel_size=1)  # 创建 1x1 卷积层

    # 定义前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        此方法接收输入张量 x，并对其应用卷积层、ReLU 激活和另一个卷积层。返回与输入张量的相加结果。
        """

        h = self.act(self.block1(x))  # 先通过第一个卷积层并应用激活函数
        h = self.block2(h)  # 再通过第二个卷积层

        return h + x  # 返回卷积结果与输入的和


# LightAdapter 类是一个轻量适配器模型
class LightAdapter(nn.Module):
    r"""
    有关更多信息，请参阅 [`T2IAdapter`]。
    """

    # 初始化 LightAdapter 类，设置输入通道数、通道列表、ResNet 块数量及下采样因子
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280],
        num_res_blocks: int = 4,
        downscale_factor: int = 8,
    # 初始化方法
        ):
            # 调用父类的初始化方法
            super().__init__()
    
            # 计算输入通道数，考虑下采样因子
            in_channels = in_channels * downscale_factor**2
    
            # 初始化像素反shuffle操作，依据下采样因子
            self.unshuffle = nn.PixelUnshuffle(downscale_factor)
    
            # 创建一个模块列表，包含多个 LightAdapterBlock
            self.body = nn.ModuleList(
                [
                    # 第一个 LightAdapterBlock，处理输入通道到第一个输出通道
                    LightAdapterBlock(in_channels, channels[0], num_res_blocks),
                    # 使用列表推导创建后续的 LightAdapterBlock，处理每对通道
                    *[
                        LightAdapterBlock(channels[i], channels[i + 1], num_res_blocks, down=True)
                        for i in range(len(channels) - 1)
                    ],
                    # 最后一个 LightAdapterBlock，处理最后一组通道
                    LightAdapterBlock(channels[-1], channels[-1], num_res_blocks, down=True),
                ]
            )
    
            # 计算总下采样因子
            self.total_downscale_factor = downscale_factor * (2 ** len(channels))
    
        # 前向传播方法
        def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
            r"""
            此方法接收输入张量 x，进行下采样，并将结果存入特征张量列表中。每个特征张量对应于 LightAdapter 中的不同处理级别。
            """
            # 对输入张量进行反shuffle处理
            x = self.unshuffle(x)
    
            # 初始化特征列表
            features = []
    
            # 遍历模块列表中的每个块
            for block in self.body:
                # 通过当前块处理输入张量
                x = block(x)
                # 将处理后的张量添加到特征列表中
                features.append(x)
    
            # 返回特征列表
            return features
# LightAdapterBlock 类是一个帮助模型，包含多个 LightAdapterResnetBlocks，用于 LightAdapter 模型中
class LightAdapterBlock(nn.Module):
    r"""
    A `LightAdapterBlock` is a helper model that contains multiple `LightAdapterResnetBlocks`. It is used in the
    `LightAdapter` model.

    Parameters:
        in_channels (`int`):
            Number of channels of LightAdapterBlock's input.
        out_channels (`int`):
            Number of channels of LightAdapterBlock's output.
        num_res_blocks (`int`):
            Number of LightAdapterResnetBlocks in the LightAdapterBlock.
        down (`bool`, *optional*, defaults to `False`):
            Whether to perform downsampling on LightAdapterBlock's input.
    """

    # 初始化方法，接收输入输出通道数、残差块数量和是否下采样的标志
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False):
        super().__init__()  # 调用父类构造函数
        mid_channels = out_channels // 4  # 计算中间通道数

        self.downsample = None  # 初始化下采样层为 None
        if down:  # 如果需要下采样
            # 创建平均池化层，kernel_size为2，步幅为2，向上取整
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 定义输入卷积层，kernel_size为1
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        # 创建残差块序列，数量为 num_res_blocks
        self.resnets = nn.Sequential(*[LightAdapterResnetBlock(mid_channels) for _ in range(num_res_blocks)])
        # 定义输出卷积层，kernel_size为1
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    # 前向传播方法，接收输入张量 x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This method takes tensor x as input and performs downsampling if required. Then it applies in convolution
        layer, a sequence of residual blocks, and out convolutional layer.
        """
        if self.downsample is not None:  # 如果定义了下采样层
            x = self.downsample(x)  # 对输入 x 进行下采样

        x = self.in_conv(x)  # 通过输入卷积层处理 x
        x = self.resnets(x)  # 通过残差块序列处理 x
        x = self.out_conv(x)  # 通过输出卷积层处理 x

        return x  # 返回处理后的结果


# LightAdapterResnetBlock 类是一个帮助模型，实现类似 ResNet 的块，具有与 AdapterResnetBlock 略微不同的架构
class LightAdapterResnetBlock(nn.Module):
    """
    A `LightAdapterResnetBlock` is a helper model that implements a ResNet-like block with a slightly different
    architecture than `AdapterResnetBlock`.

    Parameters:
        channels (`int`):
            Number of channels of LightAdapterResnetBlock's input and output.
    """

    # 初始化方法，接收通道数
    def __init__(self, channels: int):
        super().__init__()  # 调用父类构造函数
        # 定义第一个卷积层，kernel_size为3，padding为1
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()  # 定义 ReLU 激活函数
        # 定义第二个卷积层，kernel_size为3，padding为1
        self.block2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    # 前向传播方法，接收输入张量 x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This function takes input tensor x and processes it through one convolutional layer, ReLU activation, and
        another convolutional layer and adds it to input tensor.
        """
        h = self.act(self.block1(x))  # 通过第一个卷积层和 ReLU 激活处理 x
        h = self.block2(h)  # 通过第二个卷积层处理 h

        return h + x  # 将处理结果与输入 x 相加并返回
```