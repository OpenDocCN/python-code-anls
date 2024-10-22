# `.\diffusers\models\unets\unet_2d.py`

```py
# 版权声明，表示该代码由 HuggingFace 团队所有
# 
# 根据 Apache 2.0 许可证进行许可；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下地址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，软件按 "原样" 分发，
# 不提供任何形式的保证或条件，无论是明示或暗示。
# 请参阅许可证以了解有关权限和
# 限制的具体信息。
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Optional, Tuple, Union  # 从 typing 模块导入类型提示相关的类

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入配置混合类和注册函数
from ...utils import BaseOutput  # 从工具模块导入基础输出类
from ..embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps  # 从嵌入模块导入相关类
from ..modeling_utils import ModelMixin  # 从建模工具导入模型混合类
from .unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block  # 从 UNet 2D 块导入相关组件


@dataclass
class UNet2DOutput(BaseOutput):  # 定义 UNet2DOutput 类，继承自 BaseOutput
    """
    [`UNet2DModel`] 的输出类。

    参数：
        sample (`torch.Tensor` 形状为 `(batch_size, num_channels, height, width)`):
            从模型最后一层输出的隐藏状态。
    """

    sample: torch.Tensor  # 定义输出样本的属性，类型为 torch.Tensor


class UNet2DModel(ModelMixin, ConfigMixin):  # 定义 UNet2DModel 类，继承自 ModelMixin 和 ConfigMixin
    r"""
    一个 2D UNet 模型，接收一个有噪声的样本和一个时间步，返回一个样本形状的输出。

    此模型继承自 [`ModelMixin`]。请查看超类文档以了解其为所有模型实现的通用方法
    （例如下载或保存）。
    """

    @register_to_config  # 使用装饰器将该方法注册到配置中
    def __init__(  # 定义初始化方法
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,  # 可选的样本大小，可以是整数或整数元组
        in_channels: int = 3,  # 输入通道数，默认为 3
        out_channels: int = 3,  # 输出通道数，默认为 3
        center_input_sample: bool = False,  # 是否将输入样本居中，默认为 False
        time_embedding_type: str = "positional",  # 时间嵌入类型，默认为 "positional"
        freq_shift: int = 0,  # 频率偏移量，默认为 0
        flip_sin_to_cos: bool = True,  # 是否将正弦函数翻转为余弦函数，默认为 True
        down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),  # 下采样块类型
        up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),  # 上采样块类型
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),  # 各块输出通道数
        layers_per_block: int = 2,  # 每个块的层数，默认为 2
        mid_block_scale_factor: float = 1,  # 中间块缩放因子，默认为 1
        downsample_padding: int = 1,  # 下采样的填充大小，默认为 1
        downsample_type: str = "conv",  # 下采样类型，默认为卷积
        upsample_type: str = "conv",  # 上采样类型，默认为卷积
        dropout: float = 0.0,  # dropout 概率，默认为 0.0
        act_fn: str = "silu",  # 激活函数类型，默认为 "silu"
        attention_head_dim: Optional[int] = 8,  # 注意力头维度，默认为 8
        norm_num_groups: int = 32,  # 规范化组数量，默认为 32
        attn_norm_num_groups: Optional[int] = None,  # 注意力规范化组数量，可选
        norm_eps: float = 1e-5,  # 规范化的 epsilon 值，默认为 1e-5
        resnet_time_scale_shift: str = "default",  # ResNet 时间缩放偏移类型，默认为 "default"
        add_attention: bool = True,  # 是否添加注意力机制，默认为 True
        class_embed_type: Optional[str] = None,  # 类别嵌入类型，可选
        num_class_embeds: Optional[int] = None,  # 类别嵌入数量，可选
        num_train_timesteps: Optional[int] = None,  # 训练时间步数量，可选
    # 定义一个名为 forward 的方法
        def forward(
            # 输入参数 sample，类型为 torch.Tensor，表示样本数据
            self,
            sample: torch.Tensor,
            # 输入参数 timestep，可以是 torch.Tensor、float 或 int，表示时间步
            timestep: Union[torch.Tensor, float, int],
            # 可选参数 class_labels，类型为 torch.Tensor，表示分类标签，默认为 None
            class_labels: Optional[torch.Tensor] = None,
            # 可选参数 return_dict，类型为 bool，表示是否以字典形式返回结果，默认为 True
            return_dict: bool = True,
```