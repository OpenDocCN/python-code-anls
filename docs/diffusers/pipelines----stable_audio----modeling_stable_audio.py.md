# `.\diffusers\pipelines\stable_audio\modeling_stable_audio.py`

```py
# 版权所有 2024 Stability AI 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定， 
# 根据许可证分发的软件按“原样”提供，
# 不提供任何明示或暗示的担保或条件。
# 请参阅许可证了解特定语言所规定的权限和
# 限制。

from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from math import pi  # 从 math 模块导入圆周率 π
from typing import Optional  # 从 typing 模块导入 Optional 类型

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.utils.checkpoint  # 导入 PyTorch 的检查点功能

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入配置混合和注册功能
from ...models.modeling_utils import ModelMixin  # 从模型工具导入模型混合
from ...utils import BaseOutput, logging  # 从工具模块导入基础输出和日志功能

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

class StableAudioPositionalEmbedding(nn.Module):  # 定义一个名为 StableAudioPositionalEmbedding 的类，继承自 nn.Module
    """用于连续时间的嵌入层"""

    def __init__(self, dim: int):  # 构造函数，接受一个维度参数
        super().__init__()  # 调用父类构造函数
        assert (dim % 2) == 0  # 确保维度为偶数
        half_dim = dim // 2  # 计算一半的维度
        self.weights = nn.Parameter(torch.randn(half_dim))  # 初始化权重为随机值并作为可训练参数

    def forward(self, times: torch.Tensor) -> torch.Tensor:  # 定义前向传播方法，接受一个时间张量
        times = times[..., None]  # 将时间张量扩展为最后一维
        freqs = times * self.weights[None] * 2 * pi  # 计算频率
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  # 计算频率的正弦和余弦值
        fouriered = torch.cat((times, fouriered), dim=-1)  # 将时间和频率特征拼接
        return fouriered  # 返回处理后的张量

@dataclass
class StableAudioProjectionModelOutput(BaseOutput):  # 定义一个数据类，用于稳定音频投影模型的输出
    """
    参数：
    用于稳定音频投影层输出的类。
        text_hidden_states (`torch.Tensor`，形状为 `(batch_size, sequence_length, hidden_size)`，*可选*):
            通过线性投影获得的文本编码器的隐状态序列。
        seconds_start_hidden_states (`torch.Tensor`，形状为 `(batch_size, 1, hidden_size)`，*可选*):
            通过线性投影获得的音频开始隐状态序列。
        seconds_end_hidden_states (`torch.Tensor`，形状为 `(batch_size, 1, hidden_size)`，*可选*):
            通过线性投影获得的音频结束隐状态序列。
    """

    text_hidden_states: Optional[torch.Tensor] = None  # 可选的文本隐状态张量
    seconds_start_hidden_states: Optional[torch.Tensor] = None  # 可选的音频开始隐状态张量
    seconds_end_hidden_states: Optional[torch.Tensor] = None  # 可选的音频结束隐状态张量


class StableAudioNumberConditioner(nn.Module):  # 定义一个名为 StableAudioNumberConditioner 的类，继承自 nn.Module
    """
    一个简单的线性投影模型，将数字映射到潜在空间。
    # 参数说明
    Args:
        number_embedding_dim (`int`):
            # 数字嵌入的维度
            Dimensionality of the number embeddings.
        min_value (`int`):
            # 秒数条件模块的最小值
            The minimum value of the seconds number conditioning modules.
        max_value (`int`):
            # 秒数条件模块的最大值
            The maximum value of the seconds number conditioning modules
        internal_dim (`int`):
            # 中间数字隐藏状态的维度
            Dimensionality of the intermediate number hidden states.
    """

    # 初始化方法
    def __init__(
        self,
        # 数字嵌入的维度
        number_embedding_dim,
        # 秒数条件模块的最小值
        min_value,
        # 秒数条件模块的最大值
        max_value,
        # 中间数字隐藏状态的维度，默认为256
        internal_dim: Optional[int] = 256,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 创建时间位置嵌入层，包含稳定音频位置嵌入和线性变换
        self.time_positional_embedding = nn.Sequential(
            StableAudioPositionalEmbedding(internal_dim),
            # 从内部维度到数字嵌入维度的线性转换
            nn.Linear(in_features=internal_dim + 1, out_features=number_embedding_dim),
        )

        # 保存数字嵌入的维度
        self.number_embedding_dim = number_embedding_dim
        # 保存最小值
        self.min_value = min_value
        # 保存最大值
        self.max_value = max_value

    # 前向传播方法
    def forward(
        self,
        # 输入的浮点张量
        floats: torch.Tensor,
    ):
        # 将浮点数限制在最小值和最大值之间
        floats = floats.clamp(self.min_value, self.max_value)

        # 将浮点数归一化到[0, 1]范围
        normalized_floats = (floats - self.min_value) / (self.max_value - self.min_value)

        # 将浮点数转换为嵌入器相同的类型
        embedder_dtype = next(self.time_positional_embedding.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        # 通过时间位置嵌入层生成嵌入
        embedding = self.time_positional_embedding(normalized_floats)
        # 调整嵌入的形状
        float_embeds = embedding.view(-1, 1, self.number_embedding_dim)

        # 返回浮点数嵌入
        return float_embeds
# 定义一个稳定音频投影模型类，继承自 ModelMixin 和 ConfigMixin
class StableAudioProjectionModel(ModelMixin, ConfigMixin):
    """
    一个简单的线性投影模型，用于将条件值映射到共享的潜在空间。

    参数：
        text_encoder_dim (`int`):
            文本编码器 (T5) 的文本嵌入维度。
        conditioning_dim (`int`):
            输出条件张量的维度。
        min_value (`int`):
            秒数条件模块的最小值。
        max_value (`int`):
            秒数条件模块的最大值。
    """

    # 注册构造函数到配置
    @register_to_config
    def __init__(self, text_encoder_dim, conditioning_dim, min_value, max_value):
        # 调用父类构造函数
        super().__init__()
        # 根据条件维度选择合适的投影方式
        self.text_projection = (
            nn.Identity() if conditioning_dim == text_encoder_dim else nn.Linear(text_encoder_dim, conditioning_dim)
        )
        # 初始化开始时间条件模块
        self.start_number_conditioner = StableAudioNumberConditioner(conditioning_dim, min_value, max_value)
        # 初始化结束时间条件模块
        self.end_number_conditioner = StableAudioNumberConditioner(conditioning_dim, min_value, max_value)

    # 定义前向传播方法
    def forward(
        self,
        text_hidden_states: Optional[torch.Tensor] = None,
        start_seconds: Optional[torch.Tensor] = None,
        end_seconds: Optional[torch.Tensor] = None,
    ):
        # 如果没有文本隐藏状态，则使用输入的文本隐藏状态，否则进行投影
        text_hidden_states = (
            text_hidden_states if text_hidden_states is None else self.text_projection(text_hidden_states)
        )
        # 如果没有开始秒数，则使用输入的开始秒数，否则进行条件处理
        seconds_start_hidden_states = (
            start_seconds if start_seconds is None else self.start_number_conditioner(start_seconds)
        )
        # 如果没有结束秒数，则使用输入的结束秒数，否则进行条件处理
        seconds_end_hidden_states = end_seconds if end_seconds is None else self.end_number_conditioner(end_seconds)

        # 返回一个稳定音频投影模型输出对象，包含所有隐藏状态
        return StableAudioProjectionModelOutput(
            text_hidden_states=text_hidden_states,
            seconds_start_hidden_states=seconds_start_hidden_states,
            seconds_end_hidden_states=seconds_end_hidden_states,
        )
```