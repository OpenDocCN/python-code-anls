# `.\diffusers\models\transformers\stable_audio_transformer.py`

```py
# 版权声明，注明版权归属
# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# 根据 Apache License 2.0 授权协议进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 只有在遵守许可证的情况下，您才能使用此文件
# you may not use this file except in compliance with the License.
# 您可以在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，软件按“原样”分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的保证或条件，明示或暗示
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参见许可证以获取有关权限和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.


# 导入类型注解
from typing import Any, Dict, Optional, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的检查点工具
import torch.utils.checkpoint

# 从配置工具导入相关类
from ...configuration_utils import ConfigMixin, register_to_config
# 从注意力模块导入前馈网络
from ...models.attention import FeedForward
# 从注意力处理器模块导入多个类
from ...models.attention_processor import (
    Attention,
    AttentionProcessor,
    StableAudioAttnProcessor2_0,
)
# 从建模工具导入模型混合类
from ...models.modeling_utils import ModelMixin
# 从变换器模型导入输出类
from ...models.transformers.transformer_2d import Transformer2DModelOutput
# 导入实用工具
from ...utils import is_torch_version, logging
# 导入可能允许图形中的工具函数
from ...utils.torch_utils import maybe_allow_in_graph

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableAudioGaussianFourierProjection(nn.Module):
    """用于噪声级别的高斯傅里叶嵌入。"""

    # 从 diffusers.models.embeddings.GaussianFourierProjection.__init__ 复制的内容
    def __init__(
        self, embedding_size: int = 256, scale: float = 1.0, set_W_to_weight=True, log=True, flip_sin_to_cos=False
    ):
        super().__init__()  # 调用父类的构造函数
        # 初始化权重为随机值，且不需要计算梯度
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.log = log  # 是否对输入取对数的标志
        self.flip_sin_to_cos = flip_sin_to_cos  # 是否翻转正弦和余弦的顺序

        if set_W_to_weight:  # 如果设置将 W 赋值给权重
            # 之后将删除此行
            del self.weight  # 删除原有权重
            # 初始化 W 为随机值，并不计算梯度
            self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
            self.weight = self.W  # 将 W 赋值给权重
            del self.W  # 删除 W

    def forward(self, x):
        # 如果 log 为 True，则对输入进行对数变换
        if self.log:
            x = torch.log(x)

        # 计算投影，使用 2π 乘以输入和权重的外积
        x_proj = 2 * np.pi * x[:, None] @ self.weight[None, :]

        if self.flip_sin_to_cos:  # 如果翻转正弦和余弦
            # 连接余弦和正弦，形成输出
            out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            # 连接正弦和余弦，形成输出
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out  # 返回输出


@maybe_allow_in_graph  # 可能允许在计算图中使用
class StableAudioDiTBlock(nn.Module):
    r"""
    用于稳定音频模型的变换器块 (https://github.com/Stability-AI/stable-audio-tools)。允许跳跃连接和 QKNorm
    # 参数说明
    Parameters:
        dim (`int`): 输入和输出的通道数。
        num_attention_heads (`int`): 查询状态所使用的头数。
        num_key_value_attention_heads (`int`): 键和值状态所使用的头数。
        attention_head_dim (`int`): 每个头中的通道数。
        dropout (`float`, *optional*, defaults to 0.0): 使用的丢弃概率。
        cross_attention_dim (`int`, *optional*): 跨注意力的 encoder_hidden_states 向量的大小。
        upcast_attention (`bool`, *optional*):
            是否将注意力计算上升到 float32。这对混合精度训练很有用。
    """

    # 初始化函数
    def __init__(
        self,
        dim: int,  # 输入和输出的通道数
        num_attention_heads: int,  # 查询状态的头数
        num_key_value_attention_heads: int,  # 键和值状态的头数
        attention_head_dim: int,  # 每个头的通道数
        dropout=0.0,  # 丢弃概率，默认为0
        cross_attention_dim: Optional[int] = None,  # 跨注意力的维度，可选
        upcast_attention: bool = False,  # 是否上升到 float32，默认为 False
        norm_eps: float = 1e-5,  # 归一化层的小常数
        ff_inner_dim: Optional[int] = None,  # 前馈层内部维度，可选
    ):
        super().__init__()  # 调用父类构造函数
        # 定义三个模块。每个模块都有自己的归一化层。
        # 1. 自注意力层
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)  # 自注意力的归一化层
        self.attn1 = Attention(  # 自注意力模块
            query_dim=dim,  # 查询维度
            heads=num_attention_heads,  # 头数
            dim_head=attention_head_dim,  # 每个头的维度
            dropout=dropout,  # 丢弃概率
            bias=False,  # 不使用偏置
            upcast_attention=upcast_attention,  # 是否上升到 float32
            out_bias=False,  # 不使用输出偏置
            processor=StableAudioAttnProcessor2_0(),  # 使用的处理器
        )

        # 2. 跨注意力层
        self.norm2 = nn.LayerNorm(dim, norm_eps, True)  # 跨注意力的归一化层

        self.attn2 = Attention(  # 跨注意力模块
            query_dim=dim,  # 查询维度
            cross_attention_dim=cross_attention_dim,  # 跨注意力维度
            heads=num_attention_heads,  # 头数
            dim_head=attention_head_dim,  # 每个头的维度
            kv_heads=num_key_value_attention_heads,  # 键和值的头数
            dropout=dropout,  # 丢弃概率
            bias=False,  # 不使用偏置
            upcast_attention=upcast_attention,  # 是否上升到 float32
            out_bias=False,  # 不使用输出偏置
            processor=StableAudioAttnProcessor2_0(),  # 使用的处理器
        )  # 如果 encoder_hidden_states 为 None，则为自注意力

        # 3. 前馈层
        self.norm3 = nn.LayerNorm(dim, norm_eps, True)  # 前馈层的归一化层
        self.ff = FeedForward(  # 前馈神经网络模块
            dim,  # 输入维度
            dropout=dropout,  # 丢弃概率
            activation_fn="swiglu",  # 激活函数
            final_dropout=False,  # 最后是否丢弃
            inner_dim=ff_inner_dim,  # 内部维度
            bias=True,  # 使用偏置
        )

        # 将块大小默认设置为 None
        self._chunk_size = None  # 块大小
        self._chunk_dim = 0  # 块维度

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # 设置块前馈
        self._chunk_size = chunk_size  # 设置块大小
        self._chunk_dim = dim  # 设置块维度
    # 定义前向传播方法，接收隐藏状态和可选的注意力掩码等参数
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            rotary_embedding: Optional[torch.FloatTensor] = None,
        ) -> torch.Tensor:
            # 注意：在后续计算中，归一化总是应用于实际计算之前。
            # 0. 自注意力
            # 对输入的隐藏状态进行归一化处理
            norm_hidden_states = self.norm1(hidden_states)
    
            # 计算自注意力输出
            attn_output = self.attn1(
                norm_hidden_states,
                attention_mask=attention_mask,
                rotary_emb=rotary_embedding,
            )
    
            # 将自注意力输出与原始隐藏状态相加
            hidden_states = attn_output + hidden_states
    
            # 2. 跨注意力
            # 对更新后的隐藏状态进行归一化处理
            norm_hidden_states = self.norm2(hidden_states)
    
            # 计算跨注意力输出
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            # 将跨注意力输出与更新后的隐藏状态相加
            hidden_states = attn_output + hidden_states
    
            # 3. 前馈网络
            # 对隐藏状态进行归一化处理
            norm_hidden_states = self.norm3(hidden_states)
            # 计算前馈网络输出
            ff_output = self.ff(norm_hidden_states)
    
            # 将前馈网络输出与当前隐藏状态相加
            hidden_states = ff_output + hidden_states
    
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个名为 StableAudioDiTModel 的类，继承自 ModelMixin 和 ConfigMixin
class StableAudioDiTModel(ModelMixin, ConfigMixin):
    """
    Stable Audio 中引入的扩散变换器模型。

    参考文献：https://github.com/Stability-AI/stable-audio-tools

    参数：
        sample_size ( `int`, *可选*, 默认值为 1024)：输入样本的大小。
        in_channels (`int`, *可选*, 默认值为 64)：输入中的通道数。
        num_layers (`int`, *可选*, 默认值为 24)：使用的变换器块的层数。
        attention_head_dim (`int`, *可选*, 默认值为 64)：每个头的通道数。
        num_attention_heads (`int`, *可选*, 默认值为 24)：用于查询状态的头数。
        num_key_value_attention_heads (`int`, *可选*, 默认值为 12)：
            用于键和值状态的头数。
        out_channels (`int`, 默认值为 64)：输出通道的数量。
        cross_attention_dim ( `int`, *可选*, 默认值为 768)：交叉注意力投影的维度。
        time_proj_dim ( `int`, *可选*, 默认值为 256)：时间步内投影的维度。
        global_states_input_dim ( `int`, *可选*, 默认值为 1536)：
            全局隐藏状态投影的输入维度。
        cross_attention_input_dim ( `int`, *可选*, 默认值为 768)：
            交叉注意力投影的输入维度。
    """

    # 支持梯度检查点
    _supports_gradient_checkpointing = True

    # 注册到配置的构造函数
    @register_to_config
    def __init__(
        # 输入样本的大小，默认为1024
        self,
        sample_size: int = 1024,
        # 输入的通道数，默认为64
        in_channels: int = 64,
        # 变换器块的层数，默认为24
        num_layers: int = 24,
        # 每个头的通道数，默认为64
        attention_head_dim: int = 64,
        # 查询状态的头数，默认为24
        num_attention_heads: int = 24,
        # 键和值状态的头数，默认为12
        num_key_value_attention_heads: int = 12,
        # 输出通道的数量，默认为64
        out_channels: int = 64,
        # 交叉注意力投影的维度，默认为768
        cross_attention_dim: int = 768,
        # 时间步内投影的维度，默认为256
        time_proj_dim: int = 256,
        # 全局隐藏状态投影的输入维度，默认为1536
        global_states_input_dim: int = 1536,
        # 交叉注意力投影的输入维度，默认为768
        cross_attention_input_dim: int = 768,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置样本大小
        self.sample_size = sample_size
        # 设置输出通道数
        self.out_channels = out_channels
        # 计算内部维度，等于注意力头数量乘以每个头的维度
        self.inner_dim = num_attention_heads * attention_head_dim

        # 创建稳定音频高斯傅里叶投影对象，embedding_size 为时间投影维度的一半
        self.time_proj = StableAudioGaussianFourierProjection(
            embedding_size=time_proj_dim // 2,
            flip_sin_to_cos=True,  # 是否翻转正弦和余弦
            log=False,  # 是否使用对数
            set_W_to_weight=False,  # 是否将 W 设置为权重
        )

        # 时间步投影的神经网络序列，包含两个线性层和一个激活函数
        self.timestep_proj = nn.Sequential(
            nn.Linear(time_proj_dim, self.inner_dim, bias=True),  # 输入为 time_proj_dim，输出为 inner_dim
            nn.SiLU(),  # 使用 SiLU 激活函数
            nn.Linear(self.inner_dim, self.inner_dim, bias=True),  # 再次投影到 inner_dim
        )

        # 全局状态投影的神经网络序列，包含两个线性层和一个激活函数
        self.global_proj = nn.Sequential(
            nn.Linear(global_states_input_dim, self.inner_dim, bias=False),  # 输入为 global_states_input_dim，输出为 inner_dim
            nn.SiLU(),  # 使用 SiLU 激活函数
            nn.Linear(self.inner_dim, self.inner_dim, bias=False),  # 再次投影到 inner_dim
        )

        # 交叉注意力投影的神经网络序列，包含两个线性层和一个激活函数
        self.cross_attention_proj = nn.Sequential(
            nn.Linear(cross_attention_input_dim, cross_attention_dim, bias=False),  # 输入为 cross_attention_input_dim，输出为 cross_attention_dim
            nn.SiLU(),  # 使用 SiLU 激活函数
            nn.Linear(cross_attention_dim, cross_attention_dim, bias=False),  # 再次投影到 cross_attention_dim
        )

        # 一维卷积层，用于预处理，卷积核大小为 1，不使用偏置
        self.preprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        # 输入线性层，将输入通道数投影到 inner_dim，不使用偏置
        self.proj_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        # 创建一个模块列表，包含多个 StableAudioDiTBlock
        self.transformer_blocks = nn.ModuleList(
            [
                StableAudioDiTBlock(
                    dim=self.inner_dim,  # 输入维度为 inner_dim
                    num_attention_heads=num_attention_heads,  # 注意力头数量
                    num_key_value_attention_heads=num_key_value_attention_heads,  # 键值注意力头数量
                    attention_head_dim=attention_head_dim,  # 每个注意力头的维度
                    cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                )
                for i in range(num_layers)  # 根据层数创建相应数量的块
            ]
        )

        # 输出线性层，将 inner_dim 投影到输出通道数，不使用偏置
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)
        # 一维卷积层，用于后处理，卷积核大小为 1，不使用偏置
        self.postprocess_conv = nn.Conv1d(self.out_channels, self.out_channels, 1, bias=False)

        # 初始化梯度检查点标志，默认为 False
        self.gradient_checkpointing = False

    @property
    # 从 UNet2DConditionModel 复制的属性
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # 创建一个空字典，用于存储注意力处理器
        processors = {}

        # 定义递归函数，用于添加注意力处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 如果模块有 get_processor 方法，获取处理器并添加到字典
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            # 遍历子模块，递归调用添加处理器函数
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        # 遍历当前对象的所有子模块，调用递归函数
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        # 返回包含所有处理器的字典
        return processors
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor 复制而来
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的注意力处理器。

        参数：
            processor (`dict` of `AttentionProcessor` 或 `AttentionProcessor`):
                实例化的处理器类或将作为处理器设置为 **所有** `Attention` 层的处理器类字典。

                如果 `processor` 是一个字典，键需要定义对应的交叉注意力处理器的路径。
                当设置可训练的注意力处理器时，这一点强烈推荐。

        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())

        # 如果传入的是字典且字典长度与注意力层数量不匹配，则抛出异常
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器字典，但处理器数量 {len(processor)} 与"
                f" 注意力层数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
            )

        # 定义递归设置注意力处理器的函数
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块有 set_processor 方法，则设置处理器
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    # 如果处理器不是字典，则直接设置
                    module.set_processor(processor)
                else:
                    # 从字典中弹出对应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 遍历模块的子模块，递归调用设置处理器的函数
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前对象的子模块，调用递归设置处理器的函数
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # 从 diffusers.models.transformers.hunyuan_transformer_2d.HunyuanDiT2DModel.set_default_attn_processor 复制而来，将 Hunyuan 替换为 StableAudio
    def set_default_attn_processor(self):
        """
        禁用自定义注意力处理器，并设置默认的注意力实现。
        """
        # 调用设置注意力处理器的方法，使用 StableAudioAttnProcessor2_0 实例
        self.set_attn_processor(StableAudioAttnProcessor2_0())

    # 设置梯度检查点的私有方法
    def _set_gradient_checkpointing(self, module, value=False):
        # 如果模块有 gradient_checkpointing 属性，则设置其值
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # 前向传播方法定义
    def forward(
        # 输入的隐藏状态张量
        hidden_states: torch.FloatTensor,
        # 时间步张量，默认为 None
        timestep: torch.LongTensor = None,
        # 编码器的隐藏状态张量，默认为 None
        encoder_hidden_states: torch.FloatTensor = None,
        # 全局隐藏状态张量，默认为 None
        global_hidden_states: torch.FloatTensor = None,
        # 旋转嵌入张量，默认为 None
        rotary_embedding: torch.FloatTensor = None,
        # 是否返回字典格式，默认为 True
        return_dict: bool = True,
        # 注意力掩码，默认为 None
        attention_mask: Optional[torch.LongTensor] = None,
        # 编码器的注意力掩码，默认为 None
        encoder_attention_mask: Optional[torch.LongTensor] = None,
```