# `.\diffusers\models\controlnet_sparsectrl.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（"许可证"）进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件
# 根据许可证分发是按“原样”基础进行的，
# 不提供任何种类的保证或条件，无论是明示或暗示的。
# 请参阅许可证以获取与权限和
# 限制相关的特定语言。

from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示所需的类型

import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块
from torch.nn import functional as F  # 导入 PyTorch 神经网络功能模块，通常用于定义激活函数等

from ..configuration_utils import ConfigMixin, register_to_config  # 从父级模块导入配置混合类和注册函数
from ..loaders import FromOriginalModelMixin  # 从父级模块导入原始模型混合类
from ..utils import BaseOutput, logging  # 从父级模块导入基础输出类和日志记录工具
from .attention_processor import (  # 从当前模块导入注意力处理相关类
    ADDED_KV_ATTENTION_PROCESSORS,  # 导入新增键值注意力处理器
    CROSS_ATTENTION_PROCESSORS,  # 导入交叉注意力处理器
    AttentionProcessor,  # 导入注意力处理器基类
    AttnAddedKVProcessor,  # 导入新增键值的注意力处理器
    AttnProcessor,  # 导入普通注意力处理器
)
from .embeddings import TimestepEmbedding, Timesteps  # 从当前模块导入时间步嵌入和时间步类
from .modeling_utils import ModelMixin  # 从当前模块导入模型混合类
from .unets.unet_2d_blocks import UNetMidBlock2DCrossAttn  # 从 2D UNet 模块导入中间块交叉注意力类
from .unets.unet_2d_condition import UNet2DConditionModel  # 从 2D UNet 模块导入条件模型
from .unets.unet_motion_model import CrossAttnDownBlockMotion, DownBlockMotion  # 从运动模型模块导入相关类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，便于后续日志记录使用

@dataclass  # 将类标记为数据类，自动生成初始化等方法
class SparseControlNetOutput(BaseOutput):  # 定义 SparseControlNetOutput 类，继承自 BaseOutput
    """
    [`SparseControlNetModel`] 的输出。

    参数:
        down_block_res_samples (`tuple[torch.Tensor]`):
            一个包含每个下采样块在不同分辨率下激活的元组。每个张量的形状应为
            `(batch_size, channel * resolution, height // resolution, width // resolution)`。输出可用于条件
            原始 UNet 的下采样激活。
        mid_down_block_re_sample (`torch.Tensor`):
            中间块（最低采样分辨率）的激活。每个张量的形状应为
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`。
            输出可用于条件原始 UNet 的中间块激活。
    """

    down_block_res_samples: Tuple[torch.Tensor]  # 定义下采样块结果样本的属性
    mid_block_res_sample: torch.Tensor  # 定义中间块结果样本的属性


class SparseControlNetConditioningEmbedding(nn.Module):  # 定义 SparseControlNetConditioningEmbedding 类，继承自 nn.Module
    def __init__(  # 初始化方法
        self,
        conditioning_embedding_channels: int,  # 条件嵌入通道数
        conditioning_channels: int = 3,  # 条件通道数，默认为 3
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),  # 块输出通道数的元组，包含多个值
    ):
        # 初始化父类，调用父类的构造函数
        super().__init__()

        # 定义输入卷积层，接受条件通道数并输出块的第一个通道数，卷积核大小为3，填充为1
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        # 创建一个空的模块列表，用于存储后续的卷积块
        self.blocks = nn.ModuleList([])

        # 遍历块输出通道数列表，构建卷积块
        for i in range(len(block_out_channels) - 1):
            # 当前通道数
            channel_in = block_out_channels[i]
            # 下一层的通道数
            channel_out = block_out_channels[i + 1]
            # 添加一个卷积层，输入通道数和输出通道数均为当前通道数
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            # 添加一个卷积层，输入通道数为当前通道数，输出通道数为下一层通道数，步幅为2
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        # 定义输出卷积层，接受最后一个块的输出通道数并输出条件嵌入通道数，卷积核大小为3，填充为1
        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    # 前向传播函数，接受一个张量作为输入，返回一个张量作为输出
    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        # 通过输入卷积层处理条件张量，得到嵌入
        embedding = self.conv_in(conditioning)
        # 应用激活函数 SiLU
        embedding = F.silu(embedding)

        # 遍历每个卷积块，依次处理嵌入
        for block in self.blocks:
            # 通过当前卷积块处理嵌入
            embedding = block(embedding)
            # 再次应用激活函数 SiLU
            embedding = F.silu(embedding)

        # 通过输出卷积层处理嵌入，得到最终输出
        embedding = self.conv_out(embedding)
        # 返回最终输出
        return embedding
# 定义一个稀疏控制网络模型类，继承自多个混合类以获得其功能
class SparseControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    根据 [SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion
    Models](https://arxiv.org/abs/2311.16933) 的描述，定义一个稀疏控制网络模型。
    """

    # 支持梯度检查点，允许在训练时节省内存
    _supports_gradient_checkpointing = True

    # 将初始化方法注册到配置中
    @register_to_config
    def __init__(
        # 输入通道数，默认为4
        in_channels: int = 4,
        # 条件通道数，默认为4
        conditioning_channels: int = 4,
        # 是否将正弦函数翻转为余弦函数，默认为True
        flip_sin_to_cos: bool = True,
        # 频率偏移量，默认为0
        freq_shift: int = 0,
        # 下采样块的类型，默认为三个交叉注意力块和一个下块
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "DownBlockMotion",
        ),
        # 是否仅使用交叉注意力，默认为False
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        # 每个块的输出通道数，默认为指定的四个值
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        # 每个块的层数，默认为2
        layers_per_block: int = 2,
        # 下采样时的填充大小，默认为1
        downsample_padding: int = 1,
        # 中间块的缩放因子，默认为1
        mid_block_scale_factor: float = 1,
        # 激活函数类型，默认为"silu"
        act_fn: str = "silu",
        # 归一化的组数，默认为32
        norm_num_groups: Optional[int] = 32,
        # 归一化的epsilon值，默认为1e-5
        norm_eps: float = 1e-5,
        # 交叉注意力的维度，默认为768
        cross_attention_dim: int = 768,
        # 每个块的变换器层数，默认为1
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        # 每个中间块的变换器层数，默认为None
        transformer_layers_per_mid_block: Optional[Union[int, Tuple[int]]] = None,
        # 每个块的时间变换器层数，默认为1
        temporal_transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        # 注意力头的维度，默认为8
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        # 注意力头的数量，默认为None
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        # 是否使用线性投影，默认为False
        use_linear_projection: bool = False,
        # 是否提升注意力计算精度，默认为False
        upcast_attention: bool = False,
        # ResNet时间尺度偏移，默认为"default"
        resnet_time_scale_shift: str = "default",
        # 条件嵌入的输出通道数，默认为指定的四个值
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        # 是否全局池条件，默认为False
        global_pool_conditions: bool = False,
        # 控制网络条件通道的顺序，默认为"rgb"
        controlnet_conditioning_channel_order: str = "rgb",
        # 最大的运动序列长度，默认为32
        motion_max_seq_length: int = 32,
        # 运动部分的注意力头数量，默认为8
        motion_num_attention_heads: int = 8,
        # 是否拼接条件掩码，默认为True
        concat_conditioning_mask: bool = True,
        # 是否使用简化的条件嵌入，默认为True
        use_simplified_condition_embedding: bool = True,
    # 定义一个类方法，用于从UNet模型创建稀疏控制网络模型
    @classmethod
    def from_unet(
        cls,
        # 输入的UNet模型
        unet: UNet2DConditionModel,
        # 控制网络条件通道的顺序，默认为"rgb"
        controlnet_conditioning_channel_order: str = "rgb",
        # 条件嵌入的输出通道数，默认为指定的四个值
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        # 是否从UNet加载权重，默认为True
        load_weights_from_unet: bool = True,
        # 条件通道数，默认为3
        conditioning_channels: int = 3,
    # 实例化一个 [`SparseControlNetModel`]，来源于 [`UNet2DConditionModel`]。
    ) -> "SparseControlNetModel":
        r"""
        实例化一个 [`SparseControlNetModel`]，源自 [`UNet2DConditionModel`]。
        
        参数：
            unet (`UNet2DConditionModel`):
                需要复制到 [`SparseControlNetModel`] 的 UNet 模型权重。所有适用的配置选项也会被复制。
        """
        # 获取 UNet 配置中的 transformer_layers_per_block，默认值为 1
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        # 获取 UNet 配置中的 down_block_types
        down_block_types = unet.config.down_block_types
    
        # 遍历每种下采样块类型
        for i in range(len(down_block_types)):
            # 检查下采样块类型是否包含 "CrossAttn"
            if "CrossAttn" in down_block_types[i]:
                # 替换为 "CrossAttnDownBlockMotion"
                down_block_types[i] = "CrossAttnDownBlockMotion"
            # 检查下采样块类型是否包含 "Down"
            elif "Down" in down_block_types[i]:
                # 替换为 "DownBlockMotion"
                down_block_types[i] = "DownBlockMotion"
            # 如果类型无效，抛出异常
            else:
                raise ValueError("Invalid `block_type` encountered. Must be a cross-attention or down block")
    
        # 创建 SparseControlNetModel 实例
        controlnet = cls(
            in_channels=unet.config.in_channels,  # 输入通道数
            conditioning_channels=conditioning_channels,  # 条件通道数
            flip_sin_to_cos=unet.config.flip_sin_to_cos,  # 是否翻转正弦到余弦
            freq_shift=unet.config.freq_shift,  # 频率偏移
            down_block_types=unet.config.down_block_types,  # 下采样块类型
            only_cross_attention=unet.config.only_cross_attention,  # 仅使用交叉注意力
            block_out_channels=unet.config.block_out_channels,  # 块输出通道数
            layers_per_block=unet.config.layers_per_block,  # 每个块的层数
            downsample_padding=unet.config.downsample_padding,  # 下采样填充
            mid_block_scale_factor=unet.config.mid_block_scale_factor,  # 中间块缩放因子
            act_fn=unet.config.act_fn,  # 激活函数
            norm_num_groups=unet.config.norm_num_groups,  # 归一化组数
            norm_eps=unet.config.norm_eps,  # 归一化的 epsilon
            cross_attention_dim=unet.config.cross_attention_dim,  # 交叉注意力维度
            transformer_layers_per_block=transformer_layers_per_block,  # 每个块的 transformer 层数
            attention_head_dim=unet.config.attention_head_dim,  # 注意力头维度
            num_attention_heads=unet.config.num_attention_heads,  # 注意力头数量
            use_linear_projection=unet.config.use_linear_projection,  # 是否使用线性投影
            upcast_attention=unet.config.upcast_attention,  # 是否上升注意力
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,  # ResNet 时间缩放偏移
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,  # 条件嵌入输出通道
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,  # 控制网条件通道顺序
        )
    
        # 如果需要从 UNet 加载权重
        if load_weights_from_unet:
            # 加载输入卷积层的权重
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict(), strict=False)
            # 加载时间投影层的权重
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict(), strict=False)
            # 加载时间嵌入层的权重
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict(), strict=False)
            # 加载下采样块的权重
            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict(), strict=False)
            # 加载中间块的权重
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict(), strict=False)
    
        # 返回控制网模型实例
        return controlnet
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors 复制的
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回：
            `dict` 的注意力处理器：一个字典，包含模型中使用的所有注意力处理器，
            按其权重名称索引。
        """
        # 初始化一个空字典，用于存储处理器
        processors = {}
    
        # 定义一个递归函数，用于添加处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 检查模块是否具有获取处理器的函数
            if hasattr(module, "get_processor"):
                # 将处理器添加到字典中，使用名称作为键
                processors[f"{name}.processor"] = module.get_processor()
    
            # 遍历模块的子模块
            for sub_name, child in module.named_children():
                # 递归调用该函数以处理子模块
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
            return processors
    
        # 遍历当前对象的子模块
        for name, module in self.named_children():
            # 调用递归函数以添加所有处理器
            fn_recursive_add_processors(name, module, processors)
    
        # 返回所有处理器的字典
        return processors
    
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor 复制的
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。
    
        参数：
            processor (`dict` of `AttentionProcessor` 或仅 `AttentionProcessor`):
                实例化的处理器类或处理器类的字典，将作为 **所有** `Attention` 层的处理器设置。
    
                如果 `processor` 是字典，键需要定义相应的交叉注意力处理器的路径。
                在设置可训练的注意力处理器时，强烈推荐这样做。
    
        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())
    
        # 如果传入的是字典，检查其长度是否与注意力层数量匹配
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传递了一个处理器字典，但处理器数量 {len(processor)} 与"
                f" 注意力层数量 {count} 不匹配。请确保传递 {count} 个处理器类。"
            )
    
        # 定义一个递归函数，用于设置处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 检查模块是否具有设置处理器的函数
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，直接设置
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中弹出对应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))
    
            # 遍历模块的子模块
            for sub_name, child in module.named_children():
                # 递归调用该函数以处理子模块
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
        # 遍历当前对象的子模块
        for name, module in self.named_children():
            # 调用递归函数以设置所有处理器
            fn_recursive_attn_processor(name, module, processor)
    
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor 复制的
    # 设置默认的注意力处理器
    def set_default_attn_processor(self):
        # 禁用自定义注意力处理器，并设置默认的注意力实现
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        # 检查所有注意力处理器是否属于添加的键值注意力处理器
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 创建添加的键值注意力处理器实例
            processor = AttnAddedKVProcessor()
        # 检查所有注意力处理器是否属于交叉注意力处理器
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 创建标准注意力处理器实例
            processor = AttnProcessor()
        else:
            # 如果注意力处理器类型不符合要求，则抛出错误
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )
    
        # 设置所选的注意力处理器
        self.set_attn_processor(processor)
    
    # 从 diffusers.models.unets.unet_2d_condition 中复制的设置梯度检查点的方法
    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        # 检查模块是否属于指定的类型
        if isinstance(module, (CrossAttnDownBlockMotion, DownBlockMotion, UNetMidBlock2DCrossAttn)):
            # 设置模块的梯度检查点属性
            module.gradient_checkpointing = value
    
    # 前向传播方法
    def forward(
        self,
        # 输入的样本张量
        sample: torch.Tensor,
        # 时间步长，可以是张量、浮点数或整数
        timestep: Union[torch.Tensor, float, int],
        # 编码器的隐藏状态张量
        encoder_hidden_states: torch.Tensor,
        # 控制网络条件张量
        controlnet_cond: torch.Tensor,
        # 条件缩放因子，默认为 1.0
        conditioning_scale: float = 1.0,
        # 可选的时间步条件张量
        timestep_cond: Optional[torch.Tensor] = None,
        # 可选的注意力掩码张量
        attention_mask: Optional[torch.Tensor] = None,
        # 可选的交叉注意力参数字典
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 可选的条件掩码张量
        conditioning_mask: Optional[torch.Tensor] = None,
        # 猜测模式，默认为 False
        guess_mode: bool = False,
        # 返回字典，默认为 True
        return_dict: bool = True,
# 从 diffusers.models.controlnet.zero_module 复制而来
def zero_module(module: nn.Module) -> nn.Module:
    # 遍历传入模块的所有参数
    for p in module.parameters():
        # 将每个参数初始化为零
        nn.init.zeros_(p)
    # 返回已初始化的模块
    return module
```