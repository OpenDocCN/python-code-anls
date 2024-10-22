# `.\diffusers\models\unets\unet_i2vgen_xl.py`

```
# 版权声明，表明版权归2024年阿里巴巴DAMO-VILAB和HuggingFace团队所有
# 提供Apache许可证2.0版本的使用条款
# 说明只能在遵循许可证的情况下使用此文件
# 可在指定网址获取许可证副本
#
# 除非适用法律或书面协议另有约定，否则软件按“原样”分发
# 不提供任何形式的担保或条件
# 请参见许可证以获取与权限和限制相关的具体信息

from typing import Any, Dict, Optional, Tuple, Union  # 导入类型提示工具，用于类型注解

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.utils.checkpoint  # 导入PyTorch的检查点工具

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入类和函数
from ...loaders import UNet2DConditionLoadersMixin  # 导入2D条件加载器混合类
from ...utils import logging  # 导入日志工具
from ..activations import get_activation  # 导入激活函数获取工具
from ..attention import Attention, FeedForward  # 导入注意力机制和前馈网络
from ..attention_processor import (  # 从注意力处理器模块导入多个处理器
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from ..embeddings import TimestepEmbedding, Timesteps  # 导入时间步嵌入和时间步类
from ..modeling_utils import ModelMixin  # 导入模型混合类
from ..transformers.transformer_temporal import TransformerTemporalModel  # 导入时间变换器模型
from .unet_3d_blocks import (  # 从3D U-Net块模块导入多个类
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
from .unet_3d_condition import UNet3DConditionOutput  # 导入3D条件输出类

logger = logging.get_logger(__name__)  # 创建日志记录器，用于记录当前模块的信息

class I2VGenXLTransformerTemporalEncoder(nn.Module):  # 定义一个名为I2VGenXLTransformerTemporalEncoder的类，继承自nn.Module
    def __init__(  # 构造函数，用于初始化类的实例
        self,
        dim: int,  # 输入的特征维度
        num_attention_heads: int,  # 注意力头的数量
        attention_head_dim: int,  # 每个注意力头的维度
        activation_fn: str = "geglu",  # 激活函数类型，默认使用geglu
        upcast_attention: bool = False,  # 是否提升注意力计算的精度
        ff_inner_dim: Optional[int] = None,  # 前馈网络的内部维度，默认为None
        dropout: int = 0.0,  # dropout概率，默认为0.0
    ):
        super().__init__()  # 调用父类构造函数
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-5)  # 初始化层归一化层
        self.attn1 = Attention(  # 初始化注意力层
            query_dim=dim,  # 查询维度
            heads=num_attention_heads,  # 注意力头数量
            dim_head=attention_head_dim,  # 每个头的维度
            dropout=dropout,  # dropout概率
            bias=False,  # 不使用偏置
            upcast_attention=upcast_attention,  # 是否提升注意力计算精度
            out_bias=True,  # 输出使用偏置
        )
        self.ff = FeedForward(  # 初始化前馈网络
            dim,  # 输入维度
            dropout=dropout,  # dropout概率
            activation_fn=activation_fn,  # 激活函数类型
            final_dropout=False,  # 最后层不使用dropout
            inner_dim=ff_inner_dim,  # 内部维度
            bias=True,  # 使用偏置
        )

    def forward(  # 定义前向传播方法
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态
    # 该方法返回处理后的隐藏状态张量
    ) -> torch.Tensor:
        # 对隐藏状态进行归一化处理
        norm_hidden_states = self.norm1(hidden_states)
        # 计算注意力输出，使用归一化后的隐藏状态
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        # 将注意力输出与原始隐藏状态相加，更新隐藏状态
        hidden_states = attn_output + hidden_states
        # 如果隐藏状态是四维，则去掉第一维
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
    
        # 通过前馈网络处理隐藏状态
        ff_output = self.ff(hidden_states)
        # 将前馈输出与当前隐藏状态相加，更新隐藏状态
        hidden_states = ff_output + hidden_states
        # 如果隐藏状态是四维，则去掉第一维
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
    
        # 返回最终的隐藏状态
        return hidden_states
# 定义 I2VGenXL UNet 类，继承多个混入类以增加功能
class I2VGenXLUNet(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    I2VGenXL UNet。一个条件3D UNet模型，接收噪声样本、条件状态和时间步，
    返回与样本形状相同的输出。

    该模型继承自 [`ModelMixin`]。有关所有模型实现的通用方法（如下载或保存），
    请查看超类文档。

    参数：
        sample_size (`int` 或 `Tuple[int, int]`, *可选*, 默认值为 `None`):
            输入/输出样本的高度和宽度。
        in_channels (`int`, *可选*, 默认值为 4): 输入样本的通道数。
        out_channels (`int`, *可选*, 默认值为 4): 输出样本的通道数。
        down_block_types (`Tuple[str]`, *可选*, 默认值为 `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            使用的下采样块的元组。
        up_block_types (`Tuple[str]`, *可选*, 默认值为 `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")`):
            使用的上采样块的元组。
        block_out_channels (`Tuple[int]`, *可选*, 默认值为 `(320, 640, 1280, 1280)`):
            每个块的输出通道元组。
        layers_per_block (`int`, *可选*, 默认值为 2): 每个块的层数。
        norm_num_groups (`int`, *可选*, 默认值为 32): 用于归一化的组数。
            如果为 `None`，则跳过后处理中的归一化和激活层。
        cross_attention_dim (`int`, *可选*, 默认值为 1280): 跨注意力特征的维度。
        attention_head_dim (`int`, *可选*, 默认值为 64): 注意力头的维度。
        num_attention_heads (`int`, *可选*): 注意力头的数量。
    """

    # 设置不支持梯度检查点的属性为 False
    _supports_gradient_checkpointing = False

    @register_to_config
    # 初始化方法，接受多种可选参数以设置模型配置
    def __init__(
        self,
        sample_size: Optional[int] = None,  # 输入/输出样本大小，默认为 None
        in_channels: int = 4,  # 输入样本的通道数，默认为 4
        out_channels: int = 4,  # 输出样本的通道数，默认为 4
        down_block_types: Tuple[str, ...] = (  # 下采样块的类型，默认为指定的元组
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (  # 上采样块的类型，默认为指定的元组
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),  # 每个块的输出通道，默认为指定的元组
        layers_per_block: int = 2,  # 每个块的层数，默认为 2
        norm_num_groups: Optional[int] = 32,  # 归一化组数，默认为 32
        cross_attention_dim: int = 1024,  # 跨注意力特征的维度，默认为 1024
        attention_head_dim: Union[int, Tuple[int]] = 64,  # 注意力头的维度，默认为 64
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,  # 注意力头的数量，默认为 None
    @property
    # 该属性从 UNet2DConditionModel 的 attn_processors 复制
    # 定义返回注意力处理器的函数，返回类型为字典，键为字符串，值为 AttentionProcessor 对象
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # 创建一个空字典，用于存储处理器
        processors = {}

        # 定义一个递归函数，用于添加处理器到字典
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 检查模块是否有 get_processor 方法
            if hasattr(module, "get_processor"):
                # 将处理器添加到字典中，键为名称加上 ".processor"
                processors[f"{name}.processor"] = module.get_processor()

            # 遍历模块的子模块
            for sub_name, child in module.named_children():
                # 递归调用，处理子模块
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            # 返回更新后的处理器字典
            return processors

        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数，将处理器添加到字典中
            fn_recursive_add_processors(name, module, processors)

        # 返回包含所有处理器的字典
        return processors

    # 从 diffusers.models.unets.unet_2d_condition 中复制的设置注意力处理器的函数
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
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())

        # 如果传入的是字典且数量不匹配，则引发错误
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        # 定义一个递归函数，用于设置处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 检查模块是否有 set_processor 方法
            if hasattr(module, "set_processor"):
                # 如果传入的处理器不是字典，直接设置
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中移除并设置对应的处理器
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 遍历模块的子模块
            for sub_name, child in module.named_children():
                # 递归调用，设置子模块的处理器
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数，为每个模块设置处理器
            fn_recursive_attn_processor(name, module, processor)

    # 从 diffusers.models.unets.unet_3d_condition 中复制的启用前向分块的函数
    # 启用前馈层的分块处理
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        设置注意力处理器使用[前馈分块](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers)。

        参数:
            chunk_size (`int`, *可选*):
                前馈层的块大小。如果未指定，将对维度为`dim`的每个张量单独运行前馈层。
            dim (`int`, *可选*, 默认为`0`):
                前馈计算应分块的维度。可以选择dim=0（批次）或dim=1（序列长度）。
        """
        # 检查维度是否在有效范围内
        if dim not in [0, 1]:
            # 抛出错误，确保dim只为0或1
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # 默认块大小为1
        chunk_size = chunk_size or 1

        # 定义递归函数，用于设置每个模块的前馈分块
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            # 如果模块有设置分块前馈的方法，调用该方法
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            # 递归遍历子模块
            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        # 对当前对象的所有子模块应用前馈分块设置
        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # 从diffusers.models.unets.unet_3d_condition.UNet3DConditionModel复制的禁用前馈分块的方法
    def disable_forward_chunking(self):
        # 定义递归函数，用于禁用模块的前馈分块
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            # 如果模块有设置分块前馈的方法，调用该方法
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            # 递归遍历子模块
            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        # 对当前对象的所有子模块应用禁用前馈分块设置
        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    # 从diffusers.models.unets.unet_2d_condition.UNet2DConditionModel复制的设置默认注意力处理器的方法
    def set_default_attn_processor(self):
        """
        禁用自定义注意力处理器并设置默认的注意力实现。
        """
        # 检查所有注意力处理器是否属于已添加的KV注意力处理器类
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 如果是，则设置为已添加KV处理器
            processor = AttnAddedKVProcessor()
        # 检查所有注意力处理器是否属于交叉注意力处理器类
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 如果是，则设置为标准注意力处理器
            processor = AttnProcessor()
        else:
            # 抛出错误，说明当前的注意力处理器类型不被支持
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        # 设置当前对象的注意力处理器为选择的处理器
        self.set_attn_processor(processor)

    # 从diffusers.models.unets.unet_3d_condition.UNet3DConditionModel复制的设置梯度检查点的方法
    # 设置梯度检查点，指定模块和布尔值
    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        # 检查模块是否为指定的类型之一
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            # 设置模块的梯度检查点属性为指定值
            module.gradient_checkpointing = value

    # 从 UNet2DConditionModel 中复制的启用 FreeU 方法
    def enable_freeu(self, s1, s2, b1, b2):
        r"""启用 FreeU 机制，详情见 https://arxiv.org/abs/2309.11497.

        后缀表示缩放因子应用的阶段块。

        请参考 [官方库](https://github.com/ChenyangSi/FreeU) 以获取适用于不同管道（如 Stable Diffusion v1, v2 和 Stable Diffusion XL）的有效值组合。

        参数：
            s1 (`float`):
                阶段 1 的缩放因子，用于减弱跳过特征的贡献，以缓解增强去噪过程中的“过平滑效应”。
            s2 (`float`):
                阶段 2 的缩放因子，用于减弱跳过特征的贡献，以缓解增强去噪过程中的“过平滑效应”。
            b1 (`float`): 阶段 1 的缩放因子，用于放大主干特征的贡献。
            b2 (`float`): 阶段 2 的缩放因子，用于放大主干特征的贡献。
        """
        # 遍历上采样块，索引 i 和块对象 upsample_block
        for i, upsample_block in enumerate(self.up_blocks):
            # 设置上采样块的属性 s1 为给定值 s1
            setattr(upsample_block, "s1", s1)
            # 设置上采样块的属性 s2 为给定值 s2
            setattr(upsample_block, "s2", s2)
            # 设置上采样块的属性 b1 为给定值 b1
            setattr(upsample_block, "b1", b1)
            # 设置上采样块的属性 b2 为给定值 b2
            setattr(upsample_block, "b2", b2)

    # 从 UNet2DConditionModel 中复制的禁用 FreeU 方法
    def disable_freeu(self):
        """禁用 FreeU 机制。"""
        # 定义 FreeU 相关的属性键
        freeu_keys = {"s1", "s2", "b1", "b2"}
        # 遍历上采样块，索引 i 和块对象 upsample_block
        for i, upsample_block in enumerate(self.up_blocks):
            # 遍历 FreeU 属性键
            for k in freeu_keys:
                # 如果上采样块具有该属性或属性值不为 None
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    # 将上采样块的该属性设置为 None
                    setattr(upsample_block, k, None)

    # 从 UNet2DConditionModel 中复制的融合 QKV 投影方法
    # 定义一个方法，用于启用融合的 QKV 投影
    def fuse_qkv_projections(self):
        # 提供方法的文档字符串，描述其功能和警告信息
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.
    
        <Tip warning={true}>
    
        This API is 🧪 experimental.
    
        </Tip>
        """
        # 初始化原始注意力处理器为 None
        self.original_attn_processors = None
    
        # 遍历当前对象的注意力处理器
        for _, attn_processor in self.attn_processors.items():
            # 检查处理器类名中是否包含 "Added"
            if "Added" in str(attn_processor.__class__.__name__):
                # 如果包含，抛出异常提示不支持
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")
    
        # 保存当前的注意力处理器以备后用
        self.original_attn_processors = self.attn_processors
    
        # 遍历当前对象的所有模块
        for module in self.modules():
            # 检查模块是否为 Attention 类型
            if isinstance(module, Attention):
                # 调用模块的方法，启用融合投影
                module.fuse_projections(fuse=True)
    
        # 设置注意力处理器为 FusedAttnProcessor2_0 的实例
        self.set_attn_processor(FusedAttnProcessor2_0())
    
    # 从 UNet2DConditionModel 复制的方法，用于禁用融合的 QKV 投影
    def unfuse_qkv_projections(self):
        # 提供方法的文档字符串，描述其功能和警告信息
        """Disables the fused QKV projection if enabled.
    
        <Tip warning={true}>
    
        This API is 🧪 experimental.
    
        </Tip>
    
        """
        # 检查原始注意力处理器是否不为 None
        if self.original_attn_processors is not None:
            # 如果不为 None，恢复原始的注意力处理器
            self.set_attn_processor(self.original_attn_processors)
    
    # 定义前向传播方法，接受多个输入参数
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        fps: torch.Tensor,
        image_latents: torch.Tensor,
        image_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
```