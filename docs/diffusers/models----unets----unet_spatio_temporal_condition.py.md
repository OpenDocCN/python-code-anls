# `.\diffusers\models\unets\unet_spatio_temporal_condition.py`

```py
# 从数据类模块导入数据类装饰器
from dataclasses import dataclass
# 导入字典、可选、元组和联合类型的类型注解
from typing import Dict, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
import torch.nn as nn

# 导入配置相关的工具
from ...configuration_utils import ConfigMixin, register_to_config
# 导入用于加载 UNet2D 条件模型的混合类
from ...loaders import UNet2DConditionLoadersMixin
# 导入基本输出类和日志工具
from ...utils import BaseOutput, logging
# 导入注意力处理器相关内容
from ..attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
# 导入时间步嵌入和时间步类
from ..embeddings import TimestepEmbedding, Timesteps
# 导入模型混合类
from ..modeling_utils import ModelMixin
# 导入 UNet 3D 块的相关功能
from .unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个数据类，用于存储 UNet 空间时间条件模型的输出
@dataclass
class UNetSpatioTemporalConditionOutput(BaseOutput):
    """
    [`UNetSpatioTemporalConditionModel`] 的输出。

    参数：
        sample (`torch.Tensor` 形状为 `(batch_size, num_frames, num_channels, height, width)`):
            根据 `encoder_hidden_states` 输入条件的隐藏状态输出。模型最后一层的输出。
    """

    # 定义一个可选的张量，默认为 None
    sample: torch.Tensor = None

# 定义一个条件空间时间 UNet 模型类
class UNetSpatioTemporalConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    一个条件空间时间 UNet 模型，接受噪声视频帧、条件状态和时间步，返回指定形状的样本输出。

    此模型继承自 [`ModelMixin`]。请查看超类文档以了解为所有模型实现的通用方法
    （例如下载或保存）。
    # 函数参数说明
    Parameters:
        # 输入/输出样本的高度和宽度，类型为整型或整型元组，默认为 None
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        # 输入样本的通道数，默认为 8
        in_channels (`int`, *optional*, defaults to 8): Number of channels in the input sample.
        # 输出样本的通道数，默认为 4
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        # 用于下采样的块的元组，默认为指定的四个下采样块
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal")`):
            The tuple of downsample blocks to use.
        # 用于上采样的块的元组，默认为指定的四个上采样块
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal")`):
            The tuple of upsample blocks to use.
        # 每个块的输出通道数的元组，默认为 (320, 640, 1280, 1280)
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        # 用于编码附加时间 ID 的维度，默认为 256
        addition_time_embed_dim: (`int`, defaults to 256):
            Dimension to encode the additional time ids.
        # 编码 `added_time_ids` 的投影维度，默认为 768
        projection_class_embeddings_input_dim (`int`, defaults to 768):
            The dimension of the projection of encoded `added_time_ids`.
        # 每个块的层数，默认为 2
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        # 交叉注意力特征的维度，类型为整型或整型元组，默认为 1280
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        # 变换器块的数量，相关于特定类型的下/上块，默认为 1
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unets.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal`],
            [`~models.unets.unet_3d_blocks.CrossAttnUpBlockSpatioTemporal`],
            [`~models.unets.unet_3d_blocks.UNetMidBlockSpatioTemporal`].
        # 注意力头的数量，默认为 (5, 10, 10, 20)
        num_attention_heads (`int`, `Tuple[int]`, defaults to `(5, 10, 10, 20)`):
            The number of attention heads.
        # 使用的 dropout 概率，默认为 0.0
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    # 启用梯度检查点
    _supports_gradient_checkpointing = True

    # 注册到配置中
    @register_to_config
    # 初始化方法，用于创建类的实例并设置其属性
        def __init__(
            self,
            # 可选的样本大小参数，默认为 None
            sample_size: Optional[int] = None,
            # 输入通道数量，默认为 8
            in_channels: int = 8,
            # 输出通道数量，默认为 4
            out_channels: int = 4,
            # 各下采样块的类型，默认为指定的类型元组
            down_block_types: Tuple[str] = (
                "CrossAttnDownBlockSpatioTemporal",
                "CrossAttnDownBlockSpatioTemporal",
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            # 各上采样块的类型，默认为指定的类型元组
            up_block_types: Tuple[str] = (
                "UpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
            ),
            # 各块输出通道数量，默认为指定的整数元组
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            # 附加时间嵌入维度，默认为 256
            addition_time_embed_dim: int = 256,
            # 投影类嵌入输入维度，默认为 768
            projection_class_embeddings_input_dim: int = 768,
            # 每个块的层数，可以是整数或整数元组，默认为 2
            layers_per_block: Union[int, Tuple[int]] = 2,
            # 交叉注意力维度，可以是整数或整数元组，默认为 1024
            cross_attention_dim: Union[int, Tuple[int]] = 1024,
            # 每个块的变换器层数，可以是整数或元组，默认为 1
            transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
            # 注意力头的数量，可以是整数或整数元组，默认为 (5, 10, 20, 20)
            num_attention_heads: Union[int, Tuple[int]] = (5, 10, 20, 20),
            # 帧的数量，默认为 25
            num_frames: int = 25,
        # 定义属性装饰器，返回注意力处理器字典
        @property
        def attn_processors(self) -> Dict[str, AttentionProcessor]:
            r"""
            Returns:
                `dict` of attention processors: A dictionary containing all attention processors used in the model with
                indexed by its weight name.
            """
            # 创建一个空字典用于存储注意力处理器
            processors = {}
    
            # 定义递归函数以添加处理器
            def fn_recursive_add_processors(
                # 当前模块名称
                name: str,
                # 当前模块对象
                module: torch.nn.Module,
                # 存储处理器的字典
                processors: Dict[str, AttentionProcessor],
            ):
                # 检查当前模块是否有获取处理器的方法
                if hasattr(module, "get_processor"):
                    # 将处理器添加到字典中，键为模块名称
                    processors[f"{name}.processor"] = module.get_processor()
    
                # 遍历当前模块的所有子模块
                for sub_name, child in module.named_children():
                    # 递归调用，将子模块的处理器添加到字典中
                    fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
                # 返回处理器字典
                return processors
    
            # 遍历当前实例的所有子模块
            for name, module in self.named_children():
                # 调用递归函数以添加所有处理器
                fn_recursive_add_processors(name, module, processors)
    
            # 返回所有注意力处理器的字典
            return processors
    # 设置用于计算注意力的处理器
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r""" 
        设置要用于计算注意力的处理器。
    
        参数：
            processor（`dict` 或 `AttentionProcessor`）：
                实例化的处理器类或将被设置为 **所有** `Attention` 层的处理器类字典。
    
                如果 `processor` 是字典，则键需要定义相应的交叉注意力处理器的路径。 
                强烈建议在设置可训练的注意力处理器时使用此方法。
    
        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())
    
        # 检查传入的处理器字典长度是否与注意力层数量匹配
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器字典，但处理器数量 {len(processor)} 与注意力层数量 {count} 不匹配。"
                f" 请确保传入 {count} 个处理器类。"
            )
    
        # 定义递归设置注意力处理器的函数
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块具有设置处理器的属性
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，直接设置
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中取出相应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))
    
            # 遍历模块的子模块
            for sub_name, child in module.named_children():
                # 递归调用设置处理器的函数
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
        # 遍历当前对象的子模块
        for name, module in self.named_children():
            # 递归设置注意力处理器
            fn_recursive_attn_processor(name, module, processor)
    
    # 设置默认的注意力处理器
    def set_default_attn_processor(self):
        """
        禁用自定义注意力处理器，并设置默认的注意力实现。
        """
        # 检查当前的处理器是否都在交叉注意力处理器列表中
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 创建默认的注意力处理器实例
            processor = AttnProcessor()
        else:
            # 抛出错误，不能设置默认处理器
            raise ValueError(
                f"当注意力处理器类型为 {next(iter(self.attn_processors.values()))} 时，无法调用 `set_default_attn_processor`"
            )
    
        # 设置默认的注意力处理器
        self.set_attn_processor(processor)
    
    # 设置模块的梯度检查点
    def _set_gradient_checkpointing(self, module, value=False):
        # 如果模块具有梯度检查点属性，则设置其值
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
    
    # 从 diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking 复制的内容
    # 定义一个方法以启用前馈层的分块处理，参数为分块大小和维度
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        设置注意力处理器以使用 [前馈分块处理](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers)。
    
        参数:
            chunk_size (`int`, *可选*):
                前馈层的分块大小。如果未指定，将单独在每个维度为 `dim` 的张量上运行前馈层。
            dim (`int`, *可选*, 默认值为 `0`):
                前馈计算应该在哪个维度上进行分块。选择 dim=0（批量）或 dim=1（序列长度）。
        """
        # 如果 dim 不是 0 或 1，则引发值错误
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")
    
        # 默认的分块大小为 1
        chunk_size = chunk_size or 1
    
        # 定义一个递归函数以设置每个子模块的前馈分块处理
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            # 如果模块有设置前馈分块的方法，则调用该方法
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
    
            # 遍历模块的所有子模块并递归调用
            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)
    
        # 遍历当前对象的所有子模块并调用递归函数
        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)
    
    # 定义前馈方法，接收样本、时间步、编码器隐藏状态、额外时间ID和返回字典参数
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
```