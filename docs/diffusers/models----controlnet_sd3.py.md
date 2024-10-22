# `.\diffusers\models\controlnet_sd3.py`

```
# 版权所有 2024 Stability AI, HuggingFace 团队和 InstantX 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下位置获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是以“原样”基础分发的，
# 不提供任何形式的保证或条件，无论是明示或暗示的。
# 有关许可证的特定权限和限制，请参见许可证。

# 从 dataclasses 模块导入 dataclass 装饰器，用于简化类的定义
from dataclasses import dataclass
# 从 typing 模块导入类型提示的相关类型
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库及其神经网络模块
import torch
import torch.nn as nn

# 导入配置和注册功能相关的模块
from ..configuration_utils import ConfigMixin, register_to_config
# 导入模型加载的混合接口
from ..loaders import FromOriginalModelMixin, PeftAdapterMixin
# 导入联合变换器块的定义
from ..models.attention import JointTransformerBlock
# 导入注意力处理相关的模块
from ..models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0
# 导入变换器 2D 模型输出的定义
from ..models.modeling_outputs import Transformer2DModelOutput
# 导入模型的通用功能混合接口
from ..models.modeling_utils import ModelMixin
# 导入工具函数和常量
from ..utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
# 导入控制网络相关的基础输出和零模块
from .controlnet import BaseOutput, zero_module
# 导入组合时间步文本投影嵌入和补丁嵌入的定义
from .embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed

# 创建日志记录器实例，用于记录信息和调试
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义数据类 SD3ControlNetOutput，用于存储控制网络块的样本输出
@dataclass
class SD3ControlNetOutput(BaseOutput):
    # 控制网络块的样本，使用元组存储张量
    controlnet_block_samples: Tuple[torch.Tensor]

# 定义 SD3ControlNetModel 类，集成多种混合接口以实现模型功能
class SD3ControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    # 支持梯度检查点，允许节省内存
    _supports_gradient_checkpointing = True

    @register_to_config
    # 初始化方法，设置模型的各种参数，提供默认值
    def __init__(
        self,
        sample_size: int = 128,  # 输入样本大小
        patch_size: int = 2,  # 补丁大小
        in_channels: int = 16,  # 输入通道数
        num_layers: int = 18,  # 模型层数
        attention_head_dim: int = 64,  # 注意力头的维度
        num_attention_heads: int = 18,  # 注意力头的数量
        joint_attention_dim: int = 4096,  # 联合注意力的维度
        caption_projection_dim: int = 1152,  # 标题投影的维度
        pooled_projection_dim: int = 2048,  # 池化投影的维度
        out_channels: int = 16,  # 输出通道数
        pos_embed_max_size: int = 96,  # 位置嵌入的最大尺寸
    ):
        # 初始化父类
        super().__init__()
        # 默认输出通道设置为输入通道
        default_out_channels = in_channels
        # 输出通道为指定值或默认值
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        # 内部维度等于注意力头数量乘以每个头的维度
        self.inner_dim = num_attention_heads * attention_head_dim

        # 创建位置嵌入对象，用于处理图像补丁
        self.pos_embed = PatchEmbed(
            height=sample_size,  # 输入图像高度
            width=sample_size,   # 输入图像宽度
            patch_size=patch_size,  # 图像补丁大小
            in_channels=in_channels,  # 输入通道数量
            embed_dim=self.inner_dim,  # 嵌入维度
            pos_embed_max_size=pos_embed_max_size,  # 最大位置嵌入大小
        )
        # 创建时间和文本的联合嵌入
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,  # 嵌入维度
            pooled_projection_dim=pooled_projection_dim  # 聚合投影维度
        )
        # 定义上下文嵌入的线性层
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)

        # 注意力头维度加倍以适应混合
        # 需要在实际检查点中处理
        self.transformer_blocks = nn.ModuleList(
            [
                # 创建多个联合变换块
                JointTransformerBlock(
                    dim=self.inner_dim,  # 块的维度
                    num_attention_heads=num_attention_heads,  # 注意力头数量
                    attention_head_dim=self.config.attention_head_dim,  # 每个头的维度
                    context_pre_only=False,  # 是否仅上下文先行
                )
                for i in range(num_layers)  # 根据层数生成块
            ]
        )

        # 控制网络块
        self.controlnet_blocks = nn.ModuleList([])  # 初始化空的控制网络块列表
        for _ in range(len(self.transformer_blocks)):  # 根据变换块数量创建控制网络块
            controlnet_block = nn.Linear(self.inner_dim, self.inner_dim)  # 创建线性层
            controlnet_block = zero_module(controlnet_block)  # 零化模块以初始化
            self.controlnet_blocks.append(controlnet_block)  # 添加到控制网络块列表
        # 创建位置嵌入输入对象
        pos_embed_input = PatchEmbed(
            height=sample_size,  # 输入图像高度
            width=sample_size,   # 输入图像宽度
            patch_size=patch_size,  # 图像补丁大小
            in_channels=in_channels,  # 输入通道数量
            embed_dim=self.inner_dim,  # 嵌入维度
            pos_embed_type=None,  # 不使用位置嵌入类型
        )
        # 零化位置嵌入输入
        self.pos_embed_input = zero_module(pos_embed_input)

        # 关闭梯度检查点
        self.gradient_checkpointing = False

    # 从 diffusers.models.unets.unet_3d_condition.UNet3DConditionModel 复制的启用前向分块方法
    # 定义一个方法，启用前馈层的分块处理
        def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
            """
            设置注意力处理器使用前馈分块。
            
            参数：
                chunk_size (`int`, *optional*):
                    前馈层的分块大小。如果未指定，将对每个维度为`dim`的张量单独运行前馈层。
                dim (`int`, *optional*, defaults to `0`):
                    应该进行前馈计算的维度。可以选择dim=0（批次）或dim=1（序列长度）。
            """
            # 检查dim是否在允许的范围内
            if dim not in [0, 1]:
                raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")
    
            # 默认的分块大小为1
            chunk_size = chunk_size or 1
    
            # 定义一个递归函数，处理每个模块的前馈分块设置
            def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
                # 如果模块有设置前馈分块的方法，则调用它
                if hasattr(module, "set_chunk_feed_forward"):
                    module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
    
                # 递归处理子模块
                for child in module.children():
                    fn_recursive_feed_forward(child, chunk_size, dim)
    
            # 对当前对象的所有子模块应用分块设置
            for module in self.children():
                fn_recursive_feed_forward(module, chunk_size, dim)
    
        @property
        # 从其他模型复制的属性，返回注意力处理器
        def attn_processors(self) -> Dict[str, AttentionProcessor]:
            r"""
            返回：
                `dict` 注意力处理器：包含模型中所有注意力处理器的字典，按权重名称索引。
            """
            # 定义一个空字典来存储处理器
            processors = {}
    
            # 定义递归函数，添加处理器到字典中
            def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
                # 如果模块有获取处理器的方法，则将其添加到字典中
                if hasattr(module, "get_processor"):
                    processors[f"{name}.processor"] = module.get_processor()
    
                # 递归处理子模块
                for sub_name, child in module.named_children():
                    fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
                return processors
    
            # 对当前对象的所有子模块添加处理器
            for name, module in self.named_children():
                fn_recursive_add_processors(name, module, processors)
    
            # 返回所有处理器的字典
            return processors
    
        # 从其他模型复制的设置方法
    # 定义设置注意力处理器的方法，接收一个注意力处理器或处理器字典
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。

        参数：
            processor (`dict` of `AttentionProcessor` 或 `AttentionProcessor`):
                实例化的处理器类或将作为处理器设置到**所有** `Attention` 层的处理器类字典。

                如果 `processor` 是字典，键需要定义对应的交叉注意力处理器的路径。当设置可训练的注意力处理器时，强烈建议使用字典。

        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())

        # 如果传入的是字典且字典的长度与当前处理器数量不匹配，则抛出错误
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器字典，但处理器数量 {len(processor)} 与注意力层数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
            )

        # 定义递归处理注意力处理器的函数
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块有设置处理器的方法
            if hasattr(module, "set_processor"):
                # 如果传入的不是字典，则直接设置处理器
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中获取对应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 遍历子模块，递归调用自身
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前对象的子模块
        for name, module in self.named_children():
            # 对每个子模块调用递归处理器设置函数
            fn_recursive_attn_processor(name, module, processor)

    # 从 diffusers.models.transformers.transformer_sd3.SD3Transformer2DModel.fuse_qkv_projections 复制的方法
    def fuse_qkv_projections(self):
        """
        启用融合的 QKV 投影。对于自注意力模块，所有投影矩阵（即查询、键、值）被融合。
        对于交叉注意力模块，键和值的投影矩阵被融合。

        <提示 警告={true}>

        此 API 是 🧪 实验性的。

        </提示>
        """
        # 初始化原始注意力处理器为 None
        self.original_attn_processors = None

        # 检查所有注意力处理器，确保没有添加的 KV 投影
        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` 不支持具有添加的 KV 投影的模型。")

        # 保存当前的注意力处理器以备后用
        self.original_attn_processors = self.attn_processors

        # 遍历模型中的所有模块
        for module in self.modules():
            # 如果模块是 Attention 类型
            if isinstance(module, Attention):
                # 融合投影矩阵
                module.fuse_projections(fuse=True)

        # 设置新的融合注意力处理器
        self.set_attn_processor(FusedJointAttnProcessor2_0())

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections 复制的方法
    # 定义一个方法来禁用已启用的融合 QKV 投影
    def unfuse_qkv_projections(self):
        """如果启用了融合的 QKV 投影，则禁用它。

        <Tip warning={true}>

        此 API 是 🧪 实验性的。

        </Tip>

        """
        # 如果原始注意力处理器不为空，则恢复到原始设置
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    # 定义一个方法来设置梯度检查点
    def _set_gradient_checkpointing(self, module, value=False):
        # 如果模块有梯度检查点属性，则设置其值
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # 定义一个类方法从 Transformer 创建 ControlNet 实例
    @classmethod
    def from_transformer(cls, transformer, num_layers=12, load_weights_from_transformer=True):
        # 获取 Transformer 的配置
        config = transformer.config
        # 设置层数，如果未指定则使用配置中的层数
        config["num_layers"] = num_layers or config.num_layers
        # 创建 ControlNet 实例，传入配置参数
        controlnet = cls(**config)

        # 如果需要从 Transformer 加载权重
        if load_weights_from_transformer:
            # 加载位置嵌入的权重
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            # 加载时间文本嵌入的权重
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            # 加载上下文嵌入器的权重
            controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            # 加载变换器块的权重，严格模式为 False
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)

            # 将位置嵌入输入初始化为零模块
            controlnet.pos_embed_input = zero_module(controlnet.pos_embed_input)

        # 返回创建的 ControlNet 实例
        return controlnet

    # 定义前向传播方法
    def forward(
        # 输入的隐藏状态张量
        hidden_states: torch.FloatTensor,
        # 控制网条件张量
        controlnet_cond: torch.Tensor,
        # 条件缩放因子，默认值为 1.0
        conditioning_scale: float = 1.0,
        # 编码器隐藏状态张量，默认为 None
        encoder_hidden_states: torch.FloatTensor = None,
        # 池化投影张量，默认为 None
        pooled_projections: torch.FloatTensor = None,
        # 时间步长张量，默认为 None
        timestep: torch.LongTensor = None,
        # 联合注意力参数，默认为 None
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 是否返回字典格式的输出，默认为 True
        return_dict: bool = True,
# SD3MultiControlNetModel 类，继承自 ModelMixin
class SD3MultiControlNetModel(ModelMixin):
    r"""
    `SD3ControlNetModel` 的包装类，用于 Multi-SD3ControlNet

    该模块是多个 `SD3ControlNetModel` 实例的包装。`forward()` API 设计与 `SD3ControlNetModel` 兼容。

    参数:
        controlnets (`List[SD3ControlNetModel]`):
            在去噪过程中为 unet 提供额外的条件。必须将多个 `SD3ControlNetModel` 作为列表设置。
    """

    # 初始化函数，接收控制网列表并调用父类构造
    def __init__(self, controlnets):
        super().__init__()  # 调用父类的初始化方法
        self.nets = nn.ModuleList(controlnets)  # 将控制网列表存储为模块列表

    # 前向传播函数，接收多个输入参数以处理数据
    def forward(
        self,
        hidden_states: torch.FloatTensor,  # 隐藏状态张量
        controlnet_cond: List[torch.tensor],  # 控制网条件列表
        conditioning_scale: List[float],  # 条件缩放因子列表
        pooled_projections: torch.FloatTensor,  # 池化的投影张量
        encoder_hidden_states: torch.FloatTensor = None,  # 可选编码器隐藏状态
        timestep: torch.LongTensor = None,  # 可选时间步长
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的联合注意力参数
        return_dict: bool = True,  # 返回格式，默认为字典
    ) -> Union[SD3ControlNetOutput, Tuple]:  # 返回类型可以是输出对象或元组
        # 遍历控制网条件、缩放因子和控制网
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            # 调用控制网的前向传播以获取块样本
            block_samples = controlnet(
                hidden_states=hidden_states,  # 传递隐藏状态
                timestep=timestep,  # 传递时间步长
                encoder_hidden_states=encoder_hidden_states,  # 传递编码器隐藏状态
                pooled_projections=pooled_projections,  # 传递池化投影
                controlnet_cond=image,  # 传递控制网条件
                conditioning_scale=scale,  # 传递条件缩放因子
                joint_attention_kwargs=joint_attention_kwargs,  # 传递联合注意力参数
                return_dict=return_dict,  # 传递返回格式
            )

            # 合并样本
            if i == 0:  # 如果是第一个控制网
                control_block_samples = block_samples  # 直接使用块样本
            else:  # 如果不是第一个控制网
                # 将当前块样本与之前的样本逐元素相加
                control_block_samples = [
                    control_block_sample + block_sample
                    for control_block_sample, block_sample in zip(control_block_samples[0], block_samples[0])
                ]
                control_block_samples = (tuple(control_block_samples),)  # 将合并结果转为元组

        # 返回合并后的控制块样本
        return control_block_samples
```