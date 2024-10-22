# `.\diffusers\models\transformers\transformer_sd3.py`

```
# 版权所有 2024 Stability AI, The HuggingFace Team 和 The InstantX Team。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）许可；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有规定，按照许可证分发的软件是以“原样”基础提供的，
# 不提供任何形式的保证或条件，无论是明示的还是暗示的。
# 有关许可证下权限和限制的具体语言，请参见许可证。


from typing import Any, Dict, List, Optional, Union  # 从 typing 模块导入各种类型注释

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并命名为 nn

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入配置混合类和注册函数
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin  # 从加载器导入模型混合类
from ...models.attention import JointTransformerBlock  # 从注意力模块导入联合变换器块
from ...models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0  # 导入不同的注意力处理器
from ...models.modeling_utils import ModelMixin  # 导入模型混合类
from ...models.normalization import AdaLayerNormContinuous  # 导入自适应层归一化模块
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers  # 导入工具函数和变量
from ..embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed  # 从嵌入模块导入嵌入类
from ..modeling_outputs import Transformer2DModelOutput  # 导入变换器 2D 模型输出类


logger = logging.get_logger(__name__)  # 创建一个记录器实例，名称为当前模块名，禁用 pylint 对名称的警告


class SD3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):  # 定义 SD3 变换器 2D 模型类，继承多个混合类
    """
    Stable Diffusion 3 中引入的变换器模型。

    参考文献: https://arxiv.org/abs/2403.03206

    参数：
        sample_size (`int`): 潜在图像的宽度。训练期间固定使用，因为
            它用于学习一组位置嵌入。
        patch_size (`int`): 将输入数据转化为小块的块大小。
        in_channels (`int`, *可选*, 默认为 16): 输入的通道数量。
        num_layers (`int`, *可选*, 默认为 18): 使用的变换器块层数。
        attention_head_dim (`int`, *可选*, 默认为 64): 每个头的通道数量。
        num_attention_heads (`int`, *可选*, 默认为 18): 多头注意力使用的头数。
        cross_attention_dim (`int`, *可选*): 用于 `encoder_hidden_states` 维度的数量。
        caption_projection_dim (`int`): 用于投影 `encoder_hidden_states` 的维度数量。
        pooled_projection_dim (`int`): 用于投影 `pooled_projections` 的维度数量。
        out_channels (`int`, 默认为 16): 输出通道的数量。

    """

    _supports_gradient_checkpointing = True  # 表示模型支持梯度检查点功能

    @register_to_config  # 使用装饰器将此方法注册到配置中
    # 初始化方法，设置模型的基本参数
        def __init__(
            self,
            sample_size: int = 128,  # 输入样本的大小，默认值为128
            patch_size: int = 2,  # 每个补丁的大小，默认值为2
            in_channels: int = 16,  # 输入通道数，默认值为16
            num_layers: int = 18,  # Transformer层的数量，默认值为18
            attention_head_dim: int = 64,  # 每个注意力头的维度，默认值为64
            num_attention_heads: int = 18,  # 注意力头的数量，默认值为18
            joint_attention_dim: int = 4096,  # 联合注意力维度，默认值为4096
            caption_projection_dim: int = 1152,  # 标题投影维度，默认值为1152
            pooled_projection_dim: int = 2048,  # 池化投影维度，默认值为2048
            out_channels: int = 16,  # 输出通道数，默认值为16
            pos_embed_max_size: int = 96,  # 位置嵌入的最大大小，默认值为96
        ):
            super().__init__()  # 调用父类的初始化方法
            default_out_channels = in_channels  # 设置默认的输出通道为输入通道数
            # 如果指定输出通道，则使用指定值，否则使用默认值
            self.out_channels = out_channels if out_channels is not None else default_out_channels
            # 计算内部维度，等于注意力头数量乘以每个注意力头的维度
            self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
    
            # 创建位置嵌入模块，用于将输入图像转为嵌入表示
            self.pos_embed = PatchEmbed(
                height=self.config.sample_size,  # 高度设置为样本大小
                width=self.config.sample_size,  # 宽度设置为样本大小
                patch_size=self.config.patch_size,  # 补丁大小
                in_channels=self.config.in_channels,  # 输入通道数
                embed_dim=self.inner_dim,  # 嵌入维度
                pos_embed_max_size=pos_embed_max_size,  # 当前硬编码位置嵌入最大大小
            )
            # 创建时间与文本嵌入的组合模块
            self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim,  # 嵌入维度
                pooled_projection_dim=self.config.pooled_projection_dim  # 池化投影维度
            )
            # 创建线性层，用于将上下文信息映射到标题投影维度
            self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)
    
            # 创建Transformer块的列表
            self.transformer_blocks = nn.ModuleList(
                [
                    JointTransformerBlock(
                        dim=self.inner_dim,  # 输入维度为内部维度
                        num_attention_heads=self.config.num_attention_heads,  # 注意力头的数量
                        attention_head_dim=self.config.attention_head_dim,  # 每个注意力头的维度
                        context_pre_only=i == num_layers - 1,  # 仅在最后一层设置上下文优先
                    )
                    for i in range(self.config.num_layers)  # 遍历创建每一层
                ]
            )
    
            # 创建自适应层归一化层
            self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
            # 创建线性层，用于输出映射
            self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
    
            # 设置梯度检查点开关，默认值为False
            self.gradient_checkpointing = False
    
        # 从diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking复制的方法
    # 定义一个启用前馈分块的函数，接受可选的分块大小和维度
        def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
            """
            设置注意力处理器使用前馈分块机制。
    
            参数：
                chunk_size (`int`, *可选*):
                    前馈层的分块大小。如果未指定，将单独在维度为`dim`的每个张量上运行前馈层。
                dim (`int`, *可选*, 默认值为`0`):
                    前馈计算应分块的维度。选择dim=0（批量）或dim=1（序列长度）。
            """
            # 如果维度不是0或1，则抛出值错误
            if dim not in [0, 1]:
                raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")
    
            # 默认分块大小为1
            chunk_size = chunk_size or 1
    
            # 定义递归前馈函数，接受模块、分块大小和维度作为参数
            def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
                # 如果模块有设置分块前馈的属性，则调用该方法
                if hasattr(module, "set_chunk_feed_forward"):
                    module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
    
                # 遍历模块的子模块并递归调用前馈函数
                for child in module.children():
                    fn_recursive_feed_forward(child, chunk_size, dim)
    
            # 遍历当前对象的子模块，应用递归前馈函数
            for module in self.children():
                fn_recursive_feed_forward(module, chunk_size, dim)
    
        # 从diffusers.models.unets.unet_3d_condition复制的方法，禁用前馈分块
        def disable_forward_chunking(self):
            # 定义递归前馈函数，接受模块、分块大小和维度作为参数
            def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
                # 如果模块有设置分块前馈的属性，则调用该方法
                if hasattr(module, "set_chunk_feed_forward"):
                    module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
    
                # 遍历模块的子模块并递归调用前馈函数
                for child in module.children():
                    fn_recursive_feed_forward(child, chunk_size, dim)
    
            # 遍历当前对象的子模块，应用递归前馈函数，分块大小为None，维度为0
            for module in self.children():
                fn_recursive_feed_forward(module, None, 0)
    
        @property
        # 从diffusers.models.unets.unet_2d_condition复制的属性，获取注意力处理器
        def attn_processors(self) -> Dict[str, AttentionProcessor]:
            r"""
            返回：
                `dict`类型的注意力处理器：一个包含模型中所有注意力处理器的字典，以其权重名称索引。
            """
            # 初始化处理器字典
            processors = {}
    
            # 定义递归添加处理器的函数
            def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
                # 如果模块有获取处理器的方法，则将其添加到处理器字典
                if hasattr(module, "get_processor"):
                    processors[f"{name}.processor"] = module.get_processor()
    
                # 遍历子模块并递归调用添加处理器函数
                for sub_name, child in module.named_children():
                    fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
                return processors
    
            # 遍历当前对象的子模块，应用递归添加处理器函数
            for name, module in self.named_children():
                fn_recursive_add_processors(name, module, processors)
    
            # 返回处理器字典
            return processors
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor 复制而来
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。

        参数：
            processor（`dict` 类型的 `AttentionProcessor` 或仅为 `AttentionProcessor`）：
                实例化的处理器类或一个处理器类的字典，将被设置为 **所有** `Attention` 层的处理器。

                如果 `processor` 是一个字典，键需要定义相应的交叉注意力处理器的路径。
                在设置可训练的注意力处理器时，强烈建议使用这种方式。

        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())

        # 如果传入的处理器是字典且数量与注意力层数量不匹配，抛出异常
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器字典，但处理器数量 {len(processor)} 与注意力层数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
            )

        # 定义递归函数，用于设置每个模块的注意力处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 检查模块是否具有设置处理器的方法
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，直接设置
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中弹出相应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 遍历模块的子模块，递归调用
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前实例的子模块，并为每个模块设置处理器
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections 复制而来
    def fuse_qkv_projections(self):
        """
        启用融合的 QKV 投影。对于自注意力模块，所有投影矩阵（即查询、键、值）被融合。
        对于交叉注意力模块，键和值投影矩阵被融合。

        <Tip warning={true}>

        此 API 是 🧪 实验性。

        </Tip>
        """
        # 初始化原始注意力处理器为 None
        self.original_attn_processors = None

        # 检查每个注意力处理器是否包含 "Added"
        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                # 如果发现不支持的处理器，抛出异常
                raise ValueError("`fuse_qkv_projections()` 不支持具有添加 KV 投影的模型。")

        # 将当前的注意力处理器保存为原始处理器
        self.original_attn_processors = self.attn_processors

        # 遍历所有模块，如果模块是 Attention 类型，则进行投影融合
        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        # 设置注意力处理器为 FusedJointAttnProcessor2_0 的实例
        self.set_attn_processor(FusedJointAttnProcessor2_0())

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections 复制而来
    # 定义一个方法，用于禁用已启用的融合 QKV 投影
    def unfuse_qkv_projections(self):
        """禁用融合的 QKV 投影（如果已启用）。
    
        <Tip warning={true}>
    
        此 API 为 🧪 实验性。
    
        </Tip>
    
        """
        # 检查原始注意力处理器是否存在
        if self.original_attn_processors is not None:
            # 将当前的注意力处理器设置为原始的
            self.set_attn_processor(self.original_attn_processors)
    
    # 定义一个私有方法，用于设置梯度检查点
    def _set_gradient_checkpointing(self, module, value=False):
        # 检查模块是否具有梯度检查点属性
        if hasattr(module, "gradient_checkpointing"):
            # 将梯度检查点属性设置为指定值
            module.gradient_checkpointing = value
    
    # 定义前向传播方法，接受多个输入参数
    def forward(
        self,
        hidden_states: torch.FloatTensor,  # 输入的隐藏状态张量
        encoder_hidden_states: torch.FloatTensor = None,  # 编码器的隐藏状态张量，可选
        pooled_projections: torch.FloatTensor = None,  # 池化后的投影张量，可选
        timestep: torch.LongTensor = None,  # 时间步长张量，可选
        block_controlnet_hidden_states: List = None,  # 控制网的隐藏状态列表，可选
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,  # 联合注意力的额外参数，可选
        return_dict: bool = True,  # 指示是否返回字典格式的结果，默认为 True
```