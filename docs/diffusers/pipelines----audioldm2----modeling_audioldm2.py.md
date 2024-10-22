# `.\diffusers\pipelines\audioldm2\modeling_audioldm2.py`

```py
# 版权信息，说明此文件的版权归 HuggingFace 团队所有
# 
# 根据 Apache License 2.0（“许可证”）许可；
# 你只能在遵循许可证的情况下使用此文件。
# 你可以在以下地址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有规定，软件
# 根据许可证分发是“按原样”提供的，
# 不附带任何形式的保证或条件，无论是明示还是暗示的。
# 请参阅许可证了解特定语言的权限和
# 限制。

# 从 dataclasses 模块导入 dataclass 装饰器，用于简化类的创建
from dataclasses import dataclass
# 从 typing 模块导入类型注释工具
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 nn 模块以构建神经网络
import torch.nn as nn
# 导入 checkpoint 功能以支持模型检查点
import torch.utils.checkpoint

# 从配置工具模块导入 ConfigMixin 和 register_to_config
from ...configuration_utils import ConfigMixin, register_to_config
# 从加载器模块导入 UNet2DConditionLoadersMixin
from ...loaders import UNet2DConditionLoadersMixin
# 从激活函数模块导入 get_activation 函数
from ...models.activations import get_activation
# 从注意力处理器模块导入各种注意力处理器
from ...models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,  # 额外键值注意力处理器
    CROSS_ATTENTION_PROCESSORS,      # 交叉注意力处理器
    AttentionProcessor,               # 注意力处理器基类
    AttnAddedKVProcessor,            # 额外键值注意力处理器
    AttnProcessor,                   # 注意力处理器
)
# 从嵌入模块导入时间步嵌入和时间步类
from ...models.embeddings import (
    TimestepEmbedding,  # 时间步嵌入
    Timesteps,         # 时间步类
)
# 从建模工具模块导入 ModelMixin
from ...models.modeling_utils import ModelMixin
# 从 ResNet 模块导入下采样、ResNet 块和上采样类
from ...models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
# 从 2D 转换器模块导入 Transformer2DModel
from ...models.transformers.transformer_2d import Transformer2DModel
# 从 UNet 2D 块模块导入下块和上块类
from ...models.unets.unet_2d_blocks import DownBlock2D, UpBlock2D
# 从 UNet 2D 条件模块导入 UNet2DConditionOutput
from ...models.unets.unet_2d_condition import UNet2DConditionOutput
# 从工具模块导入 BaseOutput、is_torch_version 和 logging
from ...utils import BaseOutput, is_torch_version, logging

# 创建一个日志记录器，使用当前模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个函数，用于添加特殊标记到隐藏状态和注意力掩码
def add_special_tokens(hidden_states, attention_mask, sos_token, eos_token):
    # 获取批量大小
    batch_size = hidden_states.shape[0]

    # 如果存在注意力掩码
    if attention_mask is not None:
        # 为注意力掩码添加两个额外的步骤
        new_attn_mask_step = attention_mask.new_ones((batch_size, 1))  # 创建全为 1 的新步骤
        # 将新的步骤添加到注意力掩码的前后
        attention_mask = torch.concat([new_attn_mask_step, attention_mask, new_attn_mask_step], dim=-1)

    # 在序列的开始/结束处添加 SOS / EOS 标记
    sos_token = sos_token.expand(batch_size, 1, -1)  # 扩展 SOS 标记的维度
    eos_token = eos_token.expand(batch_size, 1, -1)  # 扩展 EOS 标记的维度
    # 将 SOS 和 EOS 标记添加到隐藏状态
    hidden_states = torch.concat([sos_token, hidden_states, eos_token], dim=1)
    # 返回更新后的隐藏状态和注意力掩码
    return hidden_states, attention_mask

# 定义一个数据类，用于表示音频 LDM2 投影模型的输出
@dataclass
class AudioLDM2ProjectionModelOutput(BaseOutput):
    """
    参数：
    # 定义一个类，用于存储 AudioLDM2 投影层的输出。
    # hidden_states: 一个形状为 (batch_size, sequence_length, hidden_size) 的张量，表示每个文本编码器的隐藏状态序列
    hidden_states: torch.Tensor
    # attention_mask: 一个形状为 (batch_size, sequence_length) 的可选张量，用于避免在填充标记索引上执行注意力
    attention_mask: Optional[torch.LongTensor] = None
# 定义一个音频 LDM2 投影模型类，继承自 ModelMixin 和 ConfigMixin
class AudioLDM2ProjectionModel(ModelMixin, ConfigMixin):
    """
    一个简单的线性投影模型，用于将两个文本嵌入映射到共享的潜在空间。
    它还在每个文本嵌入序列的开始和结束分别插入学习到的嵌入向量。
    每个以 `_1` 结尾的变量对应于第二个文本编码器的变量，其他则来自第一个。

    参数:
        text_encoder_dim (`int`):
            第一个文本编码器（CLAP）生成的文本嵌入的维度。
        text_encoder_1_dim (`int`):
            第二个文本编码器（T5 或 VITS）生成的文本嵌入的维度。
        langauge_model_dim (`int`):
            语言模型（GPT2）生成的文本嵌入的维度。
    """

    @register_to_config
    # 构造函数，初始化各个参数
    def __init__(
        self,
        text_encoder_dim,
        text_encoder_1_dim,
        langauge_model_dim,
        use_learned_position_embedding=None,
        max_seq_length=None,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 为每个文本编码器创建额外的投影层
        self.projection = nn.Linear(text_encoder_dim, langauge_model_dim)
        self.projection_1 = nn.Linear(text_encoder_1_dim, langauge_model_dim)

        # 为每个文本编码器的可学习 SOS/EOS 令牌嵌入
        self.sos_embed = nn.Parameter(torch.ones(langauge_model_dim))
        self.eos_embed = nn.Parameter(torch.ones(langauge_model_dim))

        self.sos_embed_1 = nn.Parameter(torch.ones(langauge_model_dim))
        self.eos_embed_1 = nn.Parameter(torch.ones(langauge_model_dim))

        # 保存是否使用学习到的位置嵌入
        self.use_learned_position_embedding = use_learned_position_embedding

        # 为 vits 编码器创建可学习的位置嵌入
        if self.use_learned_position_embedding is not None:
            self.learnable_positional_embedding = torch.nn.Parameter(
                # 初始化一个零张量作为可学习的位置嵌入
                torch.zeros((1, text_encoder_1_dim, max_seq_length))
            )

    # 定义前向传播方法
    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        hidden_states_1: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        attention_mask_1: Optional[torch.LongTensor] = None,
    # 进行隐藏状态的线性变换
        ):
            hidden_states = self.projection(hidden_states)
            # 为隐藏状态添加特殊标记，返回更新后的隐藏状态和注意力掩码
            hidden_states, attention_mask = add_special_tokens(
                hidden_states, attention_mask, sos_token=self.sos_embed, eos_token=self.eos_embed
            )
    
            # 如果使用学习的位置嵌入，则为 Vits 的隐藏状态添加位置嵌入
            if self.use_learned_position_embedding is not None:
                hidden_states_1 = (hidden_states_1.permute(0, 2, 1) + self.learnable_positional_embedding).permute(0, 2, 1)
    
            # 对隐藏状态进行线性变换
            hidden_states_1 = self.projection_1(hidden_states_1)
            # 为第二组隐藏状态添加特殊标记，返回更新后的隐藏状态和注意力掩码
            hidden_states_1, attention_mask_1 = add_special_tokens(
                hidden_states_1, attention_mask_1, sos_token=self.sos_embed_1, eos_token=self.eos_embed_1
            )
    
            # 将 clap 和 t5 的文本编码进行拼接
            hidden_states = torch.cat([hidden_states, hidden_states_1], dim=1)
    
            # 拼接注意力掩码
            if attention_mask is None and attention_mask_1 is not None:
                # 创建与 hidden_states 形状一致的全1张量作为注意力掩码
                attention_mask = attention_mask_1.new_ones((hidden_states[:2]))
            elif attention_mask is not None and attention_mask_1 is None:
                # 创建与 hidden_states_1 形状一致的全1张量作为注意力掩码
                attention_mask_1 = attention_mask.new_ones((hidden_states_1[:2]))
    
            # 如果两个注意力掩码都存在，则进行拼接
            if attention_mask is not None and attention_mask_1 is not None:
                attention_mask = torch.cat([attention_mask, attention_mask_1], dim=-1)
            else:
                # 如果没有有效的注意力掩码，则设为 None
                attention_mask = None
    
            # 返回包含隐藏状态和注意力掩码的输出对象
            return AudioLDM2ProjectionModelOutput(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
# 定义一个条件 2D UNet 模型，继承自多个混入类
class AudioLDM2UNet2DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    一个条件 2D UNet 模型，它接收一个带噪声的样本、条件状态和时间步，并返回一个样本
    形状的输出。与普通的 [`UNet2DConditionModel`] 相比，此变体可选择性地在每个 Transformer 块中包含额外的
    自注意力层，以及多个交叉注意力层。它还允许最多使用两个交叉注意力嵌入，即 `encoder_hidden_states` 和 `encoder_hidden_states_1`。

    此模型继承自 [`ModelMixin`]。请查看父类文档以获取为所有模型实现的通用方法
    （如下载或保存）。
    """

    # 支持梯度检查点
    _supports_gradient_checkpointing = True

    # 将初始化方法注册到配置中
    @register_to_config
    def __init__(
        # 样本大小的可选参数
        sample_size: Optional[int] = None,
        # 输入通道数，默认为 4
        in_channels: int = 4,
        # 输出通道数，默认为 4
        out_channels: int = 4,
        # 控制正弦和余弦的翻转，默认为 True
        flip_sin_to_cos: bool = True,
        # 频移量，默认为 0
        freq_shift: int = 0,
        # 各层块类型的元组，指定了下采样块的类型
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        # 中间块的类型，默认为 "UNetMidBlock2DCrossAttn"
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        # 各层块类型的元组，指定了上采样块的类型
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        # 仅使用交叉注意力的布尔值或元组，默认为 False
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        # 各块的输出通道数的元组
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        # 每个块的层数，默认为 2
        layers_per_block: Union[int, Tuple[int]] = 2,
        # 下采样填充，默认为 1
        downsample_padding: int = 1,
        # 中间块的缩放因子，默认为 1
        mid_block_scale_factor: float = 1,
        # 激活函数类型，默认为 "silu"
        act_fn: str = "silu",
        # 规范化的组数，默认为 32
        norm_num_groups: Optional[int] = 32,
        # 规范化的 epsilon 值，默认为 1e-5
        norm_eps: float = 1e-5,
        # 交叉注意力的维度，可以是整数或元组，默认为 1280
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        # 每个块的 Transformer 层数，可以是整数或元组，默认为 1
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        # 注意力头的维度，可以是整数或元组，默认为 8
        attention_head_dim: Union[int, Tuple[int]] = 8,
        # 注意力头的数量，可以是整数或元组，默认为 None
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        # 使用线性投影的布尔值，默认为 False
        use_linear_projection: bool = False,
        # 类嵌入类型的可选字符串
        class_embed_type: Optional[str] = None,
        # 类嵌入数量的可选整数
        num_class_embeds: Optional[int] = None,
        # 上溢注意力的布尔值，默认为 False
        upcast_attention: bool = False,
        # ResNet 时间缩放偏移，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # 时间嵌入类型，默认为 "positional"
        time_embedding_type: str = "positional",
        # 时间嵌入维度的可选整数
        time_embedding_dim: Optional[int] = None,
        # 时间嵌入激活函数的可选字符串
        time_embedding_act_fn: Optional[str] = None,
        # 时间步后激活的可选字符串
        timestep_post_act: Optional[str] = None,
        # 时间条件投影维度的可选整数
        time_cond_proj_dim: Optional[int] = None,
        # 输入卷积核大小，默认为 3
        conv_in_kernel: int = 3,
        # 输出卷积核大小，默认为 3
        conv_out_kernel: int = 3,
        # 投影类嵌入输入维度的可选整数
        projection_class_embeddings_input_dim: Optional[int] = None,
        # 类嵌入是否连接的布尔值，默认为 False
        class_embeddings_concat: bool = False,
    @property
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors 复制的属性
    # 返回模型中所有注意力处理器的字典，按权重名称索引
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # 初始化一个空字典，用于存储注意力处理器
        processors = {}
    
        # 定义一个递归函数，用于添加处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 检查模块是否具有获取处理器的方法
            if hasattr(module, "get_processor"):
                # 将处理器添加到字典中，键为处理器的名称
                processors[f"{name}.processor"] = module.get_processor()
    
            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用，以添加子模块的处理器
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
            # 返回处理器字典
            return processors
    
        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数，以添加子模块的处理器
            fn_recursive_add_processors(name, module, processors)
    
        # 返回包含所有处理器的字典
        return processors
    
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor 复制
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
        # 获取当前处理器字典中的处理器数量
        count = len(self.attn_processors.keys())
    
        # 检查传入的处理器字典与当前层数是否匹配
        if isinstance(processor, dict) and len(processor) != count:
            # 抛出值错误，提示处理器数量不匹配
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )
    
        # 定义一个递归函数，用于设置处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 检查模块是否具有设置处理器的方法
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，直接设置
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中弹出处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))
    
            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用，以设置子模块的处理器
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数，以设置子模块的处理器
            fn_recursive_attn_processor(name, module, processor)
    
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor 复制
    # 设置默认的注意力处理器，禁用自定义注意力处理器
    def set_default_attn_processor(self):
        # 文档字符串，说明此方法的功能
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        # 检查所有注意力处理器是否属于自定义 KV 注意力处理器类型
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 如果是，创建一个添加 KV 的注意力处理器实例
            processor = AttnAddedKVProcessor()
        # 检查所有注意力处理器是否属于交叉注意力处理器类型
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 如果是，创建一个标准的注意力处理器实例
            processor = AttnProcessor()
        else:
            # 如果两者都不是，抛出值错误异常
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )
    
        # 设置使用的注意力处理器
        self.set_attn_processor(processor)
    
        # 从 UNet2DConditionModel 复制的方法，用于设置梯度检查点
        # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attention_slice
        # 从 UNet2DConditionModel 复制的方法，用于设置梯度检查点
        # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel._set_gradient_checkpointing
        def _set_gradient_checkpointing(self, module, value=False):
            # 检查模块是否有梯度检查点属性
            if hasattr(module, "gradient_checkpointing"):
                # 如果有，将其设置为指定值
                module.gradient_checkpointing = value
    
        # 定义前向传播方法，接受多个参数
        def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            encoder_hidden_states_1: Optional[torch.Tensor] = None,
            encoder_attention_mask_1: Optional[torch.Tensor] = None,
# 定义一个获取下采样块的函数
def get_down_block(
    # 下采样块类型
    down_block_type,
    # 层数
    num_layers,
    # 输入通道数
    in_channels,
    # 输出通道数
    out_channels,
    # 时间嵌入通道数
    temb_channels,
    # 是否添加下采样
    add_downsample,
    # ResNet 的 epsilon 值
    resnet_eps,
    # ResNet 的激活函数
    resnet_act_fn,
    # 每个块的变换器层数（默认1）
    transformer_layers_per_block=1,
    # 注意力头数（可选）
    num_attention_heads=None,
    # ResNet 组数（可选）
    resnet_groups=None,
    # 跨注意力维度（可选）
    cross_attention_dim=None,
    # 下采样的填充（可选）
    downsample_padding=None,
    # 是否使用线性投影（默认 False）
    use_linear_projection=False,
    # 是否仅使用跨注意力（默认 False）
    only_cross_attention=False,
    # 是否上溯注意力（默认 False）
    upcast_attention=False,
    # ResNet 时间尺度偏移（默认值）
    resnet_time_scale_shift="default",
):
    # 如果下采样块类型以 "UNetRes" 开头，则去掉前缀
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    # 如果下采样块类型为 "DownBlock2D"
    if down_block_type == "DownBlock2D":
        # 返回 DownBlock2D 对象，并传递相关参数
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    # 如果下采样块类型为 "CrossAttnDownBlock2D"
    elif down_block_type == "CrossAttnDownBlock2D":
        # 如果跨注意力维度未指定，抛出异常
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        # 返回 CrossAttnDownBlock2D 对象，并传递相关参数
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    # 如果下采样块类型不匹配，抛出异常
    raise ValueError(f"{down_block_type} does not exist.")


# 定义一个获取上采样块的函数
def get_up_block(
    # 上采样块类型
    up_block_type,
    # 层数
    num_layers,
    # 输入通道数
    in_channels,
    # 输出通道数
    out_channels,
    # 上一输出通道数
    prev_output_channel,
    # 时间嵌入通道数
    temb_channels,
    # 是否添加上采样
    add_upsample,
    # ResNet 的 epsilon 值
    resnet_eps,
    # ResNet 的激活函数
    resnet_act_fn,
    # 每个块的变换器层数（默认1）
    transformer_layers_per_block=1,
    # 注意力头数（可选）
    num_attention_heads=None,
    # ResNet 组数（可选）
    resnet_groups=None,
    # 跨注意力维度（可选）
    cross_attention_dim=None,
    # 是否使用线性投影（默认 False）
    use_linear_projection=False,
    # 是否仅使用跨注意力（默认 False）
    only_cross_attention=False,
    # 是否上溯注意力（默认 False）
    upcast_attention=False,
    # ResNet 时间尺度偏移（默认值）
    resnet_time_scale_shift="default",
):
    # 如果上采样块类型以 "UNetRes" 开头，则去掉前缀
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    # 判断上升块的类型是否为 "UpBlock2D"
        if up_block_type == "UpBlock2D":
            # 返回 UpBlock2D 对象，传入所需参数
            return UpBlock2D(
                # 指定层数
                num_layers=num_layers,
                # 输入通道数
                in_channels=in_channels,
                # 输出通道数
                out_channels=out_channels,
                # 前一层的输出通道数
                prev_output_channel=prev_output_channel,
                # 时间嵌入通道数
                temb_channels=temb_channels,
                # 是否添加上采样
                add_upsample=add_upsample,
                # ResNet 的 epsilon 值
                resnet_eps=resnet_eps,
                # ResNet 的激活函数
                resnet_act_fn=resnet_act_fn,
                # ResNet 的分组数
                resnet_groups=resnet_groups,
                # ResNet 的时间尺度偏移
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
        # 判断上升块的类型是否为 "CrossAttnUpBlock2D"
        elif up_block_type == "CrossAttnUpBlock2D":
            # 如果未指定 cross_attention_dim，抛出错误
            if cross_attention_dim is None:
                raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
            # 返回 CrossAttnUpBlock2D 对象，传入所需参数
            return CrossAttnUpBlock2D(
                # 指定层数
                num_layers=num_layers,
                # 每个块的变换层数
                transformer_layers_per_block=transformer_layers_per_block,
                # 输入通道数
                in_channels=in_channels,
                # 输出通道数
                out_channels=out_channels,
                # 前一层的输出通道数
                prev_output_channel=prev_output_channel,
                # 时间嵌入通道数
                temb_channels=temb_channels,
                # 是否添加上采样
                add_upsample=add_upsample,
                # ResNet 的 epsilon 值
                resnet_eps=resnet_eps,
                # ResNet 的激活函数
                resnet_act_fn=resnet_act_fn,
                # ResNet 的分组数
                resnet_groups=resnet_groups,
                # 跨注意力维度
                cross_attention_dim=cross_attention_dim,
                # 注意力头的数量
                num_attention_heads=num_attention_heads,
                # 是否使用线性投影
                use_linear_projection=use_linear_projection,
                # 是否仅使用跨注意力
                only_cross_attention=only_cross_attention,
                # 是否上采样注意力
                upcast_attention=upcast_attention,
                # ResNet 的时间尺度偏移
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
        # 抛出错误，指示上升块类型不存在
        raise ValueError(f"{up_block_type} does not exist.")
# 定义一个二维交叉注意力下采样块，继承自 nn.Module
class CrossAttnDownBlock2D(nn.Module):
    # 初始化方法，设置下采样块的参数
    def __init__(
        # 输入通道数
        self,
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # Dropout 概率，默认为 0.0
        dropout: float = 0.0,
        # 层数，默认为 1
        num_layers: int = 1,
        # 每个块中的变换器层数，默认为 1
        transformer_layers_per_block: int = 1,
        # ResNet 的 epsilon 值，默认为 1e-6
        resnet_eps: float = 1e-6,
        # ResNet 的时间缩放偏移，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 的激活函数，默认为 "swish"
        resnet_act_fn: str = "swish",
        # ResNet 的组数，默认为 32
        resnet_groups: int = 32,
        # 是否在 ResNet 中使用预归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头的数量，默认为 1
        num_attention_heads=1,
        # 交叉注意力的维度，默认为 1280
        cross_attention_dim=1280,
        # 输出缩放因子，默认为 1.0
        output_scale_factor=1.0,
        # 下采样的填充，默认为 1
        downsample_padding=1,
        # 是否添加下采样层，默认为 True
        add_downsample=True,
        # 是否使用线性投影，默认为 False
        use_linear_projection=False,
        # 是否仅使用交叉注意力，默认为 False
        only_cross_attention=False,
        # 是否上采样注意力，默认为 False
        upcast_attention=False,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化用于存储 ResNet 模块的列表
        resnets = []
        # 初始化用于存储注意力模块的列表
        attentions = []

        # 设置是否使用交叉注意力
        self.has_cross_attention = True
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads

        # 检查交叉注意力维度是否为整数类型
        if isinstance(cross_attention_dim, int):
            # 将交叉注意力维度转换为元组
            cross_attention_dim = (cross_attention_dim,)
        # 检查交叉注意力维度是否为列表或元组，且长度超过 4
        if isinstance(cross_attention_dim, (list, tuple)) and len(cross_attention_dim) > 4:
            # 如果超过，抛出值错误
            raise ValueError(
                "Only up to 4 cross-attention layers are supported. Ensure that the length of cross-attention "
                f"dims is less than or equal to 4. Got cross-attention dims {cross_attention_dim} of length {len(cross_attention_dim)}"
            )
        # 设置交叉注意力维度
        self.cross_attention_dim = cross_attention_dim

        # 遍历层数以构建 ResNet 模块
        for i in range(num_layers):
            # 确定输入通道数，第一层使用 in_channels，其余层使用 out_channels
            in_channels = in_channels if i == 0 else out_channels
            # 添加 ResNet 块到列表中
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            # 为每个交叉注意力维度添加 Transformer 模型
            for j in range(len(cross_attention_dim)):
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim[j],
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        double_self_attention=True if cross_attention_dim[j] is None else False,
                    )
                )
        # 将注意力模块列表转换为 nn.ModuleList 以便于管理
        self.attentions = nn.ModuleList(attentions)
        # 将 ResNet 模块列表转换为 nn.ModuleList 以便于管理
        self.resnets = nn.ModuleList(resnets)

        # 检查是否添加下采样模块
        if add_downsample:
            # 创建下采样模块的 nn.ModuleList
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            # 如果不添加下采样，则设置为 None
            self.downsamplers = None

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False
    # 定义一个前向传播的方法，接受多个输入参数
        def forward(
            self,
            # 输入的隐藏状态张量
            hidden_states: torch.Tensor,
            # 可选的时间嵌入张量
            temb: Optional[torch.Tensor] = None,
            # 可选的编码器隐藏状态张量
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选的注意力掩码张量
            attention_mask: Optional[torch.Tensor] = None,
            # 可选的交叉注意力参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的编码器注意力掩码张量
            encoder_attention_mask: Optional[torch.Tensor] = None,
            # 可选的第二个编码器隐藏状态张量
            encoder_hidden_states_1: Optional[torch.Tensor] = None,
            # 可选的第二个编码器注意力掩码张量
            encoder_attention_mask_1: Optional[torch.Tensor] = None,
# 定义一个继承自 nn.Module 的 UNetMidBlock2DCrossAttn 类
class UNetMidBlock2DCrossAttn(nn.Module):
    # 初始化方法，设置各类参数
    def __init__(
        # 输入通道数
        self,
        in_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # Dropout 概率，默认为 0.0
        dropout: float = 0.0,
        # 层数，默认为 1
        num_layers: int = 1,
        # 每个块中的变换器层数，默认为 1
        transformer_layers_per_block: int = 1,
        # ResNet 的 epsilon 值，默认为 1e-6
        resnet_eps: float = 1e-6,
        # ResNet 时间缩放偏移方式，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 激活函数类型，默认为 "swish"
        resnet_act_fn: str = "swish",
        # ResNet 中的分组数，默认为 32
        resnet_groups: int = 32,
        # 是否在 ResNet 中使用预归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头的数量，默认为 1
        num_attention_heads=1,
        # 输出缩放因子，默认为 1.0
        output_scale_factor=1.0,
        # 跨注意力维度，默认为 1280
        cross_attention_dim=1280,
        # 是否使用线性投影，默认为 False
        use_linear_projection=False,
        # 是否提升注意力精度，默认为 False
        upcast_attention=False,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 设置是否使用交叉注意力标志为 True
        self.has_cross_attention = True
        # 存储注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 如果 resnet_groups 未提供，计算其默认值
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # 如果 cross_attention_dim 是整数，则将其转换为元组
        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,)
        # 如果 cross_attention_dim 是列表或元组且长度大于 4，抛出异常
        if isinstance(cross_attention_dim, (list, tuple)) and len(cross_attention_dim) > 4:
            raise ValueError(
                "Only up to 4 cross-attention layers are supported. Ensure that the length of cross-attention "
                f"dims is less than or equal to 4. Got cross-attention dims {cross_attention_dim} of length {len(cross_attention_dim)}"
            )
        # 存储交叉注意力的维度
        self.cross_attention_dim = cross_attention_dim

        # 至少有一个 ResNet 模块
        resnets = [
            # 初始化 ResnetBlock2D 模块
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        # 初始化注意力模块列表
        attentions = []

        # 创建层数的循环
        for i in range(num_layers):
            # 遍历每个交叉注意力维度
            for j in range(len(cross_attention_dim)):
                # 添加 Transformer2DModel 到注意力列表
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim[j],
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        # 如果当前交叉注意力维度为 None，设置 double_self_attention 为 True
                        double_self_attention=True if cross_attention_dim[j] is None else False,
                    )
                )
            # 在 ResNet 列表中添加另一个 ResnetBlock2D
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        # 将注意力模块列表转换为 nn.ModuleList
        self.attentions = nn.ModuleList(attentions)
        # 将 ResNet 模块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False
    # 定义前向传播函数，接收多个输入参数
        def forward(
            self,
            # 输入的隐藏状态，类型为 torch.Tensor
            hidden_states: torch.Tensor,
            # 可选的时间嵌入，类型为 torch.Tensor
            temb: Optional[torch.Tensor] = None,
            # 可选的编码器隐藏状态，类型为 torch.Tensor
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选的注意力掩码，类型为 torch.Tensor
            attention_mask: Optional[torch.Tensor] = None,
            # 可选的交叉注意力参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的编码器注意力掩码，类型为 torch.Tensor
            encoder_attention_mask: Optional[torch.Tensor] = None,
            # 可选的第二组编码器隐藏状态，类型为 torch.Tensor
            encoder_hidden_states_1: Optional[torch.Tensor] = None,
            # 可选的第二组编码器注意力掩码，类型为 torch.Tensor
            encoder_attention_mask_1: Optional[torch.Tensor] = None,
# 定义一个名为 CrossAttnUpBlock2D 的类，继承自 nn.Module
class CrossAttnUpBlock2D(nn.Module):
    # 初始化方法，定义该类的构造函数
    def __init__(
        # 输入通道数
        self,
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 上一层输出通道数
        prev_output_channel: int,
        # 时间嵌入通道数
        temb_channels: int,
        # dropout 概率，默认为 0.0
        dropout: float = 0.0,
        # 层数，默认为 1
        num_layers: int = 1,
        # 每个块的 transformer 层数，默认为 1
        transformer_layers_per_block: int = 1,
        # ResNet 中的 epsilon 值，默认为 1e-6
        resnet_eps: float = 1e-6,
        # ResNet 的时间尺度偏移方式，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 的激活函数，默认为 "swish"
        resnet_act_fn: str = "swish",
        # ResNet 的分组数，默认为 32
        resnet_groups: int = 32,
        # ResNet 是否预先归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头的数量，默认为 1
        num_attention_heads=1,
        # 交叉注意力的维度，默认为 1280
        cross_attention_dim=1280,
        # 输出缩放因子，默认为 1.0
        output_scale_factor=1.0,
        # 是否添加上采样，默认为 True
        add_upsample=True,
        # 是否使用线性投影，默认为 False
        use_linear_projection=False,
        # 是否仅使用交叉注意力，默认为 False
        only_cross_attention=False,
        # 是否上溢出注意力，默认为 False
        upcast_attention=False,
    # 初始化类，调用父类构造函数
        ):
            super().__init__()
            # 初始化空列表用于存储残差网络块
            resnets = []
            # 初始化空列表用于存储注意力模型
            attentions = []
    
            # 设置是否使用交叉注意力
            self.has_cross_attention = True
            # 存储注意力头的数量
            self.num_attention_heads = num_attention_heads
    
            # 如果交叉注意力维度是整数，将其转为元组
            if isinstance(cross_attention_dim, int):
                cross_attention_dim = (cross_attention_dim,)
            # 检查交叉注意力维度是否为列表或元组且长度超过4
            if isinstance(cross_attention_dim, (list, tuple)) and len(cross_attention_dim) > 4:
                # 抛出错误，限制交叉注意力层数
                raise ValueError(
                    "Only up to 4 cross-attention layers are supported. Ensure that the length of cross-attention "
                    f"dims is less than or equal to 4. Got cross-attention dims {cross_attention_dim} of length {len(cross_attention_dim)}"
                )
            # 存储交叉注意力维度
            self.cross_attention_dim = cross_attention_dim
    
            # 遍历层数以构建残差块和注意力模型
            for i in range(num_layers):
                # 设置残差跳跃通道数
                res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
                # 设置当前残差块的输入通道数
                resnet_in_channels = prev_output_channel if i == 0 else out_channels
    
                # 创建并添加残差块到列表
                resnets.append(
                    ResnetBlock2D(
                        # 计算输入通道数
                        in_channels=resnet_in_channels + res_skip_channels,
                        # 设置输出通道数
                        out_channels=out_channels,
                        # 传递时间嵌入通道数
                        temb_channels=temb_channels,
                        # 设置小常数以防止除零
                        eps=resnet_eps,
                        # 设置组数
                        groups=resnet_groups,
                        # 设置dropout比例
                        dropout=dropout,
                        # 设置时间嵌入的归一化
                        time_embedding_norm=resnet_time_scale_shift,
                        # 设置激活函数
                        non_linearity=resnet_act_fn,
                        # 设置输出缩放因子
                        output_scale_factor=output_scale_factor,
                        # 设置是否使用预归一化
                        pre_norm=resnet_pre_norm,
                    )
                )
                # 为每个交叉注意力维度创建注意力模型
                for j in range(len(cross_attention_dim)):
                    attentions.append(
                        Transformer2DModel(
                            # 设置注意力头数
                            num_attention_heads,
                            # 计算每个头的输出通道数
                            out_channels // num_attention_heads,
                            # 设置输入通道数
                            in_channels=out_channels,
                            # 设置每个块的层数
                            num_layers=transformer_layers_per_block,
                            # 设置当前交叉注意力维度
                            cross_attention_dim=cross_attention_dim[j],
                            # 设置组归一化数
                            norm_num_groups=resnet_groups,
                            # 设置是否使用线性投影
                            use_linear_projection=use_linear_projection,
                            # 设置是否只使用交叉注意力
                            only_cross_attention=only_cross_attention,
                            # 设置是否上溯注意力
                            upcast_attention=upcast_attention,
                            # 设置是否双重自注意力
                            double_self_attention=True if cross_attention_dim[j] is None else False,
                        )
                    )
            # 将注意力模型列表转换为模块列表
            self.attentions = nn.ModuleList(attentions)
            # 将残差块列表转换为模块列表
            self.resnets = nn.ModuleList(resnets)
    
            # 根据条件添加上采样模块
            if add_upsample:
                self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
            else:
                # 如果不添加上采样，设为 None
                self.upsamplers = None
    
            # 设置梯度检查点
            self.gradient_checkpointing = False
    # 定义一个前向传播函数
        def forward(
            # 输入参数：当前隐藏状态的张量
            self,
            hidden_states: torch.Tensor,
            # 输入参数：包含之前隐藏状态的元组
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            # 可选参数：时间嵌入的张量
            temb: Optional[torch.Tensor] = None,
            # 可选参数：编码器的隐藏状态张量
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选参数：交叉注意力的关键字参数字典
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选参数：上采样的目标大小
            upsample_size: Optional[int] = None,
            # 可选参数：注意力掩码的张量
            attention_mask: Optional[torch.Tensor] = None,
            # 可选参数：编码器的注意力掩码张量
            encoder_attention_mask: Optional[torch.Tensor] = None,
            # 可选参数：编码器隐藏状态的另一个张量
            encoder_hidden_states_1: Optional[torch.Tensor] = None,
            # 可选参数：编码器隐藏状态的注意力掩码张量
            encoder_attention_mask_1: Optional[torch.Tensor] = None,
```