# `.\diffusers\models\unets\unet_kandinsky3.py`

```py
# 版权声明，指明该文件属于 HuggingFace 团队，所有权利保留
# 
# 根据 Apache License 2.0 版（“许可证”）授权；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是在“原样”基础上分发的，不附带任何形式的保证或条件。
# 有关特定语言的许可条款和条件，请参见许可证。

from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Dict, Tuple, Union  # 导入用于类型提示的字典、元组和联合类型

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的检查点工具
from torch import nn  # 从 PyTorch 导入神经网络模块

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入混合类和注册函数
from ...utils import BaseOutput, logging  # 从工具模块导入基础输出类和日志功能
from ..attention_processor import Attention, AttentionProcessor, AttnProcessor  # 导入注意力处理器相关类
from ..embeddings import TimestepEmbedding, Timesteps  # 导入时间步嵌入相关类
from ..modeling_utils import ModelMixin  # 导入模型混合类

logger = logging.get_logger(__name__)  # 创建一个记录器，用于当前模块的日志记录

@dataclass  # 将该类标记为数据类，以简化初始化和表示
class Kandinsky3UNetOutput(BaseOutput):  # 定义 Kandinsky3UNetOutput 类，继承自 BaseOutput
    sample: torch.Tensor = None  # 定义输出样本，默认为 None

class Kandinsky3EncoderProj(nn.Module):  # 定义 Kandinsky3EncoderProj 类，继承自 nn.Module
    def __init__(self, encoder_hid_dim, cross_attention_dim):  # 初始化方法，接收隐藏维度和交叉注意力维度
        super().__init__()  # 调用父类的初始化方法
        self.projection_linear = nn.Linear(encoder_hid_dim, cross_attention_dim, bias=False)  # 定义线性投影层，不使用偏置
        self.projection_norm = nn.LayerNorm(cross_attention_dim)  # 定义层归一化层

    def forward(self, x):  # 定义前向传播方法
        x = self.projection_linear(x)  # 通过线性层处理输入
        x = self.projection_norm(x)  # 通过层归一化处理输出
        return x  # 返回处理后的结果

class Kandinsky3UNet(ModelMixin, ConfigMixin):  # 定义 Kandinsky3UNet 类，继承自 ModelMixin 和 ConfigMixin
    @register_to_config  # 将该方法注册到配置中
    def __init__(  # 初始化方法
        self,
        in_channels: int = 4,  # 输入通道数，默认值为 4
        time_embedding_dim: int = 1536,  # 时间嵌入维度，默认值为 1536
        groups: int = 32,  # 组数，默认值为 32
        attention_head_dim: int = 64,  # 注意力头维度，默认值为 64
        layers_per_block: Union[int, Tuple[int]] = 3,  # 每个块的层数，默认值为 3，可以是整数或元组
        block_out_channels: Tuple[int] = (384, 768, 1536, 3072),  # 块输出通道，默认为指定元组
        cross_attention_dim: Union[int, Tuple[int]] = 4096,  # 交叉注意力维度，默认值为 4096
        encoder_hid_dim: int = 4096,  # 编码器隐藏维度，默认值为 4096
    @property  # 定义一个属性
    def attn_processors(self) -> Dict[str, AttentionProcessor]:  # 返回注意力处理器字典
        r"""  # 文档字符串，描述该方法的功能
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # 设置一个空字典以递归存储处理器
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):  # 定义递归函数添加处理器
            if hasattr(module, "set_processor"):  # 检查模块是否具有 set_processor 属性
                processors[f"{name}.processor"] = module.processor  # 将处理器添加到字典中

            for sub_name, child in module.named_children():  # 遍历模块的所有子模块
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)  # 递归调用自身

            return processors  # 返回更新后的处理器字典

        for name, module in self.named_children():  # 遍历当前类的所有子模块
            fn_recursive_add_processors(name, module, processors)  # 调用递归函数

        return processors  # 返回包含所有处理器的字典
    # 定义设置注意力处理器的方法，参数为处理器，可以是 AttentionProcessor 类或其字典形式
        def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
            r"""
            设置用于计算注意力的处理器。
    
            参数：
                processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                    实例化的处理器类或处理器类的字典，将作为 **所有** `Attention` 层的处理器。
    
                    如果 `processor` 是一个字典，键需要定义相应交叉注意力处理器的路径。这在设置可训练注意力处理器时强烈推荐。
    
            """
            # 获取当前注意力处理器的数量
            count = len(self.attn_processors.keys())
    
            # 如果传入的是字典且其长度与注意力层的数量不匹配，则抛出错误
            if isinstance(processor, dict) and len(processor) != count:
                raise ValueError(
                    f"传入了处理器字典，但处理器的数量 {len(processor)} 与"
                    f" 注意力层的数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
                )
    
            # 定义递归设置注意力处理器的方法
            def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
                # 如果模块有设置处理器的方法
                if hasattr(module, "set_processor"):
                    # 如果处理器不是字典，则直接设置
                    if not isinstance(processor, dict):
                        module.set_processor(processor)
                    else:
                        # 从字典中获取对应的处理器并设置
                        module.set_processor(processor.pop(f"{name}.processor"))
    
                # 遍历模块的所有子模块
                for sub_name, child in module.named_children():
                    # 递归调用处理子模块
                    fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
            # 遍历当前对象的所有子模块
            for name, module in self.named_children():
                # 递归设置每个子模块的处理器
                fn_recursive_attn_processor(name, module, processor)
    
        # 定义设置默认注意力处理器的方法
        def set_default_attn_processor(self):
            """
            禁用自定义注意力处理器，并设置默认的注意力实现。
            """
            # 调用设置注意力处理器的方法，使用默认的 AttnProcessor 实例
            self.set_attn_processor(AttnProcessor())
    
        # 定义设置梯度检查点的方法
        def _set_gradient_checkpointing(self, module, value=False):
            # 如果模块有梯度检查点的属性
            if hasattr(module, "gradient_checkpointing"):
                # 设置该属性为指定的值
                module.gradient_checkpointing = value
    # 定义前向传播函数，接收样本、时间步以及可选的编码器隐藏状态和注意力掩码
    def forward(self, sample, timestep, encoder_hidden_states=None, encoder_attention_mask=None, return_dict=True):
        # 如果存在编码器注意力掩码，则进行调整以适应后续计算
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            # 增加一个维度，以便后续处理
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
    
        # 检查时间步是否为张量类型
        if not torch.is_tensor(timestep):
            # 根据时间步类型确定数据类型
            dtype = torch.float32 if isinstance(timestep, float) else torch.int32
            # 将时间步转换为张量并指定设备
            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)
        # 如果时间步为标量，则扩展为一维张量
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
    
        # 扩展时间步到与批量维度兼容的形状
        timestep = timestep.expand(sample.shape[0])
        # 通过时间投影获取时间嵌入输入并转换为样本的数据类型
        time_embed_input = self.time_proj(timestep).to(sample.dtype)
        # 获取时间嵌入
        time_embed = self.time_embedding(time_embed_input)
    
        # 对编码器隐藏状态进行线性变换
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
    
        # 如果存在编码器隐藏状态，则将时间嵌入与隐藏状态结合
        if encoder_hidden_states is not None:
            time_embed = self.add_time_condition(time_embed, encoder_hidden_states, encoder_attention_mask)
    
        # 初始化隐藏状态列表
        hidden_states = []
        # 对输入样本进行初步卷积处理
        sample = self.conv_in(sample)
        # 遍历下采样块
        for level, down_sample in enumerate(self.down_blocks):
            # 通过下采样块处理样本
            sample = down_sample(sample, time_embed, encoder_hidden_states, encoder_attention_mask)
            # 如果不是最后一个层级，记录当前样本状态
            if level != self.num_levels - 1:
                hidden_states.append(sample)
    
        # 遍历上采样块
        for level, up_sample in enumerate(self.up_blocks):
            # 如果不是第一个层级，则拼接当前样本与之前的隐藏状态
            if level != 0:
                sample = torch.cat([sample, hidden_states.pop()], dim=1)
            # 通过上采样块处理样本
            sample = up_sample(sample, time_embed, encoder_hidden_states, encoder_attention_mask)
    
        # 进行输出卷积规范化
        sample = self.conv_norm_out(sample)
        # 进行输出激活
        sample = self.conv_act_out(sample)
        # 进行最终输出卷积
        sample = self.conv_out(sample)
    
        # 根据返回标志返回相应的结果
        if not return_dict:
            return (sample,)
        # 返回结果对象
        return Kandinsky3UNetOutput(sample=sample)
# 定义 Kandinsky3UpSampleBlock 类，继承自 nn.Module
class Kandinsky3UpSampleBlock(nn.Module):
    # 初始化方法，设置各参数
    def __init__(
        self,
        in_channels,  # 输入通道数
        cat_dim,  # 拼接维度
        out_channels,  # 输出通道数
        time_embed_dim,  # 时间嵌入维度
        context_dim=None,  # 上下文维度，可选
        num_blocks=3,  # 块的数量
        groups=32,  # 分组数
        head_dim=64,  # 头维度
        expansion_ratio=4,  # 扩展比例
        compression_ratio=2,  # 压缩比例
        up_sample=True,  # 是否上采样
        self_attention=True,  # 是否使用自注意力
    ):
        # 调用父类初始化方法
        super().__init__()
        # 设置上采样分辨率
        up_resolutions = [[None, True if up_sample else None, None, None]] + [[None] * 4] * (num_blocks - 1)
        # 设置隐藏通道数
        hidden_channels = (
            [(in_channels + cat_dim, in_channels)]  # 第一层的通道
            + [(in_channels, in_channels)] * (num_blocks - 2)  # 中间层的通道
            + [(in_channels, out_channels)]  # 最后一层的通道
        )
        attentions = []  # 用于存储注意力块
        resnets_in = []  # 用于存储输入 ResNet 块
        resnets_out = []  # 用于存储输出 ResNet 块

        # 设置自注意力和上下文维度
        self.self_attention = self_attention
        self.context_dim = context_dim

        # 如果使用自注意力，添加注意力块
        if self_attention:
            attentions.append(
                Kandinsky3AttentionBlock(out_channels, time_embed_dim, None, groups, head_dim, expansion_ratio)
            )
        else:
            attentions.append(nn.Identity())  # 否则添加身份映射

        # 遍历隐藏通道和上采样分辨率
        for (in_channel, out_channel), up_resolution in zip(hidden_channels, up_resolutions):
            # 添加输入 ResNet 块
            resnets_in.append(
                Kandinsky3ResNetBlock(in_channel, in_channel, time_embed_dim, groups, compression_ratio, up_resolution)
            )

            # 如果上下文维度不为 None，添加注意力块
            if context_dim is not None:
                attentions.append(
                    Kandinsky3AttentionBlock(
                        in_channel, time_embed_dim, context_dim, groups, head_dim, expansion_ratio
                    )
                )
            else:
                attentions.append(nn.Identity())  # 否则添加身份映射

            # 添加输出 ResNet 块
            resnets_out.append(
                Kandinsky3ResNetBlock(in_channel, out_channel, time_embed_dim, groups, compression_ratio)
            )

        # 将注意力块和 ResNet 块转换为模块列表
        self.attentions = nn.ModuleList(attentions)
        self.resnets_in = nn.ModuleList(resnets_in)
        self.resnets_out = nn.ModuleList(resnets_out)

    # 前向传播方法
    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        # 遍历注意力块和 ResNet 块进行前向计算
        for attention, resnet_in, resnet_out in zip(self.attentions[1:], self.resnets_in, self.resnets_out):
            x = resnet_in(x, time_embed)  # 输入经过 ResNet 块
            if self.context_dim is not None:  # 如果上下文维度存在
                x = attention(x, time_embed, context, context_mask, image_mask)  # 应用注意力块
            x = resnet_out(x, time_embed)  # 输出经过 ResNet 块

        # 如果使用自注意力，应用首个注意力块
        if self.self_attention:
            x = self.attentions[0](x, time_embed, image_mask=image_mask)
        return x  # 返回处理后的结果


# 定义 Kandinsky3DownSampleBlock 类，继承自 nn.Module
class Kandinsky3DownSampleBlock(nn.Module):
    # 初始化方法，设置各参数
    def __init__(
        self,
        in_channels,  # 输入通道数
        out_channels,  # 输出通道数
        time_embed_dim,  # 时间嵌入维度
        context_dim=None,  # 上下文维度，可选
        num_blocks=3,  # 块的数量
        groups=32,  # 分组数
        head_dim=64,  # 头维度
        expansion_ratio=4,  # 扩展比例
        compression_ratio=2,  # 压缩比例
        down_sample=True,  # 是否下采样
        self_attention=True,  # 是否使用自注意力
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化注意力模块列表
        attentions = []
        # 初始化输入残差块列表
        resnets_in = []
        # 初始化输出残差块列表
        resnets_out = []

        # 保存自注意力标志
        self.self_attention = self_attention
        # 保存上下文维度
        self.context_dim = context_dim

        # 如果启用自注意力
        if self_attention:
            # 添加 Kandinsky3AttentionBlock 到注意力列表
            attentions.append(
                Kandinsky3AttentionBlock(in_channels, time_embed_dim, None, groups, head_dim, expansion_ratio)
            )
        else:
            # 否则添加身份层（不改变输入）
            attentions.append(nn.Identity())

        # 生成上采样分辨率列表
        up_resolutions = [[None] * 4] * (num_blocks - 1) + [[None, None, False if down_sample else None, None]]
        # 生成隐藏通道的元组列表
        hidden_channels = [(in_channels, out_channels)] + [(out_channels, out_channels)] * (num_blocks - 1)
        # 遍历隐藏通道和上采样分辨率
        for (in_channel, out_channel), up_resolution in zip(hidden_channels, up_resolutions):
            # 添加输入残差块到列表
            resnets_in.append(
                Kandinsky3ResNetBlock(in_channel, out_channel, time_embed_dim, groups, compression_ratio)
            )

            # 如果上下文维度不为 None
            if context_dim is not None:
                # 添加 Kandinsky3AttentionBlock 到注意力列表
                attentions.append(
                    Kandinsky3AttentionBlock(
                        out_channel, time_embed_dim, context_dim, groups, head_dim, expansion_ratio
                    )
                )
            else:
                # 否则添加身份层（不改变输入）
                attentions.append(nn.Identity())

            # 添加输出残差块到列表
            resnets_out.append(
                Kandinsky3ResNetBlock(
                    out_channel, out_channel, time_embed_dim, groups, compression_ratio, up_resolution
                )
            )

        # 将注意力模块列表转换为 nn.ModuleList 以便管理
        self.attentions = nn.ModuleList(attentions)
        # 将输入残差块列表转换为 nn.ModuleList 以便管理
        self.resnets_in = nn.ModuleList(resnets_in)
        # 将输出残差块列表转换为 nn.ModuleList 以便管理
        self.resnets_out = nn.ModuleList(resnets_out)

    # 定义前向传播方法
    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        # 如果启用自注意力
        if self.self_attention:
            # 使用第一个注意力模块处理输入
            x = self.attentions[0](x, time_embed, image_mask=image_mask)

        # 遍历剩余的注意力模块、输入和输出残差块
        for attention, resnet_in, resnet_out in zip(self.attentions[1:], self.resnets_in, self.resnets_out):
            # 通过输入残差块处理输入
            x = resnet_in(x, time_embed)
            # 如果上下文维度不为 None
            if self.context_dim is not None:
                # 使用当前注意力模块处理输入
                x = attention(x, time_embed, context, context_mask, image_mask)
            # 通过输出残差块处理输入
            x = resnet_out(x, time_embed)
        # 返回处理后的输出
        return x
# 定义 Kandinsky3ConditionalGroupNorm 类，继承自 nn.Module
class Kandinsky3ConditionalGroupNorm(nn.Module):
    # 初始化方法，设置分组数、标准化形状和上下文维度
    def __init__(self, groups, normalized_shape, context_dim):
        # 调用父类构造函数
        super().__init__()
        # 创建分组归一化层，不使用仿射变换
        self.norm = nn.GroupNorm(groups, normalized_shape, affine=False)
        # 定义上下文多层感知机，包含 SiLU 激活和线性层
        self.context_mlp = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, 2 * normalized_shape))
        # 将线性层的权重初始化为零
        self.context_mlp[1].weight.data.zero_()
        # 将线性层的偏置初始化为零
        self.context_mlp[1].bias.data.zero_()

    # 前向传播方法，接收输入和上下文
    def forward(self, x, context):
        # 通过上下文多层感知机处理上下文
        context = self.context_mlp(context)

        # 为了匹配输入的维度，逐层扩展上下文
        for _ in range(len(x.shape[2:])):
            context = context.unsqueeze(-1)

        # 将上下文分割为缩放和偏移量
        scale, shift = context.chunk(2, dim=1)
        # 应用归一化并进行缩放和偏移
        x = self.norm(x) * (scale + 1.0) + shift
        # 返回处理后的输入
        return x


# 定义 Kandinsky3Block 类，继承自 nn.Module
class Kandinsky3Block(nn.Module):
    # 初始化方法，设置输入通道、输出通道、时间嵌入维度等参数
    def __init__(self, in_channels, out_channels, time_embed_dim, kernel_size=3, norm_groups=32, up_resolution=None):
        # 调用父类构造函数
        super().__init__()
        # 创建条件分组归一化层
        self.group_norm = Kandinsky3ConditionalGroupNorm(norm_groups, in_channels, time_embed_dim)
        # 定义 SiLU 激活函数
        self.activation = nn.SiLU()
        # 如果需要上采样，使用转置卷积进行上采样
        if up_resolution is not None and up_resolution:
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            # 否则使用恒等映射
            self.up_sample = nn.Identity()

        # 根据卷积核大小确定填充
        padding = int(kernel_size > 1)
        # 定义卷积投影层
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        # 如果不需要上采样，定义下采样卷积层
        if up_resolution is not None and not up_resolution:
            self.down_sample = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        else:
            # 否则使用恒等映射
            self.down_sample = nn.Identity()

    # 前向传播方法，接收输入和时间嵌入
    def forward(self, x, time_embed):
        # 通过条件分组归一化处理输入
        x = self.group_norm(x, time_embed)
        # 应用激活函数
        x = self.activation(x)
        # 进行上采样
        x = self.up_sample(x)
        # 通过卷积投影层处理输入
        x = self.projection(x)
        # 进行下采样
        x = self.down_sample(x)
        # 返回处理后的输出
        return x


# 定义 Kandinsky3ResNetBlock 类，继承自 nn.Module
class Kandinsky3ResNetBlock(nn.Module):
    # 初始化方法，设置输入通道、输出通道、时间嵌入维度等参数
    def __init__(
        self, in_channels, out_channels, time_embed_dim, norm_groups=32, compression_ratio=2, up_resolutions=4 * [None]
    # 初始化父类
        ):
            super().__init__()
            # 定义卷积核的大小
            kernel_sizes = [1, 3, 3, 1]
            # 计算隐藏通道数
            hidden_channel = max(in_channels, out_channels) // compression_ratio
            # 构建隐藏通道的元组列表
            hidden_channels = (
                [(in_channels, hidden_channel)] + [(hidden_channel, hidden_channel)] * 2 + [(hidden_channel, out_channels)]
            )
            # 创建包含多个 Kandinsky3Block 的模块列表
            self.resnet_blocks = nn.ModuleList(
                [
                    Kandinsky3Block(in_channel, out_channel, time_embed_dim, kernel_size, norm_groups, up_resolution)
                    # 将隐藏通道、卷积核大小和上采样分辨率组合在一起
                    for (in_channel, out_channel), kernel_size, up_resolution in zip(
                        hidden_channels, kernel_sizes, up_resolutions
                    )
                ]
            )
            # 定义上采样的快捷连接
            self.shortcut_up_sample = (
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
                # 如果存在上采样分辨率，则使用反卷积；否则使用恒等映射
                if True in up_resolutions
                else nn.Identity()
            )
            # 定义通道数不同时的投影连接
            self.shortcut_projection = (
                nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
            )
            # 定义下采样的快捷连接
            self.shortcut_down_sample = (
                nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
                # 如果存在下采样分辨率，则使用卷积；否则使用恒等映射
                if False in up_resolutions
                else nn.Identity()
            )
    
        # 前向传播方法
        def forward(self, x, time_embed):
            # 初始化输出为输入
            out = x
            # 依次通过每个 ResNet 块
            for resnet_block in self.resnet_blocks:
                out = resnet_block(out, time_embed)
    
            # 上采样输入
            x = self.shortcut_up_sample(x)
            # 投影输入到输出通道
            x = self.shortcut_projection(x)
            # 下采样输入
            x = self.shortcut_down_sample(x)
            # 将输出与处理后的输入相加
            x = x + out
            # 返回最终输出
            return x
# 定义 Kandinsky3AttentionPooling 类，继承自 nn.Module
class Kandinsky3AttentionPooling(nn.Module):
    # 初始化方法，接受通道数、上下文维度和头维度
    def __init__(self, num_channels, context_dim, head_dim=64):
        # 调用父类构造函数
        super().__init__()
        # 创建注意力机制对象，指定输入和输出维度及其他参数
        self.attention = Attention(
            context_dim,
            context_dim,
            dim_head=head_dim,
            out_dim=num_channels,
            out_bias=False,
        )

    # 前向传播方法
    def forward(self, x, context, context_mask=None):
        # 将上下文掩码转换为与上下文相同的数据类型
        context_mask = context_mask.to(dtype=context.dtype)
        # 使用注意力机制计算上下文与其平均值的加权和
        context = self.attention(context.mean(dim=1, keepdim=True), context, context_mask)
        # 返回输入与上下文的和
        return x + context.squeeze(1)


# 定义 Kandinsky3AttentionBlock 类，继承自 nn.Module
class Kandinsky3AttentionBlock(nn.Module):
    # 初始化方法，接受多种参数
    def __init__(self, num_channels, time_embed_dim, context_dim=None, norm_groups=32, head_dim=64, expansion_ratio=4):
        # 调用父类构造函数
        super().__init__()
        # 创建条件组归一化对象，用于输入规范化
        self.in_norm = Kandinsky3ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        # 创建注意力机制对象，指定输入和输出维度及其他参数
        self.attention = Attention(
            num_channels,
            context_dim or num_channels,
            dim_head=head_dim,
            out_dim=num_channels,
            out_bias=False,
        )

        # 计算隐藏通道数，作为扩展比和通道数的乘积
        hidden_channels = expansion_ratio * num_channels
        # 创建条件组归一化对象，用于输出规范化
        self.out_norm = Kandinsky3ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        # 定义前馈网络，包含两个卷积层和激活函数
        self.feed_forward = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
        )

    # 前向传播方法
    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        # 获取输入的高度和宽度
        height, width = x.shape[-2:]
        # 对输入进行归一化处理
        out = self.in_norm(x, time_embed)
        # 将输出重塑为适合注意力机制的形状
        out = out.reshape(x.shape[0], -1, height * width).permute(0, 2, 1)
        # 如果没有上下文，则使用当前的输出作为上下文
        context = context if context is not None else out
        # 如果存在上下文掩码，转换为与上下文相同的数据类型
        if context_mask is not None:
            context_mask = context_mask.to(dtype=context.dtype)

        # 使用注意力机制处理输出和上下文
        out = self.attention(out, context, context_mask)
        # 重塑输出为原始输入形状
        out = out.permute(0, 2, 1).unsqueeze(-1).reshape(out.shape[0], -1, height, width)
        # 将处理后的输出与原输入相加
        x = x + out

        # 对相加后的结果进行输出归一化
        out = self.out_norm(x, time_embed)
        # 通过前馈网络处理归一化输出
        out = self.feed_forward(out)
        # 将处理后的输出与相加后的输入相加
        x = x + out
        # 返回最终输出
        return x
```