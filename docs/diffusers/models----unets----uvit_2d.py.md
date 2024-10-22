# `.\diffusers\models\unets\uvit_2d.py`

```py
# coding=utf-8  # 指定文件编码为 UTF-8
# Copyright 2024 The HuggingFace Inc. team.  # 版权声明，标识文件归 HuggingFace Inc. 团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 表明文件受 Apache 2.0 许可证保护
# you may not use this file except in compliance with the License.  # 指明用户必须遵循许可证的使用条款
# You may obtain a copy of the License at  # 提供获取许可证的方式
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证的具体链接
#
# Unless required by applicable law or agreed to in writing, software  # 表明软件是按 "AS IS" 基础分发
# distributed under the License is distributed on an "AS IS" BASIS,  # 不提供任何明示或暗示的担保
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 用户须自行承担使用风险
# See the License for the specific language governing permissions and  # 引导用户查看许可证了解权限
# limitations under the License.  # 许可证的限制条款

from typing import Dict, Union  # 从 typing 模块导入字典和联合类型
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
from torch import nn  # 从 PyTorch 导入神经网络模块
from torch.utils.checkpoint import checkpoint  # 从 PyTorch 导入检查点功能，用于节省内存

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入混合类和注册配置函数
from ...loaders import PeftAdapterMixin  # 从加载器导入适配器混合类
from ..attention import BasicTransformerBlock, SkipFFTransformerBlock  # 从注意力模块导入基本变换块和跳过前馈变换块
from ..attention_processor import (  # 从注意力处理器导入相关组件
    ADDED_KV_ATTENTION_PROCESSORS,  # 导入增加的键值注意力处理器
    CROSS_ATTENTION_PROCESSORS,  # 导入交叉注意力处理器
    AttentionProcessor,  # 导入注意力处理器类
    AttnAddedKVProcessor,  # 导入增加键值注意力处理器类
    AttnProcessor,  # 导入注意力处理器基类
)
from ..embeddings import TimestepEmbedding, get_timestep_embedding  # 从嵌入模块导入时间步嵌入及其获取函数
from ..modeling_utils import ModelMixin  # 从建模工具导入模型混合类
from ..normalization import GlobalResponseNorm, RMSNorm  # 从归一化模块导入全局响应归一化和 RMS 归一化
from ..resnet import Downsample2D, Upsample2D  # 从 ResNet 模块导入二维下采样和上采样

class UVit2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):  # 定义 UVit2DModel 类，继承多个混合类
    _supports_gradient_checkpointing = True  # 声明支持梯度检查点，节省内存

    @register_to_config  # 注册到配置的装饰器
    def __init__(  # 初始化方法
        self,  # 实例自身
        # global config  # 全局配置说明
        hidden_size: int = 1024,  # 隐藏层大小，默认为 1024
        use_bias: bool = False,  # 是否使用偏置，默认为 False
        hidden_dropout: float = 0.0,  # 隐藏层 dropout 概率，默认为 0
        # conditioning dimensions  # 条件维度说明
        cond_embed_dim: int = 768,  # 条件嵌入维度，默认为 768
        micro_cond_encode_dim: int = 256,  # 微条件编码维度，默认为 256
        micro_cond_embed_dim: int = 1280,  # 微条件嵌入维度，默认为 1280
        encoder_hidden_size: int = 768,  # 编码器隐藏层大小，默认为 768
        # num tokens  # 令牌数量说明
        vocab_size: int = 8256,  # 词汇表大小，默认为 8256（包括掩码令牌）
        codebook_size: int = 8192,  # 代码本大小，默认为 8192
        # `UVit2DConvEmbed`  # UVit2D 卷积嵌入说明
        in_channels: int = 768,  # 输入通道数，默认为 768
        block_out_channels: int = 768,  # 块输出通道数，默认为 768
        num_res_blocks: int = 3,  # 残差块数量，默认为 3
        downsample: bool = False,  # 是否进行下采样，默认为 False
        upsample: bool = False,  # 是否进行上采样，默认为 False
        block_num_heads: int = 12,  # 块头数，默认为 12
        # `TransformerLayer`  # 变换层说明
        num_hidden_layers: int = 22,  # 隐藏层数量，默认为 22
        num_attention_heads: int = 16,  # 注意力头数量，默认为 16
        # `Attention`  # 注意力说明
        attention_dropout: float = 0.0,  # 注意力层 dropout 概率，默认为 0
        # `FeedForward`  # 前馈层说明
        intermediate_size: int = 2816,  # 前馈层中间大小，默认为 2816
        # `Norm`  # 归一化说明
        layer_norm_eps: float = 1e-6,  # 层归一化的 epsilon 值，默认为 1e-6
        ln_elementwise_affine: bool = True,  # 是否使用元素级仿射，默认为 True
        sample_size: int = 64,  # 采样大小，默认为 64
    # 初始化父类
        ):
            super().__init__()
    
            # 创建一个线性层，用于编码器的输出投影
            self.encoder_proj = nn.Linear(encoder_hidden_size, hidden_size, bias=use_bias)
            # 创建 RMSNorm 层，对编码器输出进行层归一化
            self.encoder_proj_layer_norm = RMSNorm(hidden_size, layer_norm_eps, ln_elementwise_affine)
    
            # 初始化 UVit2DConvEmbed，进行输入通道到嵌入的转换
            self.embed = UVit2DConvEmbed(
                in_channels, block_out_channels, vocab_size, ln_elementwise_affine, layer_norm_eps, use_bias
            )
    
            # 创建时间步嵌入层，用于条件输入的嵌入
            self.cond_embed = TimestepEmbedding(
                micro_cond_embed_dim + cond_embed_dim, hidden_size, sample_proj_bias=use_bias
            )
    
            # 创建下采样块，包含多个残差块
            self.down_block = UVitBlock(
                block_out_channels,
                num_res_blocks,
                hidden_size,
                hidden_dropout,
                ln_elementwise_affine,
                layer_norm_eps,
                use_bias,
                block_num_heads,
                attention_dropout,
                downsample,
                False,
            )
    
            # 创建 RMSNorm 层，用于隐藏状态的归一化
            self.project_to_hidden_norm = RMSNorm(block_out_channels, layer_norm_eps, ln_elementwise_affine)
            # 创建线性层，将投影结果转换为隐藏层大小
            self.project_to_hidden = nn.Linear(block_out_channels, hidden_size, bias=use_bias)
    
            # 创建一个模块列表，包含多个基本的 Transformer 块
            self.transformer_layers = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=hidden_size,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=hidden_size // num_attention_heads,
                        dropout=hidden_dropout,
                        cross_attention_dim=hidden_size,
                        attention_bias=use_bias,
                        norm_type="ada_norm_continuous",
                        ada_norm_continous_conditioning_embedding_dim=hidden_size,
                        norm_elementwise_affine=ln_elementwise_affine,
                        norm_eps=layer_norm_eps,
                        ada_norm_bias=use_bias,
                        ff_inner_dim=intermediate_size,
                        ff_bias=use_bias,
                        attention_out_bias=use_bias,
                    )
                    for _ in range(num_hidden_layers)  # 遍历生成指定数量的 Transformer 块
                ]
            )
    
            # 创建 RMSNorm 层，用于隐藏状态的归一化
            self.project_from_hidden_norm = RMSNorm(hidden_size, layer_norm_eps, ln_elementwise_affine)
            # 创建线性层，将隐藏层转换为块输出通道
            self.project_from_hidden = nn.Linear(hidden_size, block_out_channels, bias=use_bias)
    
            # 创建上采样块，包含多个残差块
            self.up_block = UVitBlock(
                block_out_channels,
                num_res_blocks,
                hidden_size,
                hidden_dropout,
                ln_elementwise_affine,
                layer_norm_eps,
                use_bias,
                block_num_heads,
                attention_dropout,
                downsample=False,
                upsample=upsample,
            )
    
            # 创建卷积 MLM 层，用于生成模型的最终输出
            self.mlm_layer = ConvMlmLayer(
                block_out_channels, in_channels, use_bias, ln_elementwise_affine, layer_norm_eps, codebook_size
            )
    
            # 初始化梯度检查点标志为 False
            self.gradient_checkpointing = False
    
        # 定义梯度检查点设置函数，默认不启用
        def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
            pass
    # 定义前向传播方法，接受输入 IDs、编码器隐藏状态、池化文本嵌入、微条件及交叉注意力参数
    def forward(self, input_ids, encoder_hidden_states, pooled_text_emb, micro_conds, cross_attention_kwargs=None):
        # 对编码器隐藏状态进行线性变换
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        # 对编码器隐藏状态进行层归一化
        encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states)
    
        # 获取微条件的时间步嵌入，应用特定的配置参数
        micro_cond_embeds = get_timestep_embedding(
            micro_conds.flatten(), self.config.micro_cond_encode_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
    
        # 调整微条件嵌入的形状，匹配输入 ID 的批大小
        micro_cond_embeds = micro_cond_embeds.reshape((input_ids.shape[0], -1))
    
        # 将池化文本嵌入和微条件嵌入在维度1上连接
        pooled_text_emb = torch.cat([pooled_text_emb, micro_cond_embeds], dim=1)
        # 将池化文本嵌入转换为指定的数据类型
        pooled_text_emb = pooled_text_emb.to(dtype=self.dtype)
        # 对池化文本嵌入进行条件嵌入并转换为编码器隐藏状态的数据类型
        pooled_text_emb = self.cond_embed(pooled_text_emb).to(encoder_hidden_states.dtype)
    
        # 获取输入 ID 的嵌入表示
        hidden_states = self.embed(input_ids)
    
        # 将隐藏状态通过下一个模块处理
        hidden_states = self.down_block(
            hidden_states,
            pooled_text_emb=pooled_text_emb,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        )
    
        # 获取隐藏状态的批大小、通道、高度和宽度
        batch_size, channels, height, width = hidden_states.shape
        # 调整隐藏状态的维度顺序并重塑形状
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
    
        # 对隐藏状态进行规范化投影
        hidden_states = self.project_to_hidden_norm(hidden_states)
        # 对隐藏状态进行线性投影
        hidden_states = self.project_to_hidden(hidden_states)
    
        # 遍历每个变换层
        for layer in self.transformer_layers:
            # 如果在训练模式下并启用梯度检查点
            if self.training and self.gradient_checkpointing:
                # 定义一个带检查点的层
                def layer_(*args):
                    return checkpoint(layer, *args)
            else:
                # 否则直接使用层
                layer_ = layer
    
            # 通过当前层处理隐藏状态
            hidden_states = layer_(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs={"pooled_text_emb": pooled_text_emb},
            )
    
        # 对隐藏状态进行规范化投影
        hidden_states = self.project_from_hidden_norm(hidden_states)
        # 对隐藏状态进行线性投影
        hidden_states = self.project_from_hidden(hidden_states)
    
        # 重塑隐藏状态以匹配图像维度并调整维度顺序
        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    
        # 将隐藏状态通过上一个模块处理
        hidden_states = self.up_block(
            hidden_states,
            pooled_text_emb=pooled_text_emb,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        )
    
        # 通过 MLM 层获取最终的 logits
        logits = self.mlm_layer(hidden_states)
    
        # 返回最终的 logits
        return logits
    
    # 定义一个只读属性，可能用于后续处理
    @property
    # 定义返回注意力处理器的字典的方法
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回值:
            `dict` 的注意力处理器: 一个包含模型中所有注意力处理器的字典，以权重名称为索引。
        """
        # 创建一个空字典，用于存储注意力处理器
        processors = {}
    
        # 定义递归函数，用于添加处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 如果模块有获取处理器的方法，则将其添加到字典中
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
    
            # 遍历模块的子模块，递归调用
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
            return processors
    
        # 遍历当前对象的子模块，调用递归函数
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
    
        # 返回所有收集到的处理器
        return processors
    
    # 从 UNet2DConditionModel 复制的方法，用于设置注意力处理器
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。
    
        参数:
            processor (`dict` of `AttentionProcessor` 或 `AttentionProcessor`):
                实例化的处理器类或将被设置为**所有** `Attention` 层的处理器类的字典。
    
                如果 `processor` 是字典，则键需要定义相应的交叉注意力处理器的路径。强烈建议在设置可训练的注意力处理器时使用。
        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())
    
        # 如果传入的是字典且数量不匹配，抛出异常
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器字典，但处理器数量 {len(processor)} 与注意力层数量: {count} 不匹配。请确保传入 {count} 个处理器类。"
            )
    
        # 定义递归函数，用于设置处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块有设置处理器的方法，则设置处理器
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
    
            # 遍历子模块，递归调用设置处理器
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
        # 遍历当前对象的子模块，调用递归设置函数
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)
    
    # 从 UNet2DConditionModel 复制的方法，用于设置默认注意力处理器
    # 定义一个设置默认注意力处理器的方法
    def set_default_attn_processor(self):
        # 文档字符串，说明该方法的功能
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        # 检查所有注意力处理器是否属于新增的 KV 注意力处理器类型
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 如果是，则创建一个新增 KV 注意力处理器实例
            processor = AttnAddedKVProcessor()
        # 检查所有注意力处理器是否属于交叉注意力处理器类型
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 如果是，则创建一个标准注意力处理器实例
            processor = AttnProcessor()
        else:
            # 如果既不是新增 KV 注意力处理器也不是交叉注意力处理器，则抛出错误
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )
    
        # 设置当前实例的注意力处理器为刚创建的处理器
        self.set_attn_processor(processor)
# 定义一个二维卷积嵌入类，继承自 nn.Module
class UVit2DConvEmbed(nn.Module):
    # 初始化方法，接收输入通道数、块输出通道数、词汇表大小等参数
    def __init__(self, in_channels, block_out_channels, vocab_size, elementwise_affine, eps, bias):
        # 调用父类构造函数
        super().__init__()
        # 创建嵌入层，将词汇表大小映射到输入通道数
        self.embeddings = nn.Embedding(vocab_size, in_channels)
        # 创建 RMSNorm 层，用于归一化嵌入，支持可选的元素级仿射变换
        self.layer_norm = RMSNorm(in_channels, eps, elementwise_affine)
        # 创建 2D 卷积层，使用指定的输出通道数和偏置选项
        self.conv = nn.Conv2d(in_channels, block_out_channels, kernel_size=1, bias=bias)

    # 前向传播方法，定义输入如何经过该层处理
    def forward(self, input_ids):
        # 根据输入 ID 获取对应的嵌入
        embeddings = self.embeddings(input_ids)
        # 对嵌入进行层归一化处理
        embeddings = self.layer_norm(embeddings)
        # 调整嵌入的维度顺序，以适应卷积层的输入格式
        embeddings = embeddings.permute(0, 3, 1, 2)
        # 通过卷积层处理嵌入
        embeddings = self.conv(embeddings)
        # 返回处理后的嵌入
        return embeddings


# 定义一个 UVit 块类，继承自 nn.Module
class UVitBlock(nn.Module):
    # 初始化方法，接收多个配置参数
    def __init__(
        self,
        channels,
        num_res_blocks: int,
        hidden_size,
        hidden_dropout,
        ln_elementwise_affine,
        layer_norm_eps,
        use_bias,
        block_num_heads,
        attention_dropout,
        downsample: bool,
        upsample: bool,
    ):
        # 调用父类构造函数
        super().__init__()

        # 如果需要下采样，初始化下采样层
        if downsample:
            self.downsample = Downsample2D(
                channels,
                use_conv=True,
                padding=0,
                name="Conv2d_0",
                kernel_size=2,
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
            )
        else:
            # 否则将下采样层设为 None
            self.downsample = None

        # 创建残差块列表，包含指定数量的卷积块
        self.res_blocks = nn.ModuleList(
            [
                ConvNextBlock(
                    channels,
                    layer_norm_eps,
                    ln_elementwise_affine,
                    use_bias,
                    hidden_dropout,
                    hidden_size,
                )
                for i in range(num_res_blocks)
            ]
        )

        # 创建注意力块列表，包含指定数量的跳跃前馈变换块
        self.attention_blocks = nn.ModuleList(
            [
                SkipFFTransformerBlock(
                    channels,
                    block_num_heads,
                    channels // block_num_heads,
                    hidden_size,
                    use_bias,
                    attention_dropout,
                    channels,
                    attention_bias=use_bias,
                    attention_out_bias=use_bias,
                )
                for _ in range(num_res_blocks)
            ]
        )

        # 如果需要上采样，初始化上采样层
        if upsample:
            self.upsample = Upsample2D(
                channels,
                use_conv_transpose=True,
                kernel_size=2,
                padding=0,
                name="conv",
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
                interpolate=False,
            )
        else:
            # 否则将上采样层设为 None
            self.upsample = None
    # 定义前向传播函数，接收输入 x、池化文本嵌入、编码器隐藏状态和交叉注意力参数
    def forward(self, x, pooled_text_emb, encoder_hidden_states, cross_attention_kwargs):
        # 如果存在下采样层，则对输入进行下采样
        if self.downsample is not None:
            x = self.downsample(x)
    
        # 遍历残差块和注意力块的组合
        for res_block, attention_block in zip(self.res_blocks, self.attention_blocks):
            # 将输入通过残差块进行处理
            x = res_block(x, pooled_text_emb)
    
            # 获取当前输出的批量大小、通道数、高度和宽度
            batch_size, channels, height, width = x.shape
            # 将输出形状调整为 (批量大小, 通道数, 高度 * 宽度)，然后转置
            x = x.view(batch_size, channels, height * width).permute(0, 2, 1)
            # 将处理后的输入通过注意力块，并传递编码器隐藏状态和交叉注意力参数
            x = attention_block(
                x, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs
            )
            # 将输出转置并恢复为 (批量大小, 通道数, 高度, 宽度) 的形状
            x = x.permute(0, 2, 1).view(batch_size, channels, height, width)
    
        # 如果存在上采样层，则对输出进行上采样
        if self.upsample is not None:
            x = self.upsample(x)
    
        # 返回最终的输出
        return x
# 定义一个卷积块的类，继承自 nn.Module
class ConvNextBlock(nn.Module):
    # 初始化方法，接受多个参数以配置卷积块
    def __init__(
        self, channels, layer_norm_eps, ln_elementwise_affine, use_bias, hidden_dropout, hidden_size, res_ffn_factor=4
    ):
        # 调用父类初始化方法
        super().__init__()
        # 定义深度可分离卷积层，通道数为 channels，卷积核大小为 3，使用 padding 保持输入输出相同大小
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,  # 进行深度卷积
            bias=use_bias,  # 是否使用偏置
        )
        # 定义 RMSNorm 层，用于规范化，接受通道数和层归一化的 epsilon
        self.norm = RMSNorm(channels, layer_norm_eps, ln_elementwise_affine)
        # 定义第一个线性层，将通道数映射到一个更大的维度
        self.channelwise_linear_1 = nn.Linear(channels, int(channels * res_ffn_factor), bias=use_bias)
        # 定义激活函数，使用 GELU
        self.channelwise_act = nn.GELU()
        # 定义全局响应规范化层，输入维度为扩展后的通道数
        self.channelwise_norm = GlobalResponseNorm(int(channels * res_ffn_factor))
        # 定义第二个线性层，将大维度映射回原始通道数
        self.channelwise_linear_2 = nn.Linear(int(channels * res_ffn_factor), channels, bias=use_bias)
        # 定义 dropout 层，应用于隐藏层，使用给定的丢弃率
        self.channelwise_dropout = nn.Dropout(hidden_dropout)
        # 定义条件嵌入映射层，将隐层大小映射到两倍的通道数
        self.cond_embeds_mapper = nn.Linear(hidden_size, channels * 2, use_bias)

    # 前向传播方法，定义数据如何流经网络
    def forward(self, x, cond_embeds):
        # 保存输入以用于残差连接
        x_res = x

        # 通过深度卷积层处理输入
        x = self.depthwise(x)

        # 调整张量维度，将通道维移至最后
        x = x.permute(0, 2, 3, 1)
        # 对张量进行归一化处理
        x = self.norm(x)

        # 通过第一个线性层
        x = self.channelwise_linear_1(x)
        # 应用激活函数
        x = self.channelwise_act(x)
        # 进行规范化处理
        x = self.channelwise_norm(x)
        # 通过第二个线性层映射回通道数
        x = self.channelwise_linear_2(x)
        # 应用 dropout
        x = self.channelwise_dropout(x)

        # 再次调整张量维度，恢复通道维的位置
        x = x.permute(0, 3, 1, 2)

        # 添加残差连接，将输入与输出相加
        x = x + x_res

        # 通过条件嵌入映射生成缩放和偏移值，使用 SiLU 激活
        scale, shift = self.cond_embeds_mapper(F.silu(cond_embeds)).chunk(2, dim=1)
        # 应用缩放和偏移调整输出
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        # 返回处理后的输出
        return x


# 定义一个卷积 MLM 层的类，继承自 nn.Module
class ConvMlmLayer(nn.Module):
    # 初始化方法，接受多个参数以配置卷积 MLM 层
    def __init__(
        self,
        block_out_channels: int,
        in_channels: int,
        use_bias: bool,
        ln_elementwise_affine: bool,
        layer_norm_eps: float,
        codebook_size: int,
    ):
        # 调用父类初始化方法
        super().__init__()
        # 定义第一个卷积层，将 block_out_channels 映射到 in_channels
        self.conv1 = nn.Conv2d(block_out_channels, in_channels, kernel_size=1, bias=use_bias)
        # 定义 RMSNorm 层，用于规范化，接受输入通道数和层归一化的 epsilon
        self.layer_norm = RMSNorm(in_channels, layer_norm_eps, ln_elementwise_affine)
        # 定义第二个卷积层，将 in_channels 映射到 codebook_size
        self.conv2 = nn.Conv2d(in_channels, codebook_size, kernel_size=1, bias=use_bias)

    # 前向传播方法，定义数据如何流经网络
    def forward(self, hidden_states):
        # 通过第一个卷积层处理隐藏状态
        hidden_states = self.conv1(hidden_states)
        # 对输出进行规范化处理，调整维度顺序
        hidden_states = self.layer_norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # 通过第二个卷积层生成 logits
        logits = self.conv2(hidden_states)
        # 返回 logits
        return logits
```