# `.\diffusers\pipelines\latent_diffusion\pipeline_latent_diffusion.py`

```py
# 版权声明，标明版权信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 授权信息，声明文件的使用条件
# Licensed under the Apache License, Version 2.0 (the "License");
# 说明用户使用该文件必须遵循的许可证
# you may not use this file except in compliance with the License.
# 用户可在此处获取许可证的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 免责声明，声明软件以"原样"方式分发，不提供任何担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 inspect 模块，用于获取对象的活跃信息
import inspect
# 从 typing 模块导入类型注解
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的检查点工具
import torch.utils.checkpoint
# 从 transformers 库导入配置、预训练模型和预训练分词器
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
# 从 transformers 导入激活函数映射
from transformers.activations import ACT2FN
# 从 transformers 导入基础模型输出格式
from transformers.modeling_outputs import BaseModelOutput
# 从 transformers 导入日志工具
from transformers.utils import logging

# 导入特定模型类
from ...models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, VQModel
# 导入调度器类
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
# 导入随机张量工具
from ...utils.torch_utils import randn_tensor
# 导入扩散管道和图像输出类
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


# 定义用于文本到图像生成的管道类，继承自 DiffusionPipeline
class LDMTextToImagePipeline(DiffusionPipeline):
    r"""
    使用潜在扩散进行文本到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]。查看超类文档以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    参数：
        vqvae ([`VQModel`]):
            向量量化（VQ）模型，用于将图像编码和解码为潜在表示。
        bert ([`LDMBertModel`]):
            基于 [`~transformers.BERT`] 的文本编码模型。
        tokenizer ([`~transformers.BertTokenizer`]):
            用于对文本进行分词的 `BertTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于去噪编码图像潜在表示的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            用于与 `unet` 结合使用的调度器，用于去噪编码图像潜在表示。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`]。
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "bert->unet->vqvae"

    # 初始化方法，接受多个参数并注册模块
    def __init__(
        self,
        vqvae: Union[VQModel, AutoencoderKL],
        bert: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        unet: Union[UNet2DModel, UNet2DConditionModel],
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        # 调用父类构造函数
        super().__init__()
        # 注册所需模块
        self.register_modules(vqvae=vqvae, bert=bert, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
        # 计算 VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)

    # 禁用梯度计算，优化内存使用
    @torch.no_grad()
    # 定义一个可调用的方法，允许通过实例直接调用
        def __call__(
            # 输入提示，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]],
            # 可选参数，指定生成图像的高度
            height: Optional[int] = None,
            # 可选参数，指定生成图像的宽度
            width: Optional[int] = None,
            # 可选参数，指定推理步骤的数量，默认为50
            num_inference_steps: Optional[int] = 50,
            # 可选参数，指导比例，默认为1.0
            guidance_scale: Optional[float] = 1.0,
            # 可选参数，噪声的η值，默认为0.0
            eta: Optional[float] = 0.0,
            # 可选参数，指定随机数生成器，可以是单个或多个生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选参数，指定潜在张量，默认情况下为None
            latents: Optional[torch.Tensor] = None,
            # 可选参数，输出类型，默认为“pil”格式
            output_type: Optional[str] = "pil",
            # 可选参数，指示是否返回字典形式的结果，默认为True
            return_dict: bool = True,
            # 额外参数，允许传入其他关键字参数
            **kwargs,
################################################################################
# Code for the text transformer model
################################################################################
""" PyTorch LDMBERT model."""  # 定义该代码文件的文档字符串，表示使用 PyTorch 实现 LDMBERT 模型


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

LDMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 定义一个列表，包含预训练模型的名称
    "ldm-bert",  # LDMBERT 模型的名称
    # See all LDMBert models at https://huggingface.co/models?filter=ldmbert  # 该行注释提供了 LDMBERT 模型的更多信息链接
]


LDMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {  # 定义一个字典，映射模型名称到其配置文件的 URL
    "ldm-bert": "https://huggingface.co/valhalla/ldm-bert/blob/main/config.json",  # LDMBERT 配置文件的 URL
}


""" LDMBERT model configuration"""  # 文档字符串，描述 LDMBERT 模型的配置类


class LDMBertConfig(PretrainedConfig):  # 定义 LDMBertConfig 类，继承自 PretrainedConfig
    model_type = "ldmbert"  # 设置模型类型为 "ldmbert"
    keys_to_ignore_at_inference = ["past_key_values"]  # 定义在推理时需要忽略的键
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}  # 映射属性名称

    def __init__(  # 初始化方法
        self,
        vocab_size=30522,  # 词汇表大小
        max_position_embeddings=77,  # 最大位置嵌入数
        encoder_layers=32,  # 编码器层数
        encoder_ffn_dim=5120,  # 编码器前馈网络维度
        encoder_attention_heads=8,  # 编码器注意力头数
        head_dim=64,  # 每个注意力头的维度
        encoder_layerdrop=0.0,  # 编码器层的丢弃率
        activation_function="gelu",  # 激活函数类型
        d_model=1280,  # 模型的维度
        dropout=0.1,  # 丢弃率
        attention_dropout=0.0,  # 注意力丢弃率
        activation_dropout=0.0,  # 激活丢弃率
        init_std=0.02,  # 初始化标准差
        classifier_dropout=0.0,  # 分类器丢弃率
        scale_embedding=False,  # 是否缩放嵌入
        use_cache=True,  # 是否使用缓存
        pad_token_id=0,  # 填充标记的 ID
        **kwargs,  # 其他可选关键字参数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入数
        self.d_model = d_model  # 设置模型的维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器前馈网络维度
        self.encoder_layers = encoder_layers  # 设置编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 设置编码器注意力头数
        self.head_dim = head_dim  # 设置每个注意力头的维度
        self.dropout = dropout  # 设置丢弃率
        self.attention_dropout = attention_dropout  # 设置注意力丢弃率
        self.activation_dropout = activation_dropout  # 设置激活丢弃率
        self.activation_function = activation_function  # 设置激活函数类型
        self.init_std = init_std  # 设置初始化标准差
        self.encoder_layerdrop = encoder_layerdrop  # 设置编码器层的丢弃率
        self.classifier_dropout = classifier_dropout  # 设置分类器丢弃率
        self.use_cache = use_cache  # 设置是否使用缓存
        self.num_hidden_layers = encoder_layers  # 设置隐藏层数
        self.scale_embedding = scale_embedding  # 设置是否缩放嵌入，若为 True，则缩放因子为 sqrt(d_model)

        super().__init__(pad_token_id=pad_token_id, **kwargs)  # 调用父类构造函数，传递填充标记 ID 和其他参数


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):  # 定义一个函数用于扩展注意力掩码
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.  # 函数说明，扩展注意力掩码形状
    """
    bsz, src_len = mask.size()  # 获取批次大小和源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len  # 如果目标长度未提供，则使用源序列长度

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)  # 扩展掩码的维度并转换类型

    inverted_mask = 1.0 - expanded_mask  # 反转掩码

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)  # 将反转掩码中的 True 部分填充为最小值


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->LDMBert  # 从 BartAttention 复制的类，并将 Bart 替换为 LDMBert
class LDMBertAttention(nn.Module):  # 定义 LDMBertAttention 类，继承自 nn.Module
    """Multi-headed attention from 'Attention Is All You Need' paper"""  # 文档字符串，描述这是来自论文 "Attention Is All You Need" 的多头注意力机制
    # 初始化方法，定义多头注意力层的参数
    def __init__(
        self,
        embed_dim: int,  # 嵌入向量的维度
        num_heads: int,  # 注意力头的数量
        head_dim: int,   # 每个注意力头的维度
        dropout: float = 0.0,  # Dropout 的比率，默认值为 0.0
        is_decoder: bool = False,  # 是否为解码器，默认值为 False
        bias: bool = False,  # 是否使用偏置，默认值为 False
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存嵌入维度
        self.embed_dim = embed_dim
        # 保存注意力头的数量
        self.num_heads = num_heads
        # 保存 Dropout 比率
        self.dropout = dropout
        # 保存每个头的维度
        self.head_dim = head_dim
        # 计算内部维度，即每个头维度与头数量的乘积
        self.inner_dim = head_dim * num_heads

        # 计算缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5
        # 保存是否为解码器的标志
        self.is_decoder = is_decoder

        # 创建键投影层，映射嵌入维度到内部维度
        self.k_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        # 创建值投影层，映射嵌入维度到内部维度
        self.v_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        # 创建查询投影层，映射嵌入维度到内部维度
        self.q_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        # 创建输出投影层，映射内部维度回到嵌入维度
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)

    # 形状调整方法，将输入张量调整为适合多头注意力的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 调整张量形状，并进行维度转置
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法，定义多头注意力层的计算过程
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        key_value_states: Optional[torch.Tensor] = None,  # 可选的键值状态
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 可选的过去键值状态
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
        layer_head_mask: Optional[torch.Tensor] = None,  # 可选的层头掩码
        output_attentions: bool = False,  # 是否输出注意力权重的标志
# 定义一个名为 LDMBertEncoderLayer 的神经网络模块，继承自 nn.Module
class LDMBertEncoderLayer(nn.Module):
    # 初始化方法，接收一个配置对象 config
    def __init__(self, config: LDMBertConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 获取模型嵌入维度
        self.embed_dim = config.d_model
        # 创建自注意力层，使用配置中的参数
        self.self_attn = LDMBertAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            head_dim=config.head_dim,
            dropout=config.attention_dropout,
        )
        # 创建自注意力层的归一化层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 获取 dropout 比率
        self.dropout = config.dropout
        # 根据配置选择激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 获取激活函数的 dropout 比率
        self.activation_dropout = config.activation_dropout
        # 创建第一层全连接层，输入维度为嵌入维度，输出维度为前馈网络维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 创建第二层全连接层，输入维度为前馈网络维度，输出维度为嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 创建最终的归一化层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播方法，定义输入和输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    # 定义返回类型为元组，包含一个张量和一个可选张量
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            参数:
                hidden_states (`torch.Tensor`): 输入张量，形状为 `(seq_len, batch, embed_dim)`
                attention_mask (`torch.Tensor`): 注意力掩码，大小为
                    `(batch, 1, tgt_len, src_len)`，填充元素由非常大的负值表示。
                layer_head_mask (`torch.Tensor`): 给定层中注意力头的掩码，大小为
                    `(encoder_attention_heads,)`。
                output_attentions (`bool`, *可选*):
                    是否返回所有注意力层的注意力张量。更多详细信息请参见返回张量中的 `attentions`。
            """
            # 保存输入的残差连接
            residual = hidden_states
            # 对输入的隐藏状态进行层归一化
            hidden_states = self.self_attn_layer_norm(hidden_states)
            # 计算注意力并获取权重和其他输出
            hidden_states, attn_weights, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            # 应用 dropout 正则化
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            # 将残差添加到隐藏状态中
            hidden_states = residual + hidden_states
    
            # 保存当前隐藏状态的残差连接
            residual = hidden_states
            # 对隐藏状态进行最终层归一化
            hidden_states = self.final_layer_norm(hidden_states)
            # 应用全连接层和激活函数
            hidden_states = self.activation_fn(self.fc1(hidden_states))
            # 应用激活函数的 dropout 正则化
            hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
            # 通过第二个全连接层
            hidden_states = self.fc2(hidden_states)
            # 再次应用 dropout 正则化
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            # 将残差添加到隐藏状态中
            hidden_states = residual + hidden_states
    
            # 检查隐藏状态是否为 float16 类型，并处理无穷大或 NaN 值
            if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
            ):
                # 计算限制值
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                # 将隐藏状态限制在有效范围内
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    
            # 将最终的隐藏状态放入输出元组中
            outputs = (hidden_states,)
    
            # 如果需要，添加注意力权重到输出中
            if output_attentions:
                outputs += (attn_weights,)
    
            # 返回输出元组
            return outputs
# 复制自 transformers.models.bart.modeling_bart.BartPretrainedModel，进行 Bart->LDMBert 的修改
class LDMBertPreTrainedModel(PreTrainedModel):
    # 配置类为 LDMBertConfig
    config_class = LDMBertConfig
    # 基础模型前缀
    base_model_prefix = "model"
    # 支持梯度检查点
    _supports_gradient_checkpointing = True
    # 加载时忽略的键
    _keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]

    # 初始化权重
    def _init_weights(self, module):
        # 获取初始化标准差
        std = self.config.init_std
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有填充索引，则将其权重设为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 设置梯度检查点
    def _set_gradient_checkpointing(self, module, value=False):
        # 如果模块是 LDMBertEncoder，则设置其梯度检查点属性
        if isinstance(module, (LDMBertEncoder,)):
            module.gradient_checkpointing = value

    # 属性，返回虚拟输入
    @property
    def dummy_inputs(self):
        # 获取填充标记
        pad_token = self.config.pad_token_id
        # 创建输入 ID 张量
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 创建虚拟输入字典
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        # 返回虚拟输入
        return dummy_inputs


class LDMBertEncoder(LDMBertPreTrainedModel):
    """
    包含 *config.encoder_layers* 个自注意力层的 Transformer 编码器。每层是一个
    [`LDMBertEncoderLayer`]。
    
    参数：
        config: LDMBertConfig
        embed_tokens (nn.Embedding): 输出嵌入
    """

    # 初始化方法
    def __init__(self, config: LDMBertConfig):
        # 调用父类初始化
        super().__init__(config)

        # 设置丢弃率
        self.dropout = config.dropout

        # 获取嵌入维度
        embed_dim = config.d_model
        # 获取填充索引
        self.padding_idx = config.pad_token_id
        # 获取最大源位置
        self.max_source_positions = config.max_position_embeddings

        # 初始化嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim)
        # 初始化位置嵌入层
        self.embed_positions = nn.Embedding(config.max_position_embeddings, embed_dim)
        # 创建编码器层的模块列表
        self.layers = nn.ModuleList([LDMBertEncoderLayer(config) for _ in range(config.encoder_layers)])
        # 初始化层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)

        # 默认不使用梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播方法
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        pass  # 此处代码未完待续

class LDMBertModel(LDMBertPreTrainedModel):
    # 不进行模块拆分的模块列表
    _no_split_modules = []
    # 初始化方法，接收配置参数并设置模型相关属性
        def __init__(self, config: LDMBertConfig):
            # 调用父类的初始化方法
            super().__init__(config)
            # 创建 LDMBertEncoder 模型实例
            self.model = LDMBertEncoder(config)
            # 定义一个线性层，将隐藏层大小映射到词汇表大小
            self.to_logits = nn.Linear(config.hidden_size, config.vocab_size)
    
    # 前向传播方法，定义模型的输入和输出
        def forward(
            self,
            input_ids=None,  # 输入的 ID 列表
            attention_mask=None,  # 注意力掩码，指示哪些位置需要注意
            position_ids=None,  # 位置 ID，表示输入中每个 token 的位置
            head_mask=None,  # 头掩码，用于指定哪些注意力头被禁用
            inputs_embeds=None,  # 输入的嵌入向量，替代 input_ids
            output_attentions=None,  # 是否输出注意力权重
            output_hidden_states=None,  # 是否输出隐藏状态
            return_dict=None,  # 是否返回字典格式的输出
        ):
            # 将输入传入模型，获取输出
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,  # 将注意力掩码传入模型
                position_ids=position_ids,  # 将位置 ID 传入模型
                head_mask=head_mask,  # 将头掩码传入模型
                inputs_embeds=inputs_embeds,  # 将输入嵌入向量传入模型
                output_attentions=output_attentions,  # 是否输出注意力权重
                output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
                return_dict=return_dict,  # 是否返回字典格式的输出
            )
            # 返回模型的输出结果
            return outputs
```