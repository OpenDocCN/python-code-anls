# `.\diffusers\pipelines\kolors\text_encoder.py`

```py
# 版权声明，说明该代码的版权归属
# Copyright 2024 ChatGLM3-6B Model Team, Kwai-Kolors Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 只能在遵守许可证的情况下使用此文件
# you may not use this file except in compliance with the License.
# 可在以下网址获取许可证的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面协议另有约定，软件按“现状”分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取特定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入数学模块，提供数学函数支持
import math
# 导入类型注解支持
from typing import List, Optional, Tuple

# 导入 PyTorch 库，用于深度学习
import torch
# 导入 PyTorch 的功能模块，包含常用的激活函数和损失函数
import torch.nn.functional as F
# 导入 PyTorch 的神经网络模块
from torch import nn
# 导入 LayerNorm 归一化层
from torch.nn import LayerNorm
# 导入 PyTorch 的 skip_init 功能，用于初始化神经网络参数
from torch.nn.utils import skip_init
# 导入预训练模型配置和模型类
from transformers import PretrainedConfig, PreTrainedModel
# 导入包含模型输出的类
from transformers.modeling_outputs import BaseModelOutputWithPast

# 从 utils 模块中导入日志记录功能
from ...utils import logging

# 创建日志记录器实例，使用当前模块名称
logger = logging.get_logger(__name__)

# 定义 ChatGLMConfig 类，继承自 PretrainedConfig
class ChatGLMConfig(PretrainedConfig):
    # 指定模型类型为 "chatglm"
    model_type = "chatglm"

    # 初始化函数，定义模型的各种超参数
    def __init__(
        # 定义默认层数为 28
        num_layers=28,
        # 定义填充词汇表的大小
        padded_vocab_size=65024,
        # 定义隐藏层的大小
        hidden_size=4096,
        # 定义前馈网络的隐藏层大小
        ffn_hidden_size=13696,
        # 定义键值通道的数量
        kv_channels=128,
        # 定义注意力头的数量
        num_attention_heads=32,
        # 定义序列长度
        seq_length=2048,
        # 定义隐藏层的丢弃率
        hidden_dropout=0.0,
        # 定义分类器的丢弃率（可选）
        classifier_dropout=None,
        # 定义注意力的丢弃率
        attention_dropout=0.0,
        # 定义 LayerNorm 的 epsilon 值
        layernorm_epsilon=1e-5,
        # 定义是否使用 RMSNorm
        rmsnorm=True,
        # 定义残差连接是否在 LayerNorm 后应用
        apply_residual_connection_post_layernorm=False,
        # 定义是否使用后层归一化
        post_layer_norm=True,
        # 定义是否添加线性层的偏置
        add_bias_linear=False,
        # 定义是否添加 QKV 的偏置
        add_qkv_bias=False,
        # 定义是否融合偏置和丢弃操作
        bias_dropout_fusion=True,
        # 定义是否使用多查询注意力
        multi_query_attention=False,
        # 定义多查询的组数量
        multi_query_group_num=1,
        # 定义是否应用查询键层缩放
        apply_query_key_layer_scaling=True,
        # 定义是否在 FP32 中执行 softmax 操作
        attention_softmax_in_fp32=True,
        # 定义是否在残差连接中使用 FP32
        fp32_residual_connection=False,
        # 定义量化位数
        quantization_bit=0,
        # 定义预序列长度（可选）
        pre_seq_len=None,
        # 定义是否进行前缀投影
        prefix_projection=False,
        # 接受其他关键字参数
        **kwargs,
    ):
        # 设置网络层的数量
        self.num_layers = num_layers
        # 设置词汇表的大小
        self.vocab_size = padded_vocab_size
        # 设置填充后的词汇表大小
        self.padded_vocab_size = padded_vocab_size
        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 设置前馈网络的隐藏层大小
        self.ffn_hidden_size = ffn_hidden_size
        # 设置键值通道的数量
        self.kv_channels = kv_channels
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置序列的长度
        self.seq_length = seq_length
        # 设置隐藏层的 dropout 概率
        self.hidden_dropout = hidden_dropout
        # 设置分类器的 dropout 概率
        self.classifier_dropout = classifier_dropout
        # 设置注意力层的 dropout 概率
        self.attention_dropout = attention_dropout
        # 设置层归一化的 epsilon 值
        self.layernorm_epsilon = layernorm_epsilon
        # 是否使用 RMSNorm 进行归一化
        self.rmsnorm = rmsnorm
        # 是否在层归一化后应用残差连接
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        # 是否使用后层归一化
        self.post_layer_norm = post_layer_norm
        # 是否在线性变换中添加偏置项
        self.add_bias_linear = add_bias_linear
        # 是否在 Q、K、V 中添加偏置项
        self.add_qkv_bias = add_qkv_bias
        # 是否融合偏置和 dropout 操作
        self.bias_dropout_fusion = bias_dropout_fusion
        # 是否使用多查询注意力机制
        self.multi_query_attention = multi_query_attention
        # 设置多查询组的数量
        self.multi_query_group_num = multi_query_group_num
        # 是否在查询和键的层之间应用缩放
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        # 是否在计算 softmax 时使用 FP32 精度
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        # 是否在残差连接中使用 FP32 精度
        self.fp32_residual_connection = fp32_residual_connection
        # 设置量化位数
        self.quantization_bit = quantization_bit
        # 设置前序列的长度
        self.pre_seq_len = pre_seq_len
        # 是否使用前缀投影
        self.prefix_projection = prefix_projection
        # 调用父类的构造函数
        super().__init__(**kwargs)
# RMSNorm 类，继承自 PyTorch 的 Module 类
class RMSNorm(torch.nn.Module):
    # 初始化方法，接收标准化形状、epsilon、设备、数据类型及其他参数
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        # 调用父类的初始化方法
        super().__init__()
        # 创建可学习的权重参数，初始化为空的张量
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        # 设置 epsilon 值
        self.eps = eps

    # 前向传播方法，接收隐藏状态张量
    def forward(self, hidden_states: torch.Tensor):
        # 获取输入张量的数据类型
        input_dtype = hidden_states.dtype
        # 计算输入张量的方差
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 归一化隐藏状态张量
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # 返回加权后的隐藏状态，转换回原始数据类型
        return (self.weight * hidden_states).to(input_dtype)


# 将配置对象转换为关键字参数的辅助函数
def _config_to_kwargs(args):
    # 创建包含数据类型的通用关键字参数字典
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    # 返回关键字参数字典
    return common_kwargs


# CoreAttention 类，继承自 PyTorch 的 Module 类
class CoreAttention(torch.nn.Module):
    # 初始化方法，接收配置和层编号
    def __init__(self, config: ChatGLMConfig, layer_number):
        # 调用父类的初始化方法
        super(CoreAttention, self).__init__()

        # 从配置中获取查询-键层的缩放应用标志
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        # 获取软最大值是否在 FP32 中的配置
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        # 如果应用查询-键层缩放，强制将软最大值设置为 FP32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        # 确保层编号至少为 1
        self.layer_number = max(1, layer_number)

        # 计算投影大小
        projection_size = config.kv_channels * config.num_attention_heads

        # 每个注意力头和每个分区的值
        self.hidden_size_per_partition = projection_size
        # 每个注意力头的隐藏大小
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        # 每个分区的注意力头数量
        self.num_attention_heads_per_partition = config.num_attention_heads

        # 初始化系数为 None
        coeff = None
        # 计算归一化因子
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        # 如果应用查询-键层缩放，更新系数和归一化因子
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        # 保存系数
        self.coeff = coeff

        # 初始化注意力 dropout
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

# 按最后一个维度拆分张量的函数
def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """拆分张量的最后一个维度。

    参数：
        tensor: 输入张量。
        num_partitions: 拆分张量的分区数量
        contiguous_split_chunks: 如果为 True，使每个块在内存中连续。

    返回：
        张量列表
    """
    # 获取张量的最后一维索引
    last_dim = tensor.dim() - 1
    # 计算每个分区的最后一维大小
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # 拆分张量
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # 注意：torch.split 默认不会创建连续的张量。
    if contiguous_split_chunks:
        # 返回每个块的连续张量
        return tuple(chunk.contiguous() for chunk in tensor_list)

    # 返回拆分后的张量列表
    return tensor_list


# 应用旋转位置嵌入的 JIT 编译函数
@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    # 获取输入张量的尺寸
    sq, _b, np, _hn = x.size(0), x.size(1), x.size(2), x.size(3)
    # 计算旋转维度
    rot_dim = rope_cache.shape[-2] * 2
    # 拆分输入张量，保留旋转维度部分和其余部分
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # 截断以支持可变大小
        rope_cache = rope_cache[:sq]
        # 重塑 x 为指定形状，-1 表示自动推断维度
        xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
        # 将 rope_cache 视图转换为新的形状，以便与 xshaped 对齐
        rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
        # 计算输出 x_out2，应用旋转公式
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        # 将 x_out2 在维度 3 上展平
        x_out2 = x_out2.flatten(3)
        # 将 x_out2 和 x_pass 在最后一个维度上连接
        return torch.cat((x_out2, x_pass), dim=-1)
# 自注意力层抽象类，继承自 PyTorch 的模块
class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h] and returns output of the same size.
    """

    # 初始化方法，接受配置、层数和设备参数
    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        # 调用父类构造函数
        super(SelfAttention, self).__init__()
        # 确保层数至少为 1
        self.layer_number = max(1, layer_number)

        # 计算投影大小
        self.projection_size = config.kv_channels * config.num_attention_heads

        # 每个注意力头和每个分区的值
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        # 是否使用多查询注意力
        self.multi_query_attention = config.multi_query_attention
        # QKV 隐藏层大小
        self.qkv_hidden_size = 3 * self.projection_size
        # 如果使用多查询注意力，调整 QKV 隐藏层大小
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        # 定义线性层以获取查询、键和值
        self.query_key_value = nn.Linear(
            config.hidden_size,
            self.qkv_hidden_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            device=device,
            **_config_to_kwargs(config),
        )

        # 核心注意力模块
        self.core_attention = CoreAttention(config, self.layer_number)

        # 输出线性层
        self.dense = nn.Linear(
            self.projection_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            device=device,
            **_config_to_kwargs(config),
        )

    # 分配内存的方法
    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        # 根据是否使用多查询注意力确定注意力头数量
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        # 返回一个空的张量以存储注意力结果
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

# 多层感知机类，继承自 PyTorch 的模块
class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h hidden dimension, perform nonlinear transformation,
    and project the state back into h hidden dimension.
    """
    # 初始化 MLP 类，接收配置和设备参数
        def __init__(self, config: ChatGLMConfig, device=None):
            # 调用父类的初始化方法
            super(MLP, self).__init__()
    
            # 设置是否在线性层中添加偏置
            self.add_bias = config.add_bias_linear
    
            # 创建一个线性层，将输入维度投影到 4h 维度，如果使用 swiglu，输出宽度翻倍
            self.dense_h_to_4h = nn.Linear(
                config.hidden_size,
                config.ffn_hidden_size * 2,
                bias=self.add_bias,
                device=device,
                **_config_to_kwargs(config),
            )
    
            # 定义 swiglu 激活函数
            def swiglu(x):
                # 将输入张量分为两部分
                x = torch.chunk(x, 2, dim=-1)
                # 返回第一部分的 silu 激活值乘以第二部分
                return F.silu(x[0]) * x[1]
    
            # 设置激活函数为 swiglu
            self.activation_func = swiglu
    
            # 创建另一个线性层，将 4h 维度投影回原始 h 维度
            self.dense_4h_to_h = nn.Linear(
                config.ffn_hidden_size, config.hidden_size, bias=self.add_bias, device=device, **_config_to_kwargs(config)
            )
    
        # 前向传播方法，接收隐藏状态作为输入
        def forward(self, hidden_states):
            # 通过第一层线性层处理输入，得到中间结果
            intermediate_parallel = self.dense_h_to_4h(hidden_states)
            # 应用激活函数于中间结果
            intermediate_parallel = self.activation_func(intermediate_parallel)
            # 通过第二层线性层得到最终输出
            output = self.dense_4h_to_h(intermediate_parallel)
            # 返回输出结果
            return output
# 定义单个变换器层的类
class GLMBlock(torch.nn.Module):
    """单个变换器层。

    变换器层接受大小为 [s, b, h] 的输入并返回相同大小的输出。
    """

    # 初始化方法，接收配置、层编号和设备参数
    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        # 调用父类构造函数
        super(GLMBlock, self).__init__()
        # 设置当前层的编号
        self.layer_number = layer_number

        # 是否在层归一化后应用残差连接
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        # 是否使用 FP32 残差连接
        self.fp32_residual_connection = config.fp32_residual_connection

        # 根据配置选择归一化函数
        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # 对输入数据进行层归一化
        self.input_layernorm = LayerNormFunc(
            config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype
        )

        # 自注意力机制
        self.self_attention = SelfAttention(config, layer_number, device=device)
        # 隐藏层的 dropout 概率
        self.hidden_dropout = config.hidden_dropout

        # 对注意力输出进行层归一化
        self.post_attention_layernorm = LayerNormFunc(
            config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype
        )

        # 多层感知机
        self.mlp = MLP(config, device=device)

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # 在变换器层开始进行层归一化
        layernorm_output = self.input_layernorm(hidden_states)
        # 自注意力计算
        attention_output, kv_cache = self.self_attention(
            layernorm_output, attention_mask, rotary_pos_emb, kv_cache=kv_cache, use_cache=use_cache
        )

        # 残差连接
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # 对注意力输出进行 dropout，并准备进行层归一化输入
        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # 在自注意力后进行层归一化
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # 多层感知机计算
        mlp_output = self.mlp(layernorm_output)

        # 第二次残差连接
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # 对多层感知机输出进行 dropout，并完成最终输出
        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        # 返回输出和键值缓存
        return output, kv_cache


# 定义变换器类
class GLMTransformer(torch.nn.Module):
    """变换器类。"""
    # 初始化方法，接受配置和设备参数
        def __init__(self, config: ChatGLMConfig, device=None):
            # 调用父类初始化方法
            super(GLMTransformer, self).__init__()
    
            # 设置浮点32位残差连接选项
            self.fp32_residual_connection = config.fp32_residual_connection
            # 设置后层归一化选项
            self.post_layer_norm = config.post_layer_norm
    
            # 设置层数
            self.num_layers = config.num_layers
    
            # 定义构建层的方法
            def build_layer(layer_number):
                # 创建并返回 GLMBlock 层实例
                return GLMBlock(config, layer_number, device=device)
    
            # 生成指定数量的层并加入模块列表
            self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])
    
            # 如果启用后层归一化
            if self.post_layer_norm:
                # 选择归一化方法
                LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
                # 创建最终的层归一化层
                self.final_layernorm = LayerNormFunc(
                    config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype
                )
    
            # 初始化梯度检查点开关
            self.gradient_checkpointing = False
    
        # 获取指定层的方法
        def _get_layer(self, layer_number):
            # 返回指定层的实例
            return self.layers[layer_number]
    
        # 前向传播方法
        def forward(
            self,
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
        ):
            # 如果未提供 kv_caches，初始化为 None 列表
            if not kv_caches:
                kv_caches = [None for _ in range(self.num_layers)]
            # 根据 use_cache 设置 presents 为元组或 None
            presents = () if use_cache else None
            # 如果启用梯度检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 如果使用缓存，记录警告并禁用缓存
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
    
            # 初始化存储自注意力和隐藏状态的变量
            all_self_attentions = None
            all_hidden_states = () if output_hidden_states else None
            # 遍历每一层
            for index in range(self.num_layers):
                # 如果输出隐藏状态，记录当前隐藏状态
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
    
                # 获取当前层
                layer = self._get_layer(index)
                # 如果启用梯度检查点且处于训练模式
                if self.gradient_checkpointing and self.training:
                    # 使用检查点计算层的输出
                    layer_ret = torch.utils.checkpoint.checkpoint(
                        layer, hidden_states, attention_mask, rotary_pos_emb, kv_caches[index], use_cache
                    )
                else:
                    # 正常计算层的输出
                    layer_ret = layer(
                        hidden_states, attention_mask, rotary_pos_emb, kv_cache=kv_caches[index], use_cache=use_cache
                    )
                # 解包层输出
                hidden_states, kv_cache = layer_ret
                # 如果使用缓存，记录 kv_cache
                if use_cache:
                    presents = presents + (kv_cache,)
    
            # 如果输出隐藏状态，记录最后的隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 如果启用后层归一化
            if self.post_layer_norm:
                # 应用最终的层归一化
                hidden_states = self.final_layernorm(hidden_states)
    
            # 返回最终的隐藏状态、缓存和所有隐藏状态及自注意力
            return hidden_states, presents, all_hidden_states, all_self_attentions
# 定义一个抽象类，用于处理权重初始化和下载、加载预训练模型的接口
class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置是否可并行化
    is_parallelizable = False
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 配置类
    config_class = ChatGLMConfig
    # 基础模型前缀
    base_model_prefix = "transformer"
    # 不可拆分的模块列表
    _no_split_modules = ["GLMBlock"]

    # 初始化权重的方法
    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    # 获取掩码的方法
    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        # 获取输入的批次大小和序列长度
        batch_size, seq_length = input_ids.shape
        # 创建一个全为1的注意力掩码
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        # 只保留下三角部分的注意力掩码
        full_attention_mask.tril_()
        past_length = 0
        # 如果有过去的键值，获取其长度
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        # 如果有过去的长度，拼接全为1的掩码
        if past_length:
            full_attention_mask = torch.cat(
                (torch.ones(batch_size, seq_length, past_length, device=input_ids.device), full_attention_mask), dim=-1
            )
        # 如果提供了填充掩码，更新注意力掩码
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        # 如果没有过去的长度并且有填充掩码，调整掩码
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        # 将掩码转换为布尔值
        full_attention_mask = (full_attention_mask < 0.5).bool()
        # 增加维度
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    # 获取位置ID的方法
    def get_position_ids(self, input_ids, device):
        # 获取批次大小和序列长度
        batch_size, seq_length = input_ids.shape
        # 创建位置ID
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    # 设置梯度检查点的方法
    def _set_gradient_checkpointing(self, module, value=False):
        # 如果模块是GLMTransformer，设置其梯度检查点
        if isinstance(module, GLMTransformer):
            module.gradient_checkpointing = value


# 默认初始化方法
def default_init(cls, *args, **kwargs):
    # 创建类的实例
    return cls(*args, **kwargs)


# 定义一个嵌入类
class Embedding(torch.nn.Module):
    """Language model embeddings."""

    # 初始化方法
    def __init__(self, config: ChatGLMConfig, device=None):
        super(Embedding, self).__init__()

        # 获取隐藏层大小
        self.hidden_size = config.hidden_size
        # 创建词嵌入层
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size, self.hidden_size, dtype=config.torch_dtype, device=device
        )
        # 获取是否使用fp32残差连接的配置
        self.fp32_residual_connection = config.fp32_residual_connection

    # 前向传播方法
    def forward(self, input_ids):
        # 获取词嵌入
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # 转置数据格式以避免显式转置
        embeddings = embeddings.transpose(0, 1).contiguous()
        # 如果启用fp32残差连接，转换为浮点数
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


# 定义一个旋转嵌入类
class RotaryEmbedding(nn.Module):
    # 初始化方法，用于设置类的初始状态
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        # 调用父类的初始化方法
        super().__init__()
        # 计算逆频率，公式为 1/(10000^(2i/d))，用于位置编码
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        # 将计算得到的逆频率注册为缓冲区，以便在模型保存和加载时保持
        self.register_buffer("inv_freq", inv_freq)
        # 存储维度参数
        self.dim = dim
        # 存储原始实现的标志
        self.original_impl = original_impl

    # 前向传播实现方法，计算位置编码
    def forward_impl(self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000):
        """增强型变换器，带有旋转位置嵌入。

        来源: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT 许可证:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # 计算位置编码的 theta 值，公式为 1/(base^(2(i-1)/n_elem))
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # 创建位置索引，范围为 [0, 1, ..., seq_len - 1]
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # 计算位置索引与 theta 的外积，生成位置编码
        idx_theta = torch.outer(seq_idx, theta).float()

        # 计算缓存，将余弦和正弦值按最后一个维度堆叠
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # 为了模拟 complex32 的行为，避免不同结果，进行数据类型转换
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            # 如果 dtype 为 bfloat16，则转换缓存为 bfloat16；否则转换为 half
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        # 返回计算得到的缓存
        return cache

    # 前向传播方法，接收最大序列长度和偏移量
    def forward(self, max_seq_len, offset=0):
        # 调用 forward_impl 方法，传入相应参数
        return self.forward_impl(max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
# 定义一个前缀编码器类，继承自 PyTorch 的 nn.Module
class PrefixEncoder(torch.nn.Module):
    """
    前缀编码的 PyTorch nn 模型 输入形状: (batch-size, prefix-length) 输出形状: (batch-size,
    prefix-length, 2*layers*hidden)
    """

    # 初始化函数，接受配置对象
    def __init__(self, config: ChatGLMConfig):
        # 调用父类初始化
        super().__init__()
        # 获取前缀投影的配置
        self.prefix_projection = config.prefix_projection
        # 如果启用了前缀投影
        if self.prefix_projection:
            # 使用两层 MLP 编码前缀
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            # 创建嵌入层，输出维度为 kv_size
            self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)
            # 创建一个顺序的网络结构，包括两层线性变换和一个 Tanh 激活函数
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, config.hidden_size),  # 第一层线性变换
                torch.nn.Tanh(),  # Tanh 激活函数
                torch.nn.Linear(config.hidden_size, kv_size),  # 第二层线性变换
            )
        else:
            # 如果没有前缀投影，直接创建嵌入层，输出维度为 num_layers * kv_channels * multi_query_group_num * 2
            self.embedding = torch.nn.Embedding(
                config.pre_seq_len, config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            )

    # 前向传播函数，接受前缀张量
    def forward(self, prefix: torch.Tensor):
        # 如果启用了前缀投影
        if self.prefix_projection:
            # 使用嵌入层对前缀进行编码
            prefix_tokens = self.embedding(prefix)
            # 通过转换网络得到过去的键值对
            past_key_values = self.trans(prefix_tokens)
        else:
            # 直接通过嵌入层得到过去的键值对
            past_key_values = self.embedding(prefix)
        # 返回过去的键值对
        return past_key_values


# 定义 ChatGLM 模型类，继承自 ChatGLMPreTrainedModel
class ChatGLMModel(ChatGLMPreTrainedModel):
    # 初始化函数，接受配置对象、设备和空初始化标志
    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        # 调用父类初始化
        super().__init__(config)
        # 根据空初始化标志选择初始化方法
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        # 如果指定了设备，将设备信息添加到初始化参数
        if device is not None:
            init_kwargs["device"] = device
        # 初始化嵌入层
        self.embedding = init_method(Embedding, config, **init_kwargs)
        # 保存层数、查询组数和键值通道数
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # 旋转位置嵌入的序列长度
        self.seq_length = config.seq_length
        # 计算旋转嵌入的维度
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        # 创建旋转嵌入对象
        self.rotary_pos_emb = RotaryEmbedding(
            rotary_dim // 2, original_impl=config.original_rope, device=device, dtype=config.torch_dtype
        )
        # 初始化 GLMTransformer 编码器
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        # 初始化输出层，线性变换
        self.output_layer = init_method(
            nn.Linear,
            config.hidden_size,
            config.padded_vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            **init_kwargs,
        )
        # 获取前缀序列长度
        self.pre_seq_len = config.pre_seq_len
        # 获取前缀投影的配置
        self.prefix_projection = config.prefix_projection
        # 如果前缀序列长度不为空
        if self.pre_seq_len is not None:
            # 将所有参数的梯度计算标志设置为 False
            for param in self.parameters():
                param.requires_grad = False
            # 创建前缀 token 的张量
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            # 创建前缀编码器对象
            self.prefix_encoder = PrefixEncoder(config)
            # 创建 dropout 层，丢弃率为 0.1
            self.dropout = torch.nn.Dropout(0.1)
    # 获取输入的嵌入层
        def get_input_embeddings(self):
            # 返回嵌入层中的单词嵌入
            return self.embedding.word_embeddings
    
        # 获取提示信息，供模型使用
        def get_prompt(self, batch_size, device, dtype=torch.half):
            # 扩展前缀标记以匹配批量大小，并移动到指定设备
            prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
            # 通过前缀编码器处理前缀标记并转换为指定数据类型
            past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
            # 将处理后的数据重塑为特定形状以符合模型需求
            past_key_values = past_key_values.view(
                batch_size, self.pre_seq_len, self.num_layers * 2, self.multi_query_group_num, self.kv_channels
            )
            # 应用丢弃层以防止过拟合
            past_key_values = self.dropout(past_key_values)
            # 调整维度顺序并分割为多个张量以供后续使用
            past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
            # 返回处理后的过去键值对
            return past_key_values
    
        # 定义前向传播方法
        def forward(
            self,
            input_ids,
            # 可选的位置 ID，用于编码输入的位置信息
            position_ids: Optional[torch.Tensor] = None,
            # 可选的注意力掩码，用于屏蔽输入中的无效位置
            attention_mask: Optional[torch.BoolTensor] = None,
            # 完整注意力掩码，用于更复杂的注意力机制
            full_attention_mask: Optional[torch.BoolTensor] = None,
            # 可选的过去键值对，用于缓存上一次的计算结果
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            # 可选的输入嵌入，替代 input_ids 使用
            inputs_embeds: Optional[torch.Tensor] = None,
            # 可选的缓存使用标志
            use_cache: Optional[bool] = None,
            # 可选的隐藏状态输出标志
            output_hidden_states: Optional[bool] = None,
            # 可选的返回字典的标志
            return_dict: Optional[bool] = None,
    ):
        # 如果 output_hidden_states 没有指定，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 use_cache 没有指定，则使用配置中的默认值
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果 return_dict 没有指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取输入的 batch_size 和 seq_length
        batch_size, seq_length = input_ids.shape

        # 如果没有输入的嵌入，则使用输入的 ID 生成嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        # 检查预序列长度
        if self.pre_seq_len is not None:
            # 如果过去的键值对为空，则获取提示信息
            if past_key_values is None:
                past_key_values = self.get_prompt(
                    batch_size=batch_size, device=input_ids.device, dtype=inputs_embeds.dtype
                )
            # 如果存在注意力掩码，则将预序列的掩码添加到前面
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask.new_ones((batch_size, self.pre_seq_len)), attention_mask], dim=-1
                )

        # 如果全注意力掩码为空
        if full_attention_mask is None:
            # 如果存在注意力掩码且不是全为 1，或者过去的键值对存在且序列长度不为 1，则获取掩码
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # 计算旋转位置嵌入
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        # 如果指定了位置 ID，则根据位置 ID 索引旋转嵌入
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            # 如果没有位置 ID，则使用序列长度生成旋转嵌入
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        # 转置并确保连续性
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # 运行编码器
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        # 如果不返回字典，则返回非 None 的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        # 返回包含隐藏状态、过去的键值、所有隐藏状态和注意力的自定义输出对象
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
```