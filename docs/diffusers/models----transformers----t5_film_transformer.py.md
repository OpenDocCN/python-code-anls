# `.\diffusers\models\transformers\t5_film_transformer.py`

```py
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，依据许可证分发的软件
# 是按“原样”基础分发的，没有任何形式的保证或条件，
# 无论是明示还是暗示。有关许可证下特定语言的权限和
# 限制，请参阅许可证。
# 导入数学模块以执行数学运算
import math
# 从 typing 导入可选类型和元组
from typing import Optional, Tuple

# 导入 PyTorch 库
import torch
# 从 torch 导入神经网络模块
from torch import nn

# 导入配置工具和注册功能
from ...configuration_utils import ConfigMixin, register_to_config
# 导入注意力处理器
from ..attention_processor import Attention
# 导入获取时间步嵌入的功能
from ..embeddings import get_timestep_embedding
# 导入模型工具的基类
from ..modeling_utils import ModelMixin


class T5FilmDecoder(ModelMixin, ConfigMixin):
    r"""
    T5 风格的解码器，具有 FiLM 条件。

    参数：
        input_dims (`int`, *可选*, 默认为 `128`):
            输入维度的数量。
        targets_length (`int`, *可选*, 默认为 `256`):
            目标的长度。
        d_model (`int`, *可选*, 默认为 `768`):
            输入隐藏状态的大小。
        num_layers (`int`, *可选*, 默认为 `12`):
            使用的 `DecoderLayer` 数量。
        num_heads (`int`, *可选*, 默认为 `12`):
            使用的注意力头的数量。
        d_kv (`int`, *可选*, 默认为 `64`):
            键值投影向量的大小。
        d_ff (`int`, *可选*, 默认为 `2048`):
            `DecoderLayer` 中间前馈层的维度数量。
        dropout_rate (`float`, *可选*, 默认为 `0.1`):
            丢弃概率。
    """

    # 使用装饰器注册初始化函数到配置
    @register_to_config
    def __init__(
        # 输入维度，默认为128
        self,
        input_dims: int = 128,
        # 目标长度，默认为256
        targets_length: int = 256,
        # 最大解码噪声时间，默认为2000.0
        max_decoder_noise_time: float = 2000.0,
        # 隐藏状态的维度，默认为768
        d_model: int = 768,
        # 解码层的数量，默认为12
        num_layers: int = 12,
        # 注意力头的数量，默认为12
        num_heads: int = 12,
        # 键值维度大小，默认为64
        d_kv: int = 64,
        # 中间前馈层的维度，默认为2048
        d_ff: int = 2048,
        # 丢弃率，默认为0.1
        dropout_rate: float = 0.1,
    # 初始化父类
        ):
            super().__init__()
    
            # 创建条件嵌入层，包含两层线性变换和激活函数
            self.conditioning_emb = nn.Sequential(
                # 第一个线性层，输入维度为 d_model，输出维度为 d_model * 4，不使用偏置
                nn.Linear(d_model, d_model * 4, bias=False),
                # 使用 SiLU 激活函数
                nn.SiLU(),
                # 第二个线性层，输入维度为 d_model * 4，输出维度同样为 d_model * 4，不使用偏置
                nn.Linear(d_model * 4, d_model * 4, bias=False),
                # 使用 SiLU 激活函数
                nn.SiLU(),
            )
    
            # 创建位置编码嵌入，大小为 (targets_length, d_model)
            self.position_encoding = nn.Embedding(targets_length, d_model)
            # 禁止位置编码的权重更新
            self.position_encoding.weight.requires_grad = False
    
            # 创建连续输入的线性投影层，输入维度为 input_dims，输出维度为 d_model，不使用偏置
            self.continuous_inputs_projection = nn.Linear(input_dims, d_model, bias=False)
    
            # 创建 dropout 层，丢弃率为 dropout_rate
            self.dropout = nn.Dropout(p=dropout_rate)
    
            # 创建解码器层的模块列表
            self.decoders = nn.ModuleList()
            # 循环创建 num_layers 个解码器层
            for lyr_num in range(num_layers):
                # 初始化 FiLM 条件 T5 解码器层
                lyr = DecoderLayer(d_model=d_model, d_kv=d_kv, num_heads=num_heads, d_ff=d_ff, dropout_rate=dropout_rate)
                # 将解码器层添加到列表中
                self.decoders.append(lyr)
    
            # 创建解码器层的归一化层
            self.decoder_norm = T5LayerNorm(d_model)
    
            # 创建后续 dropout 层，丢弃率为 dropout_rate
            self.post_dropout = nn.Dropout(p=dropout_rate)
            # 创建输出层，将 d_model 的输出映射回 input_dims，不使用偏置
            self.spec_out = nn.Linear(d_model, input_dims, bias=False)
    
        # 定义编码器-解码器掩码函数
        def encoder_decoder_mask(self, query_input: torch.Tensor, key_input: torch.Tensor) -> torch.Tensor:
            # 计算查询和键输入的掩码，进行逐元素相乘
            mask = torch.mul(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
            # 返回掩码并扩展维度
            return mask.unsqueeze(-3)
    # 前向传播方法，接受编码及掩码、解码器输入和噪声时间
        def forward(self, encodings_and_masks, decoder_input_tokens, decoder_noise_time):
            # 获取批次大小和解码器输入的形状
            batch, _, _ = decoder_input_tokens.shape
            # 确保噪声时间的形状与批次一致
            assert decoder_noise_time.shape == (batch,)
    
            # 将 decoder_noise_time 重新缩放到期望的时间范围
            time_steps = get_timestep_embedding(
                decoder_noise_time * self.config.max_decoder_noise_time,
                embedding_dim=self.config.d_model,
                max_period=self.config.max_decoder_noise_time,
            ).to(dtype=self.dtype)
    
            # 使用时间步长生成条件嵌入，并扩展维度
            conditioning_emb = self.conditioning_emb(time_steps).unsqueeze(1)
    
            # 确保条件嵌入的形状正确
            assert conditioning_emb.shape == (batch, 1, self.config.d_model * 4)
    
            # 获取解码器输入的序列长度
            seq_length = decoder_input_tokens.shape[1]
    
            # 如果使用相对位置，基于编码和掩码的长度偏移序列
            decoder_positions = torch.broadcast_to(
                torch.arange(seq_length, device=decoder_input_tokens.device),
                (batch, seq_length),
            )
    
            # 计算位置编码
            position_encodings = self.position_encoding(decoder_positions)
    
            # 对解码器输入进行连续输入投影
            inputs = self.continuous_inputs_projection(decoder_input_tokens)
            # 将位置编码添加到输入中
            inputs += position_encodings
            # 应用 dropout 操作
            y = self.dropout(inputs)
    
            # 创建解码器掩码，没有填充
            decoder_mask = torch.ones(
                decoder_input_tokens.shape[:2], device=decoder_input_tokens.device, dtype=inputs.dtype
            )
    
            # 将编码掩码转换为编码器-解码器掩码
            encodings_and_encdec_masks = [(x, self.encoder_decoder_mask(decoder_mask, y)) for x, y in encodings_and_masks]
    
            # 交叉注意力风格：拼接编码
            encoded = torch.cat([x[0] for x in encodings_and_encdec_masks], dim=1)
            encoder_decoder_mask = torch.cat([x[1] for x in encodings_and_encdec_masks], dim=-1)
    
            # 对每一层解码器进行循环处理
            for lyr in self.decoders:
                y = lyr(
                    y,
                    conditioning_emb=conditioning_emb,
                    encoder_hidden_states=encoded,
                    encoder_attention_mask=encoder_decoder_mask,
                )[0]
    
            # 对输出进行归一化
            y = self.decoder_norm(y)
            # 应用 dropout 后处理
            y = self.post_dropout(y)
    
            # 生成最终的频谱输出
            spec_out = self.spec_out(y)
            # 返回频谱输出
            return spec_out
# T5 解码器层的定义
class DecoderLayer(nn.Module):
    r"""
    T5 decoder layer.  # T5解码器层的文档说明

    Args:  # 参数说明
        d_model (`int`):  # 输入隐藏状态的大小
            Size of the input hidden states.  # 输入隐藏状态的大小
        d_kv (`int`):  # 键值投影向量的大小
            Size of the key-value projection vectors.  # 键值投影向量的大小
        num_heads (`int`):  # 注意力头的数量
            Number of attention heads.  # 注意力头的数量
        d_ff (`int`):  # 中间前馈层的大小
            Size of the intermediate feed-forward layer.  # 中间前馈层的大小
        dropout_rate (`float`):  # 丢弃概率
            Dropout probability.  # 丢弃概率
        layer_norm_epsilon (`float`, *optional*, defaults to `1e-6`):  # 数值稳定性的小值
            A small value used for numerical stability to avoid dividing by zero.  # 数值稳定性的小值
    """

    # 初始化方法，定义各个参数
    def __init__(
        self, d_model: int, d_kv: int, num_heads: int, d_ff: int, dropout_rate: float, layer_norm_epsilon: float = 1e-6
    ):
        super().__init__()  # 调用父类构造函数
        self.layer = nn.ModuleList()  # 初始化模块列表以存储层

        # 条件自注意力：第 0 层
        self.layer.append(
            T5LayerSelfAttentionCond(d_model=d_model, d_kv=d_kv, num_heads=num_heads, dropout_rate=dropout_rate)  # 添加条件自注意力层
        )

        # 交叉注意力：第 1 层
        self.layer.append(
            T5LayerCrossAttention(
                d_model=d_model,  # 输入隐藏状态的大小
                d_kv=d_kv,  # 键值投影向量的大小
                num_heads=num_heads,  # 注意力头的数量
                dropout_rate=dropout_rate,  # 丢弃概率
                layer_norm_epsilon=layer_norm_epsilon,  # 数值稳定性的小值
            )
        )

        # Film Cond MLP + 丢弃：最后一层
        self.layer.append(
            T5LayerFFCond(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, layer_norm_epsilon=layer_norm_epsilon)  # 添加条件前馈层
        )

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态张量
        conditioning_emb: Optional[torch.Tensor] = None,  # 条件嵌入（可选）
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码（可选）
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态（可选）
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器注意力掩码（可选）
        encoder_decoder_position_bias=None,  # 编码器-解码器位置偏置
    ) -> Tuple[torch.Tensor]:  # 返回张量的元组
        hidden_states = self.layer[0](  # 通过第一层处理输入隐藏状态
            hidden_states,
            conditioning_emb=conditioning_emb,  # 使用条件嵌入
            attention_mask=attention_mask,  # 使用注意力掩码
        )

        # 如果存在编码器隐藏状态
        if encoder_hidden_states is not None:
            # 扩展编码器注意力掩码
            encoder_extended_attention_mask = torch.where(encoder_attention_mask > 0, 0, -1e10).to(
                encoder_hidden_states.dtype  # 转换为编码器隐藏状态的数据类型
            )

            hidden_states = self.layer[1](  # 通过第二层处理隐藏状态
                hidden_states,
                key_value_states=encoder_hidden_states,  # 使用编码器隐藏状态作为键值
                attention_mask=encoder_extended_attention_mask,  # 使用扩展的注意力掩码
            )

        # 应用 Film 条件前馈层
        hidden_states = self.layer[-1](hidden_states, conditioning_emb)  # 通过最后一层处理隐藏状态，使用条件嵌入

        return (hidden_states,)  # 返回处理后的隐藏状态元组


# T5样式的自注意力层，带条件
class T5LayerSelfAttentionCond(nn.Module):
    r"""
    T5 style self-attention layer with conditioning.  # T5样式的自注意力层，带条件说明
    # 函数参数说明
    Args:
        d_model (`int`):  # 输入隐藏状态的大小
            Size of the input hidden states.
        d_kv (`int`):  # 键值投影向量的大小
            Size of the key-value projection vectors.
        num_heads (`int`):  # 注意力头的数量
            Number of attention heads.
        dropout_rate (`float`):  # 丢弃概率
            Dropout probability.
    """

    # 初始化方法，设置类的基本参数
    def __init__(self, d_model: int, d_kv: int, num_heads: int, dropout_rate: float):
        super().__init__()  # 调用父类构造函数
        # 创建层归一化层，输入大小为 d_model
        self.layer_norm = T5LayerNorm(d_model)
        # 创建 FiLM 层，输入特征为 d_model * 4，输出特征为 d_model
        self.FiLMLayer = T5FiLMLayer(in_features=d_model * 4, out_features=d_model)
        # 创建注意力层，设定查询维度、头数、键值维度等参数
        self.attention = Attention(query_dim=d_model, heads=num_heads, dim_head=d_kv, out_bias=False, scale_qk=False)
        # 创建丢弃层，设定丢弃概率
        self.dropout = nn.Dropout(dropout_rate)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        conditioning_emb: Optional[torch.Tensor] = None,  # 可选的条件嵌入
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
    ) -> torch.Tensor:
        # 对输入的隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)

        # 如果有条件嵌入，应用 FiLM 层
        if conditioning_emb is not None:
            normed_hidden_states = self.FiLMLayer(normed_hidden_states, conditioning_emb)

        # 自注意力模块，获取注意力输出
        attention_output = self.attention(normed_hidden_states)

        # 将注意力输出与原隐藏状态相加，并应用丢弃层
        hidden_states = hidden_states + self.dropout(attention_output)

        # 返回更新后的隐藏状态
        return hidden_states
# T5风格的交叉注意力层
class T5LayerCrossAttention(nn.Module):
    r"""
    T5风格的交叉注意力层。

    参数：
        d_model (`int`):
            输入隐藏状态的大小。
        d_kv (`int`):
            键值投影向量的大小。
        num_heads (`int`):
            注意力头的数量。
        dropout_rate (`float`):
            丢弃概率。
        layer_norm_epsilon (`float`):
            用于数值稳定性的小值，避免除以零。
    """

    # 初始化方法，设置模型参数
    def __init__(self, d_model: int, d_kv: int, num_heads: int, dropout_rate: float, layer_norm_epsilon: float):
        # 调用父类初始化方法
        super().__init__()
        # 创建注意力层
        self.attention = Attention(query_dim=d_model, heads=num_heads, dim_head=d_kv, out_bias=False, scale_qk=False)
        # 创建层归一化层
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        # 创建丢弃层
        self.dropout = nn.Dropout(dropout_rate)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 计算注意力输出
        attention_output = self.attention(
            normed_hidden_states,
            encoder_hidden_states=key_value_states,
            attention_mask=attention_mask.squeeze(1),
        )
        # 计算层输出，添加丢弃
        layer_output = hidden_states + self.dropout(attention_output)
        # 返回层输出
        return layer_output


# T5风格的前馈条件层
class T5LayerFFCond(nn.Module):
    r"""
    T5风格的前馈条件层。

    参数：
        d_model (`int`):
            输入隐藏状态的大小。
        d_ff (`int`):
            中间前馈层的大小。
        dropout_rate (`float`):
            丢弃概率。
        layer_norm_epsilon (`float`):
            用于数值稳定性的小值，避免除以零。
    """

    # 初始化方法，设置模型参数
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, layer_norm_epsilon: float):
        # 调用父类初始化方法
        super().__init__()
        # 创建带门激活的前馈层
        self.DenseReluDense = T5DenseGatedActDense(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)
        # 创建条件层
        self.film = T5FiLMLayer(in_features=d_model * 4, out_features=d_model)
        # 创建层归一化层
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        # 创建丢弃层
        self.dropout = nn.Dropout(dropout_rate)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor, conditioning_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 对隐藏状态进行层归一化
        forwarded_states = self.layer_norm(hidden_states)
        # 如果存在条件嵌入，则应用条件层
        if conditioning_emb is not None:
            forwarded_states = self.film(forwarded_states, conditioning_emb)

        # 应用前馈层
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 更新隐藏状态，添加丢弃
        hidden_states = hidden_states + self.dropout(forwarded_states)
        # 返回更新后的隐藏状态
        return hidden_states


# T5风格的前馈层，具有门控激活和丢弃
class T5DenseGatedActDense(nn.Module):
    r"""
    T5风格的前馈层，具有门控激活和丢弃。
    # 参数说明部分
    Args:
        d_model (`int`):  # 输入隐藏状态的尺寸
            Size of the input hidden states.
        d_ff (`int`):  # 中间前馈层的尺寸
            Size of the intermediate feed-forward layer.
        dropout_rate (`float`):  # 丢弃概率
            Dropout probability.
    """

    # 初始化方法，接受模型参数
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()  # 调用父类的初始化方法
        # 定义第一线性变换层，不使用偏置，输入维度为d_model，输出维度为d_ff
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        # 定义第二线性变换层，不使用偏置，输入维度为d_model，输出维度为d_ff
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        # 定义输出线性变换层，不使用偏置，输入维度为d_ff，输出维度为d_model
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        # 定义丢弃层，使用指定的丢弃概率
        self.dropout = nn.Dropout(dropout_rate)
        # 初始化自定义激活函数
        self.act = NewGELUActivation()

    # 前向传播方法，接受输入的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过第一线性层和激活函数得到隐层状态
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 通过第二线性层得到隐层状态
        hidden_linear = self.wi_1(hidden_states)
        # 将两个隐层状态进行逐元素相乘
        hidden_states = hidden_gelu * hidden_linear
        # 应用丢弃层
        hidden_states = self.dropout(hidden_states)

        # 通过输出线性层得到最终的隐层状态
        hidden_states = self.wo(hidden_states)
        # 返回最终的隐层状态
        return hidden_states
# T5风格的层归一化模块
class T5LayerNorm(nn.Module):
    r"""
    T5风格的层归一化模块。

    Args:
        hidden_size (`int`):
            输入隐藏状态的大小。
        eps (`float`, `optional`, defaults to `1e-6`):
            用于数值稳定性的小值，以避免除以零。
    """

    # 初始化函数，接受隐藏状态大小和epsilon
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        构造一个T5风格的层归一化模块。没有偏置，也不减去均值。
        """
        # 调用父类构造函数
        super().__init__()
        # 初始化权重为全1的可学习参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 存储epsilon值
        self.variance_epsilon = eps

    # 前向传播函数，接受隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 计算隐藏状态的方差，使用平方和的均值，保持维度
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 按照方差进行归一化，同时考虑到epsilon
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重为半精度，转换隐藏状态为相应类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # 返回归一化后的结果乘以权重
        return self.weight * hidden_states


# 实现GELU激活函数的模块
class NewGELUActivation(nn.Module):
    """
    实现与Google BERT库中相同的GELU激活函数（与OpenAI GPT相同）。也可以参考
    Gaussian Error Linear Units论文：https://arxiv.org/abs/1606.08415
    """

    # 前向传播函数，接受输入张量
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 计算GELU激活值
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# T5风格的FiLM层
class T5FiLMLayer(nn.Module):
    """
    T5风格的FiLM层。

    Args:
        in_features (`int`):
            输入特征的数量。
        out_features (`int`):
            输出特征的数量。
    """

    # 初始化函数，接受输入和输出特征数量
    def __init__(self, in_features: int, out_features: int):
        # 调用父类构造函数
        super().__init__()
        # 定义线性层，用于生成缩放和偏移参数
        self.scale_bias = nn.Linear(in_features, out_features * 2, bias=False)

    # 前向传播函数，接受输入张量和条件嵌入
    def forward(self, x: torch.Tensor, conditioning_emb: torch.Tensor) -> torch.Tensor:
        # 通过线性层计算缩放和偏移
        emb = self.scale_bias(conditioning_emb)
        # 将结果分成缩放和偏移两个部分
        scale, shift = torch.chunk(emb, 2, -1)
        # 进行缩放和偏移操作
        x = x * (1 + scale) + shift
        # 返回处理后的结果
        return x
```