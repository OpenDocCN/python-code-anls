# `so-vits-svc\vencoder\wavlm\modules.py`

```py
# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

# 导入所需的库
import math
import warnings
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

# 定义一个用于转置的自定义模块
class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        # 如果指定了 deconstruct_idx，则对输入进行切片操作
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        # 对输入进行转置操作
        return x.transpose(-2, -1)

# 定义一个用于处理浮点数的 LayerNorm 模块
class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        # 对输入进行浮点数处理后再进行 LayerNorm 操作
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        # 将输出转换为与输入相同的数据类型
        return output.type_as(input)

# 定义一个用于处理浮点数的 GroupNorm 模块
class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        # 对输入进行浮点数处理后再进行 GroupNorm 操作
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        # 将输出转换为与输入相同的数据类型
        return output.type_as(input)

# 定义一个用于计算梯度乘积的自定义函数
class GradMultiply(torch.autograd.Function):
    @staticmethod
    # 定义一个名为 forward 的函数，接受参数 ctx, x, scale
    def forward(ctx, x, scale):
        # 将参数 scale 存储在上下文对象 ctx 中
        ctx.scale = scale
        # 创建一个与输入张量 x 具有相同数据类型和设备的新张量 res
        res = x.new(x)
        # 返回新张量 res
        return res
    
    # 定义一个静态方法 backward，接受参数 ctx, grad
    @staticmethod
    def backward(ctx, grad):
        # 返回 grad 乘以上下文对象 ctx 中的 scale，以及 None
        return grad * ctx.scale, None
class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        # 如果是因果卷积，需要移除的填充数为卷积核大小减一
        if causal:
            self.remove = kernel_size - 1
        else:
            # 如果不是因果卷积，需要移除的填充数为1（如果卷积核大小为偶数）或者0（如果卷积核大小为奇数）
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        # 如果需要移除填充，则对输入进行切片操作
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Swish(nn.Module):
    """Swish function
    """

    def __init__(self):
        """Construct an MultiHeadedAttention object."""
        super(Swish, self).__init__()
        # 初始化 Sigmoid 激活函数
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        # 返回输入和输入经过 Sigmoid 激活函数的乘积
        return x * self.act(x)


class GLU_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super(GLU_Linear, self).__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        # 根据激活函数类型初始化激活函数
        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        # 根据是否在 GLU 中使用偏置初始化线性层
        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        # 对输入进行线性变换
        x = self.linear(x)

        # 根据激活函数类型进行 GLU 操作
        if self.glu_type == "bilinear":
            x = (x[:, :, 0:self.output_dim] * x[:, :, self.output_dim:self.output_dim * 2])
        else:
            x = (x[:, :, 0:self.output_dim] * self.glu_act(x[:, :, self.output_dim:self.output_dim * 2]))

        return x


def gelu_accurate(x):
    # 如果没有定义 _a 属性，则初始化 _a 属性为精确 GELU 函数的系数
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    # 返回一个数学表达式的计算结果
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )
# 定义 GELU 激活函数，输入为 torch.Tensor，输出为 torch.Tensor
def gelu(x: torch.Tensor) -> torch.Tensor:
    # 使用 torch.nn.functional 中的 gelu 函数对输入进行 GELU 激活，并将结果转换为输入 x 的数据类型
    return torch.nn.functional.gelu(x.float()).type_as(x)


# 根据激活函数名称返回对应的激活函数
def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""

    # 如果激活函数为 relu，则返回 F.relu 函数
    if activation == "relu":
        return F.relu
    # 如果激活函数为 gelu，则返回 gelu 函数
    elif activation == "gelu":
        return gelu
    # 如果激活函数为 gelu_fast，则返回 gelu_accurate 函数，并发出警告
    elif activation == "gelu_fast":
        warnings.warn(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    # 如果激活函数为 gelu_accurate，则返回 gelu_accurate 函数
    elif activation == "gelu_accurate":
        return gelu_accurate
    # 如果激活函数为 tanh，则返回 torch.tanh 函数
    elif activation == "tanh":
        return torch.tanh
    # 如果激活函数为 linear，则返回一个匿名函数，输入为 x，输出为 x
    elif activation == "linear":
        return lambda x: x
    # 如果激活函数为 glu，则返回一个匿名函数，输入为 x，输出为 x
    elif activation == "glu":
        return lambda x: x
    # 如果激活函数不在以上列表中，则抛出运行时错误
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


# 初始化 BERT 模型特有的权重参数
def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    # 定义一个用于对数据进行正态分布初始化的函数
    def normal_(data):
        # 对于 FSDP，模块参数将在 CUDA 上，因此我们将它们转换回 CPU
        # 以便 RNG 与 FSDP 和非 FSDP 一致
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    # 如果 module 是 nn.Linear 类型
    if isinstance(module, nn.Linear):
        # 对权重数据进行正态分布初始化
        normal_(module.weight.data)
        # 如果存在偏置，则将偏置数据初始化为 0
        if module.bias is not None:
            module.bias.data.zero_()
    # 检查 module 是否为 nn.Embedding 类型
    if isinstance(module, nn.Embedding):
        # 对 module 的权重数据进行正态分布初始化
        normal_(module.weight.data)
        # 如果 module 的 padding_idx 不为 None，则将对应位置的权重数据置零
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    # 检查 module 是否为 MultiheadAttention 类型
    if isinstance(module, MultiheadAttention):
        # 对 module 的 q_proj 权重数据进行正态分布初始化
        normal_(module.q_proj.weight.data)
        # 对 module 的 k_proj 权重数据进行正态分布初始化
        normal_(module.k_proj.weight.data)
        # 对 module 的 v_proj 权重数据进行正态分布初始化
        normal_(module.v_proj.weight.data)
def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module  # 如果没有量化噪声，则不注册钩子

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))  # 断言模块类型为线性、嵌入或二维卷积

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4  # 检查权重是否为四维张量

    # 2D matrix
    if not is_conv:  # 如果不是卷积层
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"  # 断言输入特征必须是块大小的倍数

    # 4D matrix
    else:  # 如果是卷积层
        # 1x1 convolutions
        if module.kernel_size == (1, 1):  # 如果是1x1卷积
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"  # 断言输入通道必须是块大小的倍数
        # regular convolutions
        else:  # 如果是普通卷积
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"  # 断言卷积核大小必须是块大小的倍数
    # 定义一个前向传播的预钩子函数，用于在模型前向传播时添加噪音
    def _forward_pre_hook(mod, input):
        # 如果模型处于训练状态，则添加噪音
        if mod.training:
            # 如果不是卷积层
            if not is_conv:
                # 获取权重和大小
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # 将权重矩阵分成块，并随机丢弃选定的块
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # 获取权重和大小
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # 将权重矩阵分成块，并随机丢弃选定的块
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # 缩放权重并应用掩码
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    # 将前向传播的预钩子函数注册到模型中
    module.register_forward_pre_hook(_forward_pre_hook)
    # 返回变量 module 的值
    return module
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
            has_relative_attention_bias=False,
            num_buckets=32,
            max_distance=128,
            gru_rel_pos=False,
            rescale_init=False,
    def reset_parameters(self):
        if self.qkv_same_dim:
            # 如果查询、键、值的维度相同，使用缩放初始化可以更好地收敛
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            # 使用普通的初始化
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        # 初始化输出投影层的权重
        nn.init.xavier_uniform_(self.out_proj.weight)
        # 如果输出投影层有偏置，则初始化为0
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        # 如果存在偏置k，则使用正态分布初始化
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        # 如果存在偏置v，则使用正态分布初始化
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        # 如果存在相对注意力偏置，则使用正态分布初始化
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)
    # 根据相对位置计算相对位置桶
    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        # 获取桶的数量
        num_buckets = self.num_buckets
        # 获取最大距离
        max_distance = self.max_distance
        # 相对位置桶的初始值
        relative_buckets = 0

        # 如果是双向的
        if bidirectional:
            # 将桶的数量减半
            num_buckets = num_buckets // 2
            # 根据相对位置的正负情况，确定相对位置桶的值
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            # 取相对位置的绝对值
            relative_positions = torch.abs(relative_positions)
        else:
            # 如果是单向的，将相对位置取负值
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        # 计算最大精确值
        max_exact = num_buckets // 2
        # 判断相对位置是否小于最大精确值
        is_small = relative_positions < max_exact

        # 如果相对位置较大，根据公式计算相对位置桶的值
        relative_postion_if_large = max_exact + (
                torch.log(relative_positions.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        # 将计算得到的相对位置桶的值限制在桶的范围内
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        # 根据相对位置大小，选择相对位置桶的值
        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    # 计算偏置
    def compute_bias(self, query_length, key_length):
        # 生成查询序列的位置编码
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        # 生成键序列的位置编码
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        # 计算相对位置
        relative_position = memory_position - context_position
        # 根据相对位置计算相对位置桶
        relative_position_bucket = self._relative_positions_bucket(
            relative_position,
            bidirectional=True
        )
        # 将相对位置桶转移到相对注意力偏置的设备上
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        # 获取相对注意力偏置的值
        values = self.relative_attention_bias(relative_position_bucket)
        # 调整值的维度顺序
        values = values.permute([2, 0, 1])
        return values
    # 定义一个方法，用于实现注意力机制的前向传播
    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
            position_bias: Optional[Tensor] = None
    ):
    # 定义一个静态方法，用于在前向传播中追加先前的键填充蒙版
    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # 如果之前的 key padding mask 不为空，并且静态键值对为真，则使用之前的 key padding mask
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        # 如果之前的 key padding mask 不为空，并且当前的 key padding mask 也不为空，则将它们拼接起来
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # 在增量解码期间，由于填充标记进入和离开帧，会出现 prev 或 current 为 None 的情况
        elif prev_key_padding_mask is not None:
            # 如果源长度大于之前的 key padding mask 的长度，则填充零值，并拼接起来
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        # 如果当前的 key padding mask 不为空，则填充零值，并拼接起来
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        # 如果以上情况都不满足，则使用之前的 key padding mask
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    # 定义一个函数，接受一个字典类型的参数并返回一个字典类型的可选张量
    ) -> Dict[str, Optional[Tensor]]:
        # 从增量状态中获取"attn_state"的结果
        result = self.get_incremental_state(incremental_state, "attn_state")
        # 如果结果不为空，则返回结果
        if result is not None:
            return result
        # 如果结果为空，则创建一个空的字典并返回
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    # 定义一个函数，接受两个字典类型的参数并不返回任何值
    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
    ):
        # 设置增量状态中的"attn_state"为给定的缓冲区
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    # 定义一个函数，接受四个整数类型的参数和一个张量类型的参数，并返回一个张量类型的结果
    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        # 返回未经稀疏掩码处理的注意力权重
        return attn_weights
```