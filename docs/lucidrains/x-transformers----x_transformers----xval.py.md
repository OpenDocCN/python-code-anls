# `.\lucidrains\x-transformers\x_transformers\xval.py`

```py
"""
定义了一个基于离散标记的常规变换器，但对于数字是连续的
更好地泛化了算术
https://arxiv.org/abs/2310.02989
"""

# 导入所需的库
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Callable
from collections import namedtuple

from einops import rearrange
from einops.layers.torch import Rearrange

from x_transformers.x_transformers import (
    AttentionLayers,
    TokenEmbedding,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding
)

from x_transformers.autoregressive_wrapper import (
    top_k,
    top_p
)

# 常量

# 定义一个命名元组，用于表示损失的细分
LossBreakdown = namedtuple('LossBreakdown', ['cross_entropy_loss', 'numerical_mse_loss'])

# 定义一个命名元组，用于表示生成的返回结果
GenerateReturn = namedtuple('GenerateReturn', ['sampled_token_ids', 'sampled_numbers', 'is_number_mask'])

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 主要类

class XValTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        numerical_token_id,
        attn_layers: AttentionLayers,
        emb_dim = None,
        logits_dim = None,
        tie_embedding = False,
        max_mem_len = 0,
        num_memory_tokens = None,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False
    ):
        super().__init__()
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.emb_dim = emb_dim
        self.token_emb = TokenEmbedding(emb_dim, num_tokens)

        self.numerical_token_id = numerical_token_id

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len

        if not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb):
            self.pos_emb = always(0)  # 如果不使用绝对位置编码或者禁用了绝对位置编码，则将位置编码设置为常数0
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)  # 如果使用了缩放的正弦位置编码，则使用缩放的正弦位置编码
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)  # 否则使用绝对位置编码

        self.emb_dropout = nn.Dropout(emb_dropout)

        # 内存标记

        num_memory_tokens = default(num_memory_tokens, 0)
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))  # 初始化内存标记

        # 注意力层

        self.attn_layers = attn_layers

        # 转换为logits

        logits_dim = default(logits_dim, num_tokens)
        self.to_logits = nn.Linear(dim, logits_dim) if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()

        self.to_numerical_output = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...')
        )

    def forward(
        self,
        x: Tensor,
        x_num: Tensor,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        return_attn = False,
        mems = None,
        pos = None,
        prepend_embeds = None,
        **kwargs
        ):
        # 断言输入张量 x 的形状与 x_num 的形状相同
        assert x.shape == x_num.shape

        # 获取批次大小
        batch = x.shape[0]

        # 创建数值标记掩码
        is_number_mask = x == self.numerical_token_id

        # 对输入进行 token 嵌入
        x = self.token_emb(x)

        # 根据数值标记掩码调整缩放因子
        scale = torch.where(is_number_mask, x_num, 1.)
        # 重新排列张量维度，添加一个维度
        scale = rearrange(scale, '... -> ... 1')

        # 对输入进行缩放
        x = x * scale

        # 添加位置嵌入
        x = x + self.pos_emb(x, pos = pos)

        # 存储记忆令牌

        if self.has_memory_tokens:
            # 复制记忆令牌，扩展为与批次大小相同的维度
            m = repeat(self.memory_tokens, 'm d -> b m d', b = batch)
            # 打包输入张量和记忆令牌
            x, mem_ps = pack([m, x], 'b * d')

            if exists(mask):
                num_mems = m.shape[-2]
                # 在指定维度上填充掩码
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

        # 是否追加嵌入，如 PaLI 中的图像嵌入
        if exists(prepend_embeds):
            _, prepend_dim = prepend_embeds.shape[1:]
            # 断言追加的嵌入维度与模型维度相同
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            # 在指定维度上连接张量
            x = torch.cat((prepend_embeds, x), dim = -2)

        # 对输入进行嵌入层的 dropout
        x = self.emb_dropout(x)

        # 注意力层

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)

        # 分离记忆令牌

        if self.has_memory_tokens:
            m, x = unpack(x, mem_ps, 'b * d')
            intermediates.memory_tokens = m

        # 如果不返回嵌入，则生成 logits 和数值预测
        if not return_embeddings:
            logits = self.to_logits(x)
            numerical_pred = self.to_numerical_output(x)
            out = (logits, numerical_pred)
        else:
            out = x

        # 如果返回中间结果
        if return_intermediates:
            return out, intermediates

        # 如果返回记忆令牌
        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), hiddens))
            return out, new_mems

        # 如果返回注意力图
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out
class XValAutoregressiveWrapper(nn.Module):
    # 定义 XValAutoregressiveWrapper 类，继承自 nn.Module
    def __init__(
        self,
        net: XValTransformerWrapper,
        ignore_index = -100,
        pad_value = 0,
        numerical_loss_weight = 1.
    ):
        # 初始化函数，接受网络 net、ignore_index、pad_value 和 numerical_loss_weight 参数
        super().__init__()
        # 调用父类的初始化函数
        self.net = net
        # 将传入的网络赋值给对象属性 net
        self.max_seq_len = net.max_seq_len
        # 获取网络的最大序列长度
        self.numerical_loss_weight = numerical_loss_weight
        # 设置数值损失的权重
        self.ignore_index = ignore_index
        # 设置忽略的索引值

    @torch.no_grad()
    def generate(
        self,
        start_tokens: Tensor,
        start_numbers: Tensor,
        seq_len,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        **kwargs
    ):
        # 生成函数，接受起始标记、起始数字、序列长度等参数
        device = start_tokens.device
        # 获取起始标记所在设备
        was_training = self.net.training
        # 保存网络是否处于训练状态
        num_dims = len(start_tokens.shape)
        # 获取起始标记的维度数

        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'
        # 断言起始标记的维度数至少为 2
        assert start_tokens.shape == start_numbers.shape
        # 断言起始标记和起始数字的形状相同

        b, t, device = *start_tokens.shape, start_tokens.device
        # 获取起始标记的形状和设备信息
        self.net.eval()
        # 将网络设置为评估模式
        out = start_tokens
        num_out = start_numbers
        # 初始化输出和数字输出

        for _ in range(seq_len):
            # 循环生成序列
            x = out[:, -self.max_seq_len:]
            x_num = num_out[:, -self.max_seq_len:]
            # 获取最后 max_seq_len 个标记和数字

            logits, numerical_pred = self.net(x, x_num, **kwargs)
            # 使用网络生成 logits 和数值预测

            last_logits = logits[:, -1]
            last_num_pred = numerical_pred[:, -1:]
            # 获取最后一个 logits 和数值预测

            filtered_logits = filter_logits_fn(last_logits, **filter_kwargs)
            # 使用过滤函数过滤 logits

            probs = F.softmax(filtered_logits / temperature, dim=-1)
            # 计算 softmax 概率

            sample = torch.multinomial(probs, 1)
            # 从概率分布中采样一个标记

            out = torch.cat((out, sample), dim = -1)
            num_out = torch.cat((num_out, last_num_pred), dim = -1)
            # 将新生成的标记和数值添加到输出中

        out = out[:, t:]
        num_out = num_out[:, t:]
        # 去除起始标记
        is_number = out == self.net.numerical_token_id
        # 判断是否为数值标记
        num_out = torch.where(is_number, num_out, float('nan'))
        # 将非数值标记的数值设置为 NaN

        self.net.train(was_training)
        # 恢复网络的训练状态
        return GenerateReturn(out, num_out, is_number)
        # 返回生成的序列和数值信息

    def forward(
        self,
        x: Tensor,
        x_num: Tensor,
        return_loss_breakdown = False,
        **kwargs
    ):
        # 前向传播函数，接受输入 x、数值输入 x_num 和其他参数
        inp, target = x[:, :-1], x[:, 1:]
        # 获取输入和目标序列
        x_num_inp, x_num_target = x_num[:, :-1], x_num[:, 1:]
        # 获取数值输入和数值目标

        mask = kwargs.get('mask', None)
        # 获取掩码
        if exists(mask) and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask
        # 处理掩码

        logits, numerical_pred = self.net(inp, x_num_inp, **kwargs)
        # 使用网络进行前向传播

        logits = rearrange(logits, 'b n c -> b c n')
        # 重新排列 logits 的维度

        cross_entropy_loss = F.cross_entropy(logits, target, reduction = 'none', ignore_index = self.ignore_index)
        # 计算交叉熵损失

        target_mask = target != self.ignore_index
        # 创建目标掩码

        numerical_mse_loss = F.mse_loss(numerical_pred, x_num_target, reduction = 'none')
        # 计算数值均方误差损失

        numerical_mse_loss = numerical_mse_loss * target_mask
        # 根据目标掩码调整数值损失

        loss = cross_entropy_loss + numerical_mse_loss * self.numerical_loss_weight
        # 计算总损失

        if exists(mask):
            loss = loss[mask]
        # 根据掩码筛选损失

        loss = loss.mean()
        # 计算平均损失

        if not return_loss_breakdown:
            return loss
        # 如果不需要详细损失信息，直接返回总损失

        return loss, LossBreakdown(cross_entropy_loss, numerical_mse_loss)
        # 返回总损失和损失细分信息
```