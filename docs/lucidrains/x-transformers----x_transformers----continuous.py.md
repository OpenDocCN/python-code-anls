# `.\lucidrains\x-transformers\x_transformers\continuous.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn.functional 中导入 F 模块
import torch.nn.functional as F

# 从 einops 库中导入 pack, repeat, unpack 函数
from einops import pack, repeat, unpack

# 从 x_transformers.x_transformers 模块中导入以下类和函数
from x_transformers.x_transformers import (
    AttentionLayers,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding,
    LayerNorm,
    always,
    pad_at_dim
)

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

# 连续变换器包装器类
class ContinuousTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        attn_layers: AttentionLayers,
        dim_in = None,
        dim_out = None,
        emb_dim = None,
        max_mem_len = 0,
        num_memory_tokens = None,
        post_emb_norm = False,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False
    ):
        super().__init__()
        dim = attn_layers.dim

        # 最大序列长度
        self.max_seq_len = max_seq_len

        # 最大记忆长度
        self.max_mem_len = max_mem_len
        
        # 没有绝对位置编码
        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        if no_abs_pos_emb:
            # 如果没有绝对位置编码，则位置编码为常数 0
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            # 如果使用缩放的正弦位置编码，则创建 ScaledSinusoidalEmbedding 对象
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            # 否则创建 AbsolutePositionalEmbedding 对象
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        # 后置嵌入层归一化
        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        # 嵌入层丢弃
        self.emb_dropout = nn.Dropout(emb_dropout)

        # 记忆令牌

        # 默认记忆令牌数量为 0
        num_memory_tokens = default(num_memory_tokens, 0)
        # 是否有记忆令牌
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            # 如果有记忆令牌，则创建 nn.Parameter 对象
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # 注意力层

        # 设置注意力层
        self.attn_layers = attn_layers

        # 投影输入和输出

        # 输入投影层
        self.project_in = nn.Linear(dim_in, dim, bias = False) if exists(dim_in) else nn.Identity()
        # 输出投影层
        self.project_out = nn.Linear(dim, dim_out, bias = False) if exists(dim_out) else nn.Identity()

    def forward(
        self,
        x,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        return_attn = False,
        mems = None,
        mem_masks = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        **kwargs
        ):
        # 解包输入张量 x 的形状，得到 batch, seq, device
        batch, seq, device = *x.shape[:2], x.device

        # 对输入张量进行投影
        x = self.project_in(x)
        # 添加位置编码
        x = x + self.pos_emb(x, pos = pos)

        # 对投影后的张量进行归一化
        x = self.post_emb_norm(x)

        # 处理记忆令牌

        if self.has_memory_tokens:
            # 重复记忆令牌，扩展为 batch 维度
            m = repeat(self.memory_tokens, 'm d -> b m d', b = batch)
            # 打包记忆令牌和输入张量
            x, mem_ps = pack([m, x], 'b * d')

            # 如果存在 mask，则对 mask 进行处理
            if exists(mask):
                num_mems = m.shape[-2]
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

        # 是否追加嵌入，如 PaLI 中的图像嵌入

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]

            # 断言追加的嵌入维度与模型维度相同
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            # 在指定维度上连接张量
            x = torch.cat((prepend_embeds, x), dim = -2)

            # 如果存在 prepend_mask 或 mask，则对 mask 进行处理
            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((batch, seq), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((batch, prepend_seq), device = device, dtype = torch.bool))

                mask = torch.cat((prepend_mask, mask), dim = -1)

        # 对嵌入张量进行 dropout
        x = self.emb_dropout(x)

        # 注意力层

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, return_hiddens = True, **kwargs)

        # 剥离记忆令牌

        if self.has_memory_tokens:
            m, x = unpack(x, mem_ps, 'b * d')
            intermediates.memory_tokens = m

        # 输出结果
        out = self.project_out(x) if not return_embeddings else x

        # 如果需要返回中间结果
        if return_intermediates:
            return out, intermediates

        # 如果需要返回记忆
        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), hiddens))
            return out, new_mems

        # 如果需要返回注意力图
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out
# 定义一个连续自回归包装器类，继承自 nn.Module
class ContinuousAutoregressiveWrapper(nn.Module):
    # 初始化方法
    def __init__(
        self,
        net: ContinuousTransformerWrapper,  # 接收一个 ContinuousTransformerWrapper 类型的网络
        ignore_index = -100,  # 忽略索引，默认为 -100
        pad_value = 0,  # 填充值，默认为 0
        loss_fn = nn.MSELoss(reduction = 'none')  # 损失函数，默认为均方误差损失
    ):
        super().__init__()  # 调用父类的初始化方法
        self.net = net  # 将传入的网络赋值给属性
        self.max_seq_len = net.max_seq_len  # 获取网络的最大序列长度
        self.loss_fn = loss_fn  # 将传入的损失函数赋值给属性

    # 生成方法，不计算梯度
    @torch.no_grad()
    def generate(self, start_tokens, seq_len, **kwargs):
        device = start_tokens.device  # 获取起始标记的设备
        was_training = self.net.training  # 记录网络是否在训练状态
        num_dims = len(start_tokens.shape)  # 获取起始标记的维度数

        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'  # 断言起始标记的维度数必须大于等于2

        if num_dims == 2:
            start_tokens = start_tokens[None, :]  # 如果维度数为2，则在第一维度上添加一个维度

        b, t, _, device = *start_tokens.shape, start_tokens.device  # 获取起始标记的形状和设备

        self.net.eval()  # 将网络设置为评估模式
        out = start_tokens  # 初始化输出为起始标记

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]  # 获取最后 self.max_seq_len 个标记

            last = self.net(x, **kwargs)[:, -1:]  # 使用网络生成下一个标记
            out = torch.cat((out, last), dim = -2)  # 将生成的标记拼接到输出中

        out = out[:, t:]  # 去掉起始标记

        if num_dims == 2:
            out = out.squeeze(0)  # 如果维度数为2，则去掉第一维度

        self.net.train(was_training)  # 恢复网络的训练状态
        return out  # 返回生成的序列

    # 前向传播方法
    def forward(self, x, **kwargs):
        inp, target = x[:, :-1], x[:, 1:]  # 获取输入和目标序列

        assert 'prepend_embeds' not in kwargs  # 断言不应该传入 'prepend_embeds' 参数

        mask = kwargs.get('mask', None)  # 获取掩码，如果不存在则为 None
        if exists(mask) and mask.shape[1] == x.shape[1]:  # 如果掩码存在且与输入序列长度相同
            mask = mask[:, :-1]  # 去掉最后一个标记的掩码
            kwargs['mask'] = mask  # 更新 kwargs 中的掩码

        out = self.net(inp, **kwargs)  # 使用网络进行前向传播

        loss = self.loss_fn(out, target)  # 计算损失

        if exists(mask):  # 如果掩码存在
            assert loss.ndim > 1, 'loss should not be reduced if mask is passed in'  # 断言如果传入掩码，则损失不应该被减少
            loss = loss[mask]  # 根据掩码获取损失

        return loss.mean()  # 返回损失的平均值
```