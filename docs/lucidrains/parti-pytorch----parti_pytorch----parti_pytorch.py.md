# `.\lucidrains\parti-pytorch\parti_pytorch\parti_pytorch.py`

```py
# 导入所需的库
from typing import List
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from parti_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

# 辅助函数

# 检查变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 模型评估装饰器，用于在评估模式下运行模型
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 采样辅助函数

# 计算对数
def log(t, eps = 1e-20):
    return torch.log(t + eps)

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 从 Gumbel 分布中采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

# 保留前 k 个最大值，其余设为负无穷
def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 无监督分类器辅助函数

# 根据概率生成掩码
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# 归一化

# LayerNorm 模块
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 2D 相对位置偏置

class RelPosBias2d(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        arange = torch.arange(size)

        pos = torch.stack(torch.meshgrid(arange, arange, indexing = 'ij'), dim = -1)
        pos = rearrange(pos, '... c -> (...) c')
        rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

        rel_pos = rel_pos + size - 1
        h_rel, w_rel = rel_pos.unbind(dim = -1)
        pos_indices = h_rel * (2 * size - 1) + w_rel
        self.register_buffer('pos_indices', pos_indices)

    def forward(self, qk):
        i, j = qk.shape[-2:]

        bias = self.pos_bias(self.pos_indices[:i, :(j - 1)])
        bias = rearrange(bias, 'i j h -> h i j')

        bias = F.pad(bias, (j - bias.shape[-1], 0), value = 0.) # 考虑无监督分类器辅助指导的空键/值
        return bias

# 前馈网络

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden, bias = False),
        nn.GELU(),
        LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# 注意力机制

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False,
        rel_pos_bias = False,
        encoded_fmap_size = None
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置是否使用因果关系
        self.causal = causal
        # 计算缩放因子
        self.scale = dim_head ** -0.5
        # 对输入进行归一化
        self.norm = LayerNorm(dim)

        # 计算内部维度
        inner_dim = heads * dim_head
        # 设置上下文维度
        context_dim = default(context_dim, dim)
        # 如果需要对上下文进行归一化，则使用 LayerNorm，否则使用 nn.Identity()
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        # 构建查询层
        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, inner_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        # 需要用于分类器自由引导的变换器
        self.null_kv = nn.Parameter(torch.randn(dim_head))

        # 单头键/值注意力，来自 Shazeer 的多查询论文，被 Alphacode 和 PaLM 采用
        self.to_kv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, dim_head, bias = False)
        )

        # 输出层
        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias = False)
        )

        # 位置偏置
        self.rel_pos_bias = None

        # 如果需要相对位置偏置
        if rel_pos_bias:
            assert exists(encoded_fmap_size)
            # 初始化相对位置偏置
            self.rel_pos_bias = RelPosBias2d(encoded_fmap_size, heads)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        # 获取批次大小和设备信息
        batch, device = x.shape[0], x.device

        # 对输入进行归一化
        x = self.norm(x)

        # 计算查询向量
        q = self.to_q(x) * self.scale

        # 获取上下文信息
        context = default(context, x)
        context = self.norm_context(context)

        # 计算键/值对
        kv = self.to_kv(context)

        # 创建空键/值对
        null_kv = repeat(self.null_kv, 'd -> b 1 d', b = batch)
        kv = torch.cat((null_kv, kv), dim = 1)

        # 计算相似度
        sim = einsum('b h i d, b j d -> b h i j', q, kv)

        # 如果存在相对位置偏置
        if exists(self.rel_pos_bias):
            pos_bias = self.rel_pos_bias(sim)
            sim = sim + pos_bias

        # 设置掩码值
        mask_value = -torch.finfo(sim.dtype).max

        # 如果存在上下文掩码
        if exists(context_mask):
            context_mask = F.pad(context_mask, (1, 0), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        # 如果是因果关系
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # 计算注意力权重
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        # 计算输出
        out = einsum('b h i j, b j d -> b h i d', attn, kv)

        return self.to_out(out)
# 定义一个名为Parti的类，继承自nn.Module
class Parti(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        vae = None,
        vae_image_size = None,
        vae_codebook_size = None,
        t5_name = DEFAULT_T5_NAME,
        text_embed_dim = None,
        cond_drop_prob = 0.25,
        max_text_len = 128,
        ignore_index = -1
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 文本编码
        text_embed_dim = default(text_embed_dim, get_encoded_dim(t5_name))
        self.encode_texts = partial(t5_encode_text, name = t5_name)
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # 用于transformers的分类器自由引导 - @crowsonkb

        # VAE和图像处理
        assert exists(vae) ^ exists(vae_codebook_size)
        self.vae = vae

        codebook_size = default(vae_codebook_size, vae.codebook_size)
        image_size = default(vae_image_size, vae.image_size)

        self.start_token = nn.Parameter(torch.randn(dim))
        self.image_token_embed = nn.Embedding(codebook_size, dim)

        self.image_encoded_dim = vae.get_encoded_fmap_size(image_size)

        self.axial_height_pos = nn.Parameter(torch.randn(self.image_encoded_dim, dim))
        self.axial_width_pos = nn.Parameter(torch.randn(self.image_encoded_dim, dim))

        # 投影到logits
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        # 循环depth次，添加Attention、FeedForward等模块到layers中
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, causal = True, encoded_fmap_size = self.image_encoded_dim, rel_pos_bias = True, dim_head = dim_head, heads = heads, dropout = dropout),
                Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, codebook_size, bias = False)
        self.to_logits.weight = self.image_token_embed.weight

        # 默认设备
        if exists(vae):
            self.to(next(vae.parameters()).device)

        # 与损失相关
        self.ignore_index = ignore_index

    # 生成函数，用于生成图像
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        texts,
        *,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        return_pil_images = False
    ):
        device = next(self.parameters()).device

        text_token_embeds, text_mask = self.encode_texts(texts, output_device = device)

        batch = text_token_embeds.shape[0]
        image_seq_len = self.image_encoded_dim ** 2
        image_tokens = torch.empty((batch, 0), device = device, dtype = torch.long)

        # 循环生成图像序列
        for _ in range(image_seq_len):
            logits = self.forward_with_cond_scale(
                text_token_embeds = text_token_embeds,
                text_mask = text_mask,
                image_token_ids = image_tokens
            )[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            sampled = rearrange(sampled, 'b -> b 1')
            image_tokens = torch.cat((image_tokens, sampled), dim = -1)

        image_tokens = rearrange(image_tokens, 'b (h w) -> b h w', h = self.image_encoded_dim)

        # 如果没有VAE，则直接返回图像tokens
        if not exists(self.vae):
            return image_tokens

        with torch.no_grad():
            fmap = self.vae.get_fmap_from_codebook(image_tokens)
            images = self.vae.decode(fmap)

        # 如果return_pil_images为True，则返回PIL格式的图像
        if not return_pil_images:
            return images

        pil_images = list(map(T.ToPILImage(), images.unbind(dim = 0))
        return pil_images
    # 带有条件缩放的前向传播函数，根据条件缩放因子对输出进行缩放
    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        # 调用前向传播函数获取输出 logits
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        # 如果条件缩放因子为1，则直接返回 logits
        if cond_scale == 1:
            return logits

        # 否则，计算空值 logits，并返回缩放后的结果
        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    # 前向传播函数，接收文本和图像输入，返回 logits 或损失
    def forward(
        self,
        texts: List[str] = None,
        text_token_embeds = None,
        text_mask = None,
        images = None,
        image_token_ids = None,
        cond_drop_prob = None,
        return_loss = False
    ):
        # 断言文本或文本嵌入必须存在，图像或图像 token ID 必须存在
        assert exists(texts) ^ exists(text_token_embeds)
        assert exists(images) ^ exists(image_token_ids)
        # 设置条件丢弃概率为默认值或传入值
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # 编码图像

        # 如果不存在图像 token ID，则使用 VAE 对图像进行编码
        if not exists(image_token_ids):
            assert exists(self.vae), 'vae must be given if you want to encode the image live'

            with torch.no_grad():
                _, image_token_ids, _ = self.vae.encode(images, return_indices_and_loss = True)

            image_token_ids = rearrange(image_token_ids, 'b ... -> b (...)')

        # 如果需要返回损失，则截取最后一个 token 作为标签
        if return_loss:
            assert image_token_ids.shape[-1] > 1, 'not enough image tokens given to return a loss'
            image_token_ids, labels = image_token_ids[:, :-1], image_token_ids

        # 获取图像 token 嵌入
        image_token_emb = self.image_token_embed(image_token_ids)

        # 添加轴向位置嵌入

        axial_pos_emb = rearrange(self.axial_width_pos, 'w d -> 1 w d') + rearrange(self.axial_height_pos, 'h d -> h 1 d')
        axial_pos_emb = rearrange(axial_pos_emb, 'h w d -> (h w) d')

        batch, seq_len, device = *image_token_emb.shape[:2], image_token_emb.device

        image_token_emb = image_token_emb + axial_pos_emb[:seq_len]

        # 添加起始 token

        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = batch)
        image_token_emb = torch.cat((start_tokens, image_token_emb), dim = 1)

        # 文本

        # 如果不存在文本 token 嵌入，则使用编码文本函数对文本进行编码
        if not exists(text_token_embeds):
            with torch.no_grad():
                text_token_embeds, text_mask = self.encode_texts(texts, output_device = device)

        # 如果不存在文本 mask，则创建全为 True 的 mask
        if not exists(text_mask):
            text_mask = torch.ones(text_token_embeds.shape[:2], dtype = torch.bool)

        # 限制文本长度不超过最大文本长度
        text_token_embeds, text_mask = map(lambda t: t[:, :self.max_text_len], (text_token_embeds, text_mask))

        # 分类器自由引导条件丢弃

        # 如果条件丢弃概率大于0，则根据概率生成保留 mask
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        # 注意力

        x = image_token_emb
        x = self.init_norm(x)

        # 遍历每个层，依次进行自注意力、交叉注意力和前馈网络操作
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context = text_token_embeds, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # 转换为 logits

        logits = self.to_logits(x)

        # 如果不需要返回损失，则直接返回 logits
        if not return_loss:
            return logits

        # 计算交叉熵损失
        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss
```