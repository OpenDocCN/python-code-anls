# `.\lucidrains\flamingo-pytorch\flamingo_pytorch\flamingo_palm.py`

```
# 导入所需的库
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

# 导入自定义模块
from flamingo_pytorch.flamingo_pytorch import GatedCrossAttentionBlock, PerceiverResampler

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 控制在训练过程中冻结 flamingo 模型的函数

# 设置模块参数是否需要梯度
def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

# 冻结所有层
def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

# 解冻所有层
def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

# 冻结模型并设置为评估模式
def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)

# 归一化
# 使用没有偏置的层归一化，PyTorch 中没有提供这种功能

# 自定义的 LayerNorm 模块
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 残差连接

# 自定义的 Residual 模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# 旋转位置嵌入
# https://arxiv.org/abs/2104.09864

# 自定义的 RotaryEmbedding 模块
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

# 旋转半个位置嵌入
def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# 经典的 Noam Shazeer 论文，这里使用 SwiGLU 代替更流行的 GEGLU 作为前馈门控
# https://arxiv.org/abs/2002.05202

# 自定义的 SwiGLU 模块
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

# 并行注意力和前馈连接的残差块
# 由 Wang 等人和 GPT-J 的 EleutherAI 发现

# 自定义的 ParallelTransformerBlock 模块
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # 用于缓存因果掩码和旋转嵌入

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb
    def forward(self, x):
        """
        使用爱因斯坦符号表示
        b - 批次
        h - 头数
        n, i, j - 序列长度（基本序列长度，源，目标）
        d - 特征维度
        """

        n, device, h = x.shape[1], x.device, self.heads

        # 预先 Layernorm 处理

        x = self.norm(x)

        # 注意力查询、键、值和前馈内部

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # 分割头部
        # 他们使用多查询单键值注意力，另一篇 Noam Shazeer 的论文
        # 他们发现在一定规模之后没有性能损失，并且解码更有效率
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # 旋转嵌入

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # 缩放

        q = q * self.scale

        # 相似度

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # 因果掩码

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # 聚合值

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # 合并头部

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)
# transformer

# 定义一个名为FlamingoPaLM的神经网络模块
class FlamingoPaLM(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 特征维度
        num_tokens,  # 标记数量
        depth,  # 深度
        dim_head=64,  # 头部维度
        heads=8,  # 头部数量
        ff_mult=4,  # FeedForward模块的倍增因子
        media_token_id=3,  # 媒体标记ID
        cross_attn_every=3,  # 每隔多少层进行交叉注意力
        img_encoder=None,  # 图像编码器
        perceiver_num_latents=64,  # 感知器潜在特征数量
        perceiver_depth=2,  # 感知器深度
        max_video_frames = None,  # 最大视频帧数
        only_attend_immediate_media=True  # 是否只关注即时媒体
    ):
        super().__init__()

        # 标记嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 媒体标记ID，需要为媒体保留一个特殊的标记ID
        self.media_token_id = media_token_id

        # 视频帧位置嵌入
        self.video_frame_pos_emb = nn.Parameter(torch.randn(max_video_frames, dim)) if exists(max_video_frames) else None

        # 图像编码器
        self.img_encoder = img_encoder
        # 冻结图像编码器并设置为评估模式
        freeze_model_and_make_eval_(self.img_encoder)

        # 感知器重采样器
        self.perceiver_resampler = PerceiverResampler(
            dim=dim,
            depth=perceiver_depth,
            dim_head=dim_head,
            heads=heads,
            num_latents=perceiver_num_latents
        )

        # 层列表
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                GatedCrossAttentionBlock(dim=dim, dim_head=dim_head, heads=heads, only_attend_immediate_media=only_attend_immediate_media) if not (ind % cross_attn_every) else None
            ]))

        # 转换为logits
        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

        # 使用嵌入权重来绑定投影到logits，这种方式不常见，但有效
        self.to_logits[-1].weight = self.token_emb.weight
        # 初始化嵌入权重
        nn.init.normal_(self.token_emb.weight, std=0.02)
    
    # 前向传播函数
    def forward(
        self,
        text,
        *,
        images=None,
        videos=None,
        embeds=None
        ):
            # 获取文本的批次大小和设备信息
            batch, device = text.shape[0], text.device

            # 判断是否处于flamingo模式
            flamingo_mode = any([exists(t) for t in (images, videos, embeds)])

            # 根据传入的参数自动决定冻结或解冻层
            if flamingo_mode:
                # 在flamingo模式下，冻结除了perceiver和gated cross attention之外的所有层
                freeze_all_layers_(self)
                unfreeze_all_layers_(self.perceiver_resampler)
                [unfreeze_all_layers_(cross_attn) for _, cross_attn in self.layers if exists(cross_attn)]
            else:
                # 解冻所有层
                unfreeze_all_layers_(self)

            # 推导媒体令牌的ID（作为布尔张量），用于计算掩码交叉注意力
            if flamingo_mode:
                media_locations = text == self.media_token_id

            # 对文本令牌进行编码
            text_tokens = self.token_emb(text)

            # 断言不存在embeds并且存在images或video
            assert not (exists(embeds) and (exists(images) or exists(video)))

            # 将视频或图像编码为嵌入
            # 使用在init中传入的img_encoder
            # 也可以接受预先计算的图像嵌入
            if exists(images):
                assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
                images = rearrange(images, 'b t ... -> (b t) ...')

                with torch.no_grad():
                    embeds = self.img_encoder(images)

                embeds = rearrange(embeds, '(b t) ... -> b t ...', b = batch)

            if exists(videos):
                assert exists(self.img_encoder), 'img_encoder must be passed in for automatic video encoding'
                batch, media, num_times, *_ = videos.shape
                videos = rearrange(videos, '... c h w -> (...) c h w')

                with torch.no_grad():
                    embeds = self.img_encoder(videos)

                embeds = rearrange(embeds, '(b m t) ... -> b m t ...', b = batch, m = media, t = num_times)

                video_time_pos_emb = repeat(self.video_frame_pos_emb[:num_times], 't d -> b m t n d', b = batch, m = media, n = embeds.shape[-2])
                embeds = embeds + video_time_pos_emb
                embeds = rearrange(embeds, 'b m t n d -> b m (t n) d')

            if exists(embeds):
                embeds = self.perceiver_resampler(embeds)

            # 遍历层
            for attn_ff, flamingo_cross_attn in self.layers:
                text_tokens = attn_ff(text_tokens)

                # 如果存在图像嵌入并且为该层设置了flamingo交叉注意力，则进行交叉注意力
                if exists(flamingo_cross_attn) and exists(embeds):
                    text_tokens = flamingo_cross_attn(
                        text_tokens,
                        embeds,
                        media_locations = media_locations
                    )

            return self.to_logits(text_tokens)
```