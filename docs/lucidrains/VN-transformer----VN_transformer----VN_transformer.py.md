# `.\lucidrains\VN-transformer\VN_transformer\VN_transformer.py`

```py
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 torch 中导入 nn, einsum, Tensor
from torch import nn, einsum, Tensor

# 从 einops 中导入 rearrange, repeat, reduce
from einops import rearrange, repeat, reduce
# 从 einops.layers.torch 中导入 Rearrange, Reduce
from einops.layers.torch import Rearrange, Reduce
# 从 VN_transformer.attend 中导入 Attend

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 计算两个向量的内积
def inner_dot_product(x, y, *, dim = -1, keepdim = True):
    return (x * y).sum(dim = dim, keepdim = keepdim)

# layernorm

# LayerNorm 类
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# equivariant modules

# VNLinear 类
class VNLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        bias_epsilon = 0.
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))

        self.bias = None
        self.bias_epsilon = bias_epsilon

        # 在这篇论文中，他们提出使用一个小偏置进行准等变性，通过 epsilon 可控，他们声称这样可以获得更好的稳定性和结果

        if bias_epsilon > 0.:
            self.bias = nn.Parameter(torch.randn(dim_out))

    def forward(self, x):
        out = einsum('... i c, o i -> ... o c', x, self.weight)

        if exists(self.bias):
            bias = F.normalize(self.bias, dim = -1) * self.bias_epsilon
            out = out + rearrange(bias, '... -> ... 1')

        return out

# VNReLU 类
class VNReLU(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.randn(dim, dim))
        self.U = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        q = einsum('... i c, o i -> ... o c', x, self.W)
        k = einsum('... i c, o i -> ... o c', x, self.U)

        qk = inner_dot_product(q, k)

        k_norm = k.norm(dim = -1, keepdim = True).clamp(min = self.eps)
        q_projected_on_k = q - inner_dot_product(q, k / k_norm) * k

        out = torch.where(
            qk >= 0.,
            q,
            q_projected_on_k
        )

        return out

# VNAttention 类
class VNAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash = False,
        num_latents = None   # 设置此参数将启用类似于 perceiver 的跨注意力机制，从潜在变量到序列，潜在变量由 VNWeightedPool 推导而来
    ):
        super().__init__()
        assert not (l2_dist_attn and flash), 'l2 distance attention is not compatible with flash attention'

        self.scale = (dim_coor * dim_head) ** -0.5
        dim_inner = dim_head * heads
        self.heads = heads

        self.to_q_input = None
        if exists(num_latents):
            self.to_q_input = VNWeightedPool(dim, num_pooled_tokens = num_latents, squeeze_out_pooled_dim = False)

        self.to_q = VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon)
        self.to_k = VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon)
        self.to_v = VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon)
        self.to_out = VNLinear(dim_inner, dim, bias_epsilon = bias_epsilon)

        if l2_dist_attn and not exists(num_latents):
            # 对于 l2 距离注意力，查询和键是相同的，不是 perceiver-like 注意力
            self.to_k = self.to_q

        self.attend = Attend(flash = flash, l2_dist = l2_dist_attn)
    # 定义一个前向传播函数，接受输入 x 和可选的 mask 参数
    def forward(self, x, mask = None):
        """
        einstein notation
        b - batch
        n - sequence
        h - heads
        d - feature dimension (channels)
        c - coordinate dimension (3 for 3d space)
        i - source sequence dimension
        j - target sequence dimension
        """

        # 获取输入 x 的最后一个维度，即特征维度的大小
        c = x.shape[-1]

        # 如果存在 self.to_q_input 方法，则使用该方法处理输入 x 和 mask，否则直接使用 x
        if exists(self.to_q_input):
            q_input = self.to_q_input(x, mask = mask)
        else:
            q_input = x

        # 分别通过 self.to_q、self.to_k、self.to_v 方法处理 q_input，得到 q、k、v
        q, k, v = self.to_q(q_input), self.to_k(x), self.to_v(x)
        # 将 q、k、v 重排维度，将其转换为 'b h n (d c)' 的形式
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) c -> b h n (d c)', h = self.heads), (q, k, v))

        # 调用 attend 方法进行注意力计算
        out = self.attend(q, k, v, mask = mask)

        # 将输出 out 重排维度，将其转换为 'b n (h d) c' 的形式
        out = rearrange(out, 'b h n (d c) -> b n (h d) c', c = c)
        # 返回处理后的输出结果
        return self.to_out(out)
# 定义一个 VNFeedForward 类，包含线性层、ReLU 激活函数和另一个线性层
def VNFeedForward(dim, mult = 4, bias_epsilon = 0.):
    # 计算内部维度
    dim_inner = int(dim * mult)
    # 返回一个包含上述三个层的序列模块
    return nn.Sequential(
        VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon),  # VNLinear 线性层
        VNReLU(dim_inner),  # VNReLU 激活函数
        VNLinear(dim_inner, dim, bias_epsilon = bias_epsilon)  # 另一个 VNLinear 线性层
    )

# 定义一个 VNLayerNorm 类，包含 LayerNorm 层
class VNLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)  # LayerNorm 层

    def forward(self, x):
        norms = x.norm(dim = -1)
        x = x / rearrange(norms.clamp(min = self.eps), '... -> ... 1')
        ln_out = self.ln(norms)
        return x * rearrange(ln_out, '... -> ... 1')

# 定义一个 VNWeightedPool 类，包含权重参数和池化操作
class VNWeightedPool(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        num_pooled_tokens = 1,
        squeeze_out_pooled_dim = True
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.weight = nn.Parameter(torch.randn(num_pooled_tokens, dim, dim_out))  # 权重参数
        self.squeeze_out_pooled_dim = num_pooled_tokens == 1 and squeeze_out_pooled_dim

    def forward(self, x, mask = None):
        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            x = x.masked_fill(~mask, 0.)
            numer = reduce(x, 'b n d c -> b d c', 'sum')
            denom = mask.sum(dim = 1)
            mean_pooled = numer / denom.clamp(min = 1e-6)
        else:
            mean_pooled = reduce(x, 'b n d c -> b d c', 'mean')

        out = einsum('b d c, m d e -> b m e c', mean_pooled, self.weight)

        if not self.squeeze_out_pooled_dim:
            return out

        out = rearrange(out, 'b 1 d c -> b d c')
        return out

# 定义一个 VNTransformerEncoder 类，包含多层 VNAttention、VNLayerNorm 和 VNFeedForward
class VNTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        ff_mult = 4,
        final_norm = False,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash_attn = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_coor = dim_coor

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                VNAttention(dim = dim, dim_head = dim_head, heads = heads, bias_epsilon = bias_epsilon, l2_dist_attn = l2_dist_attn, flash = flash_attn),  # VNAttention 层
                VNLayerNorm(dim),  # VNLayerNorm 层
                VNFeedForward(dim = dim, mult = ff_mult, bias_epsilon = bias_epsilon),  # VNFeedForward 层
                VNLayerNorm(dim)  # 另一个 VNLayerNorm 层
            ]))

        self.norm = VNLayerNorm(dim) if final_norm else nn.Identity()

    def forward(
        self,
        x,
        mask = None
    ):
        *_, d, c = x.shape

        assert x.ndim == 4 and d == self.dim and c == self.dim_coor, 'input needs to be in the shape of (batch, seq, dim ({self.dim}), coordinate dim ({self.dim_coor}))'

        for attn, attn_post_ln, ff, ff_post_ln in self.layers:
            x = attn_post_ln(attn(x, mask = mask)) + x
            x = ff_post_ln(ff(x)) + x

        return self.norm(x)

# 定义一个 VNInvariant 类，包含 MLP 模块
class VNInvariant(nn.Module):
    def __init__(
        self,
        dim,
        dim_coor = 3,

    ):
        super().__init__()
        self.mlp = nn.Sequential(
            VNLinear(dim, dim_coor),  # VNLinear 线性层
            VNReLU(dim_coor),  # VNReLU 激活函数
            Rearrange('... d e -> ... e d')  # 重新排列维度
        )

    def forward(self, x):
        return einsum('b n d i, b n i o -> b n o', x, self.mlp(x))

# 定义一个 VNTransformer 类，包含多个参数和模块
class VNTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens = None,
        dim_feat = None,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        reduce_dim_out = True,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash_attn = False,
        translation_equivariance = False,
        translation_invariant = False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 如果 num_tokens 存在，则创建一个维度为 dim 的嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        # 设置特征维度为 dim_feat 或默认为 0
        dim_feat = default(dim_feat, 0)
        self.dim_feat = dim_feat
        # 计算坐标总维度，包括坐标和特征
        self.dim_coor_total = dim_coor + dim_feat

        # 确保平移等变性和平移不变性最多只能有一个为真
        assert (int(translation_equivariance) + int(translation_invariant)) <= 1
        self.translation_equivariance = translation_equivariance
        self.translation_invariant = translation_invariant

        # 定义输入投影层
        self.vn_proj_in = nn.Sequential(
            Rearrange('... c -> ... 1 c'),
            VNLinear(1, dim, bias_epsilon = bias_epsilon)
        )

        # 创建 VNTransformerEncoder 编码器
        self.encoder = VNTransformerEncoder(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            bias_epsilon = bias_epsilon,
            dim_coor = self.dim_coor_total,
            l2_dist_attn = l2_dist_attn,
            flash_attn = flash_attn
        )

        # 如果需要减少输出维度，则定义输出投影层
        if reduce_dim_out:
            self.vn_proj_out = nn.Sequential(
                VNLayerNorm(dim),
                VNLinear(dim, 1, bias_epsilon = bias_epsilon),
                Rearrange('... 1 c -> ... c')
            )
        else:
            self.vn_proj_out = nn.Identity()

    def forward(
        self,
        coors,
        *,
        feats = None,
        mask = None,
        return_concatted_coors_and_feats = False
    ):
        # 如果需要平移等变性或平移不变性，则计算坐标的平均值并减去
        if self.translation_equivariance or self.translation_invariant:
            coors_mean = reduce(coors, '... c -> c', 'mean')
            coors = coors - coors_mean

        x = coors

        # 如果存在特征，则将特征拼接到坐标中
        if exists(feats):
            if feats.dtype == torch.long:
                assert exists(self.token_emb), 'num_tokens must be given to the VNTransformer (to build the Embedding), if the features are to be given as indices'
                feats = self.token_emb(feats)

            assert feats.shape[-1] == self.dim_feat, f'dim_feat should be set to {feats.shape[-1]}'
            x = torch.cat((x, feats), dim = -1)

        assert x.shape[-1] == self.dim_coor_total

        # 输入投影层
        x = self.vn_proj_in(x)
        # 编码器
        x = self.encoder(x, mask = mask)
        # 输出投影层
        x = self.vn_proj_out(x)

        # 提取坐标和特征
        coors_out, feats_out = x[..., :3], x[..., 3:]

        # 如果需要平移等变性，则将坐标输出加上坐标平均值
        if self.translation_equivariance:
            coors_out = coors_out + coors_mean

        # 如果没有特征，则返回坐标输出
        if not exists(feats):
            return coors_out

        # 如果需要返回拼接的坐标和特征，则返回拼接后的结果
        if return_concatted_coors_and_feats:
            return torch.cat((coors_out, feats_out), dim = -1)

        # 否则返回坐标和特征分开的结果
        return coors_out, feats_out
```