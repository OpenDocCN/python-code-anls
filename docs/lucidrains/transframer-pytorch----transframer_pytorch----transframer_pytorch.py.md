# `.\lucidrains\transframer-pytorch\transframer_pytorch\transframer_pytorch.py`

```py
# 从 math 模块中导入 sqrt 和 pi 函数
# 从 functools 模块中导入 partial 函数
import torch
# 从 torch.nn.functional 模块中导入 F
import torch.nn.functional as F
# 从 torch.fft 模块中导入 fft 和 irfft 函数
from torch.fft import fft, irfft
# 从 torch 模块中导入 nn 和 einsum 函数
from torch import nn, einsum
# 从 einops 模块中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 kornia.color.ycbcr 模块中导入 rgb_to_ycbcr 和 ycbcr_to_rgb 函数

# helpers

# 定义 exists 函数，判断值是否存在
def exists(val):
    return val is not None

# 定义 default 函数，如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# tensor helpers

# 定义 l2norm 函数，对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# dct related encoding / decoding functions

# 定义 dct 函数，进行离散余弦变换
# 函数来源于 https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
# 修复了大多数 torch 版本 > 1.9 的问题，使用最新的 fft 和 irfft
def dct(x, norm = None):
    shape, dtype, device = x.shape, x.dtype, x.device
    N = shape[-1]

    x = rearrange(x.contiguous(), '... n -> (...) n')

    v = torch.cat([x[:, ::2], x[:, 1::2].flip((1,))], dim = 1)

    vc = torch.view_as_real(fft(v, dim=1))

    k = -torch.arange(N, dtype = dtype, device = device) * pi / (2 * N)
    k = rearrange(k, 'n -> 1 n')

    v = vc[:, :, 0] * k.cos() - vc[:, :, 1] * k.sin()

    if norm == 'ortho':
        v[:, 0] /= sqrt(N) * 2
        v[:, 1:] /= sqrt(N / 2) * 2

    v *= 2
    return v.view(*shape)

# 定义 idct 函数，进行逆离散余弦变换
def idct(x, norm = None):
    shape, dtype, device = x.shape, x.dtype, x.device
    N = shape[-1]

    x_v = rearrange(x.contiguous(), '... n -> (...) n') / 2

    if norm == 'ortho':
        x_v[:, 0] *= sqrt(N) * 2
        x_v[:, 1:] *= sqrt(N / 2) * 2

    k = torch.arange(N, dtype = dtype, device = device) * pi / (2 * N)
    k = rearrange(k, 'n -> 1 n')
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    v_t_r = x_v
    v_t_i = torch.cat([x_v[:, :1] * 0, -x_v.flip((1,))[:, :-1]], dim = 1)

    v_r = v_t_r * w_r - v_t_i * w_i
    v_i = v_t_r * w_i + v_t_i * w_r

    v = torch.stack((v_r, v_i), dim = -1)

    v = irfft(torch.view_as_complex(v), n = N, dim = 1)
    x = torch.zeros_like(v)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip((1,))[:, :N // 2]

    return x.view(*shape)

# 定义 dct_2d 函数，对二维张量进行离散余弦变换
def dct_2d(x, norm = None):
    dct_ = partial(dct, norm = norm)
    x1 = dct_(x)
    x2 = dct_(rearrange(x1, '... h w -> ...  w h'))
    return rearrange(x2, '... h w -> ... w h')

# 定义 idct_2d 函数，对二维张量进行逆离散余弦变换
def idct_2d(x, norm = None):
    idct_ = partial(idct, norm = norm)
    x1 = idct_(x)
    x2 = idct_(rearrange(x1, '... h w -> ... w h'))
    return rearrange(x2, '... h w -> ... w h')

# 定义 blockify 函数，将张量分块
def blockify(x, block_size = 8):
    assert block_size in {8, 16}
    return rearrange(x, 'b c (h bs1) (w bs2) -> (b h w) c bs1 bs2', bs1 = block_size, bs2 = block_size)

# 定义 deblockify 函数，将分块的张量还原为原始形状
def deblockify(x, h, w, block_size = 8):
    assert block_size in {8, 16}
    return rearrange(x, '(b h w) c bs1 bs2 -> b c (h bs1) (w bs2)', h = h, w = w)

# final functions from rgb -> dct and back

# 定义 images_to_dct 函数，将图像转换为离散余弦变换
def images_to_dct(images):
    raise NotImplementedError

# 定义 dct_to_images 函数，将离散余弦��换转换为图像
def dct_to_images(images):
    raise NotImplementedError

# feedforward

# 定义 FeedForward 类，包含线性层和 GELU 激活函数
def FeedForward(
    dim,
    *,
    mult = 4.
):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.LayerNorm(inner_dim),  # from normformer paper
        nn.Linear(inner_dim, dim, bias = False)
    )

# attention, what else?
# here we will use one headed key / values (as described in paper, from Noam Shazeer) - along with cosine sim attention

# 定义 Attention 类，包含多头注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        scale = 10,
        causal = False,
        norm_context = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(dim_head * heads, dim, bias = False)
    # 定义一个前向传播函数，接受输入 x，上下文 context 和上下文掩码 context_mask
    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        # 获取头数 h，缩放因子 scale，是否因果 causal，设备信息 device
        h, scale, causal, device = self.heads, self.scale, self.causal, x.device

        # 对输入 x 进行归一化处理
        x = self.norm(x)

        # 如果存在上下文 context，则使用上下文，否则使用输入 x 作为上下文
        context = default(context, x)

        # 将输入 x 转换为查询向量 q，并重新排列维度
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        # 如果存在上下文，则对上下文进行归一化处理
        if exists(context):
            context = self.norm_context(context)

        # 将上下文转换为键值对 k, v，并按最后一个维度分割成两部分
        k, v = self.to_kv(context).chunk(2, dim = -1)

        # 对查询向量 q 和键向量 k 进行 L2 归一化
        q, k = map(l2norm, (q, k))

        # 计算查询向量 q 和键向量 k 之间的相似度矩阵 sim
        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        # 计算掩码值，用于在相似度矩阵中进行掩码操作
        mask_value = -torch.finfo(sim.dtype).max

        # 如果存在上下文掩码，则对相似度矩阵进行掩码操作
        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(context_mask, mask_value)

        # 如果是因果注意力机制，则对相似度矩阵进行因果掩码操作
        if causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # 对相似度矩阵进行 softmax 操作，得到注意力权重
        attn = sim.softmax(dim = -1)

        # 根据注意力权重计算输出向量 out
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # 重新排列输出向量的维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 返回输出向量
        return self.to_out(out)
# 定义一个名为 Block 的类，继承自 nn.Module
class Block(nn.Module):
    # 初始化函数，接受输入维度 dim、输出维度 dim_out 和分组数 groups
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8
    ):
        super().__init__()
        # 创建一个卷积层，输入维度为 dim，输出维度为 dim_out，卷积核大小为 3，填充为 1
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        # 创建一个 GroupNorm 层，分组数为 groups，输出维度为 dim_out
        self.norm = nn.GroupNorm(groups, dim_out)
        # 创建一个 SiLU 激活函数层
        self.act = nn.SiLU()

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 对输入 x 进行卷积操作
        x = self.proj(x)
        # 对卷积结果进行 GroupNorm 操作
        x = self.norm(x)
        # 对 GroupNorm 结果进行 SiLU 激活函数操作
        return self.act(x)

# 定义一个名为 ResnetBlock 的类，继承自 nn.Module
class ResnetBlock(nn.Module):
    # 初始化函数，接受输入维度 dim、输出维度 dim_out 和分组数 groups
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8
    ):
        super().__init__()
        # 创建两个 Block 实例，分别作为 ResNet 块的两个子块
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        # 如果输入维度和输出维度不相等，则创建一个卷积层，否则创建一个恒等映射层
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 对输入 x 进行第一个子块的操作
        h = self.block1(x)
        # 对第一个子块的输出进行第二个子块的操作
        h = self.block2(h)
        # 返回第一个子块的输出与输入 x 经过卷积的结果的和
        return h + self.res_conv(x)

# 定义一个名为 UnetTransformerBlock 的类，继承自 nn.Module
class UnetTransformerBlock(nn.Module):
    # 初始化函数，接受输入维度 dim、注意力头维度 dim_head 和注意力头数 heads
    def __init__(
        self,
        dim,
        *,
        dim_head = 32,
        heads = 8
    ):
        super().__init__()
        # 创建一个 Attention 层，输入维度为 dim，注意力头维度为 dim_head，注意力头数为 heads
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads)
        # 创建一个 FeedForward 层，输入维度为 dim
        self.ff = FeedForward(dim = dim)

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 保存输入 x 的原始形状
        orig_shape = x.shape
        # 将输入 x 重排列为 'b c ...' 的形式
        x = rearrange(x, 'b c ... -> b (...) c')

        # 对输入 x 进行注意力操作并加上原始输入 x
        x = self.attn(x) + x
        # 对加上注意力结果的 x 进行 FeedForward 操作并加上原始输入 x
        x = self.ff(x) + x

        # 将 x 重排列为 'b n c' 的形式，再将其形状恢复为原始形状
        x = rearrange(x, 'b n c -> b c n')
        return x.reshape(*orig_shape)

# 定义一个名为 Unet 的类，继承自 nn.Module
class Unet(nn.Module):
    # 初始化函数，接受输入维度 dim、输出维度 dim_out、注意力参数 attn_kwargs
    def __init__(
        self,
        dim,
        *,
        dim_mults = (1, 2, 3, 4),
        dim_out,
        **attn_kwargs
    ):
        super().__init__()
        # 创建一个输出维度为 dim_out 的卷积层
        self.to_out = nn.Conv2d(dim, dim_out, 1)
        # 计算多层次维度倍增后的维度列表 dims
        dims = [dim, *map(lambda t: t * dim, dim_mults)]
        # 计算每一层次的维度对 dim_pairs
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        # 中间维度为 dims 的最后一个元素
        mid_dim = dims[-1]

        # 创建下采样和上采样的模块列表
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # 创建中间的 ResNet 块
        self.mid = ResnetBlock(mid_dim, mid_dim)

        # 遍历每一层次的维度对
        for dim_in, dim_out in dim_pairs:
            # 对每一层次创建下采样模块列表
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in),
                UnetTransformerBlock(dim_in, **attn_kwargs),
                nn.Conv2d(dim_in, dim_out, 3, 2, 1)
            ]))

            # 对每一层次创建上采样模块列表
            self.ups.insert(0, nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_out),
                UnetTransformerBlock(dim_out, **attn_kwargs),
                nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1)
            ]))

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 保存每个下采样阶段的隐藏状态
        hiddens = []

        # 对每个下采样阶段的模块进行操作
        for block, attn_block, downsample in self.downs:
            x = block(x)
            x = attn_block(x)
            x = downsample(x)
            hiddens.append(x)

        # 对中间的 ResNet 块进行操作
        x = self.mid(x)

        # 对每个上采样阶段的模块进行操作
        for block, attn_block, upsample in self.ups:
            x = torch.cat((x, hiddens.pop()), dim = 1)
            x = block(x)
            x = attn_block(x)
            x = upsample(x)

        # 对输出进行卷积操作并重排列输出形状
        out = self.to_out(x)
        return rearrange(out, 'b c h w -> b (h w) c')

# 定义一个名为 Transframer 的类，继承自 nn.Module
class Transframer(nn.Module):
    # 初始化函数，接受参数 unet、dim、depth、max_channels、max_positions、max_values、image_size、block_size、dim_head、heads、ff_mult 和 ignore_index
    def __init__(
        self,
        *,
        unet: Unet,
        dim,
        depth,
        max_channels,
        max_positions,
        max_values,
        image_size,
        block_size = 8,
        dim_head = 32,
        heads = 8,
        ff_mult = 4.,
        ignore_index = -100
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化 UNet 模型
        self.unet = unet

        # 初始化起始标记
        self.start_token = nn.Parameter(torch.randn(dim))

        # 初始化块位置嵌入
        self.block_pos_emb = nn.Parameter(torch.randn(2, (image_size // block_size), dim))

        # 初始化通道嵌入
        self.channels = nn.Embedding(max_channels, dim)
        # 初始化位置嵌入
        self.positions = nn.Embedding(max_positions, dim)
        # 初始化值嵌入
        self.values = nn.Embedding(max_values, dim)

        # 初始化后处理层的 LayerNorm
        self.postemb_norm = nn.LayerNorm(dim) # 在 Bloom 和 YaLM 中为了稳定性而完成

        # 初始化层列表
        self.layers = nn.ModuleList([])

        # 循环创建深度个层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads, causal = True),
                Attention(dim, dim_head = dim_head, heads = heads, norm_context = True),
                FeedForward(dim, mult = ff_mult)
            ]))

        # 初始化最终层的 LayerNorm
        self.final_norm = nn.LayerNorm(dim)

        # 为最终预测给通道和位置提供单独的嵌入

        # 初始化轴向通道嵌入
        self.axial_channels = nn.Embedding(max_channels, dim)
        # 初始化轴向位置嵌入
        self.axial_positions = nn.Embedding(max_positions, dim)

        # 初始化轴向注意力机制
        self.axial_attn = Attention(dim, dim_head = dim_head,  heads = heads, causal = True)
        # 初始化轴向前馈网络
        self.axial_ff = FeedForward(dim, mult = ff_mult)

        # 初始化轴向最终层的 LayerNorm
        self.axial_final_norm = nn.LayerNorm(dim)

        # 投影到逻辑回归

        # 线性变换到通道的逻辑回归
        self.to_channel_logits = nn.Linear(dim, max_channels)
        # 线性变换到位置的逻辑回归
        self.to_position_logits = nn.Linear(dim, max_positions)
        # 线性变换到值的逻辑回归
        self.to_value_logits = nn.Linear(dim, max_values)

        # 设置忽略索引
        self.ignore_index = ignore_index

    # 获取块位置嵌入
    def get_block_pos_emb(self):
        block_pos_emb_h, block_pos_emb_w = self.block_pos_emb.unbind(dim = 0)
        block_pos_emb = rearrange(block_pos_emb_h, 'h d -> h 1 d') + rearrange(block_pos_emb_w, 'w d -> 1 w d')
        return rearrange(block_pos_emb, '... d -> (...) d')

    # 前向传播���数
    def forward(
        self,
        x,
        context_frames,
        return_loss = False
        ):
        # 断言输入张量 x 的最后一个维度为 3
        assert x.shape[-1] == 3

        # 使用上下文帧生成编码
        encoded = self.unet(context_frames)

        # 获取批次大小
        batch = x.shape[0]

        # 将输入张量 x 拆分为通道、位置和数值
        channels, positions, values = x.unbind(dim=-1)

        # 获取通道嵌入
        channel_emb = self.channels(channels)
        # 获取位置嵌入
        position_emb = self.positions(positions)
        # 获取数值嵌入
        value_emb = self.values(values)

        # 将通道、位置和数值嵌入相加得到总嵌入
        embed = channel_emb + position_emb + value_emb

        # 在嵌入前添加起始标记
        start_token = repeat(self.start_token, 'd -> b 1 d', b=batch)
        embed = torch.cat((start_token, embed), dim=1)

        # 如果需要返回损失，则截取嵌入的最后一个元素
        if return_loss:
            embed = embed[:, :-1]

        # 对嵌入进行后处理归一化
        embed = self.postemb_norm(embed)

        # 注意力层 + 交叉注意力层
        for attn, cross_attn, ff in self.layers:
            embed = attn(embed) + embed
            embed = cross_attn(embed, encoded) + embed
            embed = ff(embed) + embed

        # 对最终嵌入进行归一化
        embed = self.final_norm(embed)

        # 进行轴向注意力，从通道 + 位置 + 数值的总嵌入到下一个通道 -> 下一个位置
        axial_channels_emb = self.axial_channels(channels)
        axial_positions_emb = self.axial_positions(positions)

        # 将嵌入与轴向嵌入堆叠
        embed = torch.stack((embed, axial_channels_emb, axial_positions_emb), dim=-2)

        # 重新排列嵌入
        embed = rearrange(embed, 'b m n d -> (b m) n d')

        # 轴向注意力层
        embed = self.axial_attn(embed) + embed
        embed = self.axial_ff(embed) + embed

        # 对轴向最终嵌入进行归一化
        embed = self.axial_final_norm(embed)

        # 重新排列嵌入
        embed = rearrange(embed, '(b m) n d -> b m n d', b=batch)

        # 分离通道、位置和数值嵌入
        pred_channel_embed, pred_position_embed, pred_value_embed = embed.unbind(dim=-2)

        # 转换为 logits

        channel_logits = self.to_channel_logits(pred_channel_embed)
        position_logits = self.to_position_logits(pred_position_embed)
        value_logits = self.to_value_logits(pred_value_embed)

        # 如果不需要返回损失，则返回通道 logits、位置 logits 和���值 logits
        if not return_loss:
            return channel_logits, position_logits, value_logits

        # 重新排列 logits
        channel_logits, position_logits, value_logits = map(lambda t: rearrange(t, 'b n c -> b c n'), (channel_logits, position_logits, value_logits))

        # 交叉熵损失函数
        ce = partial(F.cross_entropy, ignore_index=self.ignore_index)

        # 计算通道、位置和数值的损失
        channel_loss = ce(channel_logits, channels)
        position_loss = ce(position_logits, positions)
        value_loss = ce(value_logits, values)

        # 返回平均损失
        return (channel_loss + position_loss + value_loss) / 3
```