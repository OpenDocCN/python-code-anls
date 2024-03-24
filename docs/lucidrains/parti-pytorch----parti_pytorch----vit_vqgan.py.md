# `.\lucidrains\parti-pytorch\parti_pytorch\vit_vqgan.py`

```py
# 导入必要的库
import copy
import math
from math import sqrt
from functools import partial, wraps

# 导入自定义的模块
from vector_quantize_pytorch import VectorQuantize as VQ, LFQ

# 导入 PyTorch 库
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torchvision

# 导入 einops 库
from einops import rearrange, reduce, repeat, pack, unpack
from einops_exts import rearrange_many
from einops.layers.torch import Rearrange

# 定义常量
MList = nn.ModuleList

# 辅助函数

# 检查变量是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 装饰器

# 模型评估装饰器
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 移除 VGG 属性装饰器
def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

# 关键字参数辅助函数

# 从字典中选择指定键的值并弹出这些键
def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

# 根据条件将字典分组
def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

# 判断字符串是否以指定前缀开头
def string_begins_with(prefix, string_input):
    return string_input.startswith(prefix)

# 根据前缀将字典分组
def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

# 根据前缀将字典分组并去除前缀
def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs

# 张量辅助函数

# 对数函数
def log(t, eps = 1e-10):
    return torch.log(t + eps)

# 计算梯度惩罚
def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs = output, inputs = images,
                           grad_outputs = torch.ones(output.size(), device = images.device),
                           create_graph = True, retain_graph = True, only_inputs = True)[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

# L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# Leaky ReLU 激活函数
def leaky_relu(p = 0.1):
    return nn.LeakyReLU(0.1)

# 安全除法
def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

# GAN 损失函数

# Hinge 判别器损失
def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

# Hinge 生成器损失
def hinge_gen_loss(fake):
    return -fake.mean()

# BCE 判别器损失
def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

# BCE 生成器损失
def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()

# 计算损失对层的梯度
def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# 傅立叶变换

# 正弦余弦位置编码
class SinusoidalPosEmb(nn.Module):
    def __init__(
        self,
        dim,
        height_or_width,
        theta = 10000
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta

        hw_range = torch.arange(height_or_width)
        coors = torch.stack(torch.meshgrid(hw_range, hw_range, indexing = 'ij'), dim = -1)
        coors = rearrange(coors, 'h w c -> h w c')
        self.register_buffer('coors', coors, persistent = False)
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 计算特征维度的一半
        half_dim = self.dim // 2
        # 计算嵌入向量的值
        emb = math.log(self.theta) / (half_dim - 1)
        # 计算指数函数
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        # 重排坐标和嵌入向量的维度
        emb = rearrange(self.coors, 'h w c -> h w c 1') * rearrange(emb, 'j -> 1 1 1 j')
        # 将正弦和余弦部分连接起来
        fourier = torch.cat((emb.sin(), emb.cos()), dim = -1)
        # 将嵌入向量重复到与输入 x 相同的维度
        fourier = repeat(fourier, 'h w c d -> b (c d) h w', b = x.shape[0])
        # 将输入 x 和傅立叶特征连接起来
        return torch.cat((x, fourier), dim = 1)
# 定义通道层归一化模块
class ChanLayerNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # 计算输入张量 x 的方差和均值
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        # 返回归一化后的结果
        return (x - mean) * (var + self.eps).rsqrt() * self.gamma

# 定义交叉嵌入层模块
class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # 计算每个尺度的维度
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        # 对输入 x 进行卷积操作
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        # 拼接卷积结果
        return torch.cat(fmaps, dim = 1)

# 定义块模块
class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim)
        self.activation = leaky_relu()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)

# 定义残差块模块
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        groups = 8
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block = Block(dim, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block(x)
        return h + self.res_conv(x)

# 定义鉴别器模块
class Discriminator(nn.Module):
    def __init__(
        self,
        dims,
        channels = 3,
        groups = 8,
        init_kernel_size = 5,
        cross_embed_kernel_sizes = (3, 7, 15)
    ):
        super().__init__()
        init_dim, *_, final_dim = dims
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = MList([nn.Sequential(
            CrossEmbedLayer(channels, cross_embed_kernel_sizes, init_dim, stride = 1),
            leaky_relu()
        )])

        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                leaky_relu(),
                nn.GroupNorm(groups, dim_out),
                ResnetBlock(dim_out, dim_out),
            ))

        self.to_logits = nn.Sequential( # 返回 5 x 5，用于 PatchGAN 风格的训练
            nn.Conv2d(final_dim, final_dim, 1),
            leaky_relu(),
            nn.Conv2d(final_dim, 1, 4)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)

# 定义 2D 相对位置偏置模块
class RelPosBias2d(nn.Module):
    # 初始化函数，接受输入的size和heads参数
    def __init__(self, size, heads):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个嵌入层，用于存储位置偏置信息，参数为((2 * size - 1) ** 2, heads)
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        # 生成一个从0到size-1的张量
        arange = torch.arange(size)

        # 生成一个二维网格，表示位置信息
        pos = torch.stack(torch.meshgrid(arange, arange, indexing='ij'), dim=-1)
        # 重新排列张量的维度
        pos = rearrange(pos, '... c -> (...) c')
        # 计算相对位置信息
        rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

        # 将相对位置信息调整到合适的范围
        rel_pos = rel_pos + size - 1
        # 拆分相对位置信息为高度和宽度
        h_rel, w_rel = rel_pos.unbind(dim=-1)
        # 计算位置索引
        pos_indices = h_rel * (2 * size - 1) + w_rel
        # 将位置索引注册为模型的缓冲区
        self.register_buffer('pos_indices', pos_indices)

    # 前向传播函数，接受输入qk
    def forward(self, qk):
        # 获取输入张量的倒数第二和倒数第一维度的大小
        i, j = qk.shape[-2:]

        # 根据位置索引获取位置偏置信息
        bias = self.pos_bias(self.pos_indices)
        # 重新排列位置偏置信息的维度
        bias = rearrange(bias, 'i j h -> h i j')
        # 返回位置偏置信息
        return bias
# ViT 编码器/解码器

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        # 定义一个卷积层，用于投影
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        # 对输入进行投影操作
        return self.proj(x)

class SPT(nn.Module):
    """ https://arxiv.org/abs/2112.13492 """

    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        # 将输入图像划分为补丁，并进行通道层归一化和卷积操作
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            ChanLayerNorm(patch_dim),
            nn.Conv2d(patch_dim, dim, 1)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 32,
        fmap_size = None,
        rel_pos_bias = False
    ):
        super().__init__()
        # 通道层归一化
        self.norm = ChanLayerNorm(dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        # 将输入转换为查询、键、值，并进行卷积操作
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        self.primer_ds_convs = nn.ModuleList([PEG(inner_dim) for _ in range(3)])

        # 输出卷积层
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias = False)

        # 如果需要相对位置偏置，则创建相对位置偏置对象
        self.rel_pos_bias = None
        if rel_pos_bias:
            assert exists(fmap_size)
            self.rel_pos_bias = RelPosBias2d(fmap_size, heads)

    def forward(self, x):
        fmap_size = x.shape[-1]
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        q, k, v = [ds_conv(t) for ds_conv, t in zip(self.primer_ds_convs, (q, k, v))]
        q, k, v = rearrange_many((q, k, v), 'b (h d) x y -> b h (x y) d', h = h)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(self.rel_pos_bias):
            sim = sim + self.rel_pos_bias(sim)

        attn = sim.softmax(dim = -1, dtype = torch.float32)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = fmap_size, y = fmap_size)
        return self.to_out(out)

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1, bias = False),
        nn.GELU(),
        PEG(dim * mult),
        nn.Conv2d(dim * mult, dim, 1, bias = False)
    )

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        layers,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        fmap_size = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            # 每个 Transformer 层包含投影、注意力和前馈网络
            self.layers.append(nn.ModuleList([
                PEG(dim = dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, fmap_size = fmap_size, rel_pos_bias = True),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = ChanLayerNorm(dim)

    def forward(self, x):
        for peg, attn, ff in self.layers:
            x = peg(x) + x
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViTEncDec(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        channels = 3,
        layers = 4,
        patch_size = 16,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    # 初始化函数，设置编码维度和补丁大小
    def __init__(
        self,
        dim,
        patch_size,
        channels,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        layers = 12,
        image_size = 224
    ):
        # 调用父类初始化函数
        super().__init__()
        # 设置编码维度和补丁大小
        self.encoded_dim = dim
        self.patch_size = patch_size

        # 计算输入维度
        input_dim = channels * (patch_size ** 2)
        # 计算特征图大小
        fmap_size = image_size // patch_size

        # 编码器部分
        self.encoder = nn.Sequential(
            # SPT 模块
            SPT(dim = dim, patch_size = patch_size, channels = channels),
            # Transformer 模块
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers,
                fmap_size = fmap_size
            ),
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            # Transformer 模块
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers,
                fmap_size = fmap_size
            ),
            # 后续处理
            nn.Sequential(
                SinusoidalPosEmb(dim // 2, height_or_width = fmap_size),
                nn.Conv2d(2 * dim, dim * 4, 3, bias = False, padding = 1),
                nn.Tanh(),
                nn.Conv2d(dim * 4, input_dim, 1, bias = False),
            ),
            # 重排数据维度
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size)
        )

    # 获取编码后的特征图大小
    def get_encoded_fmap_size(self, image_size):
        return image_size // self.patch_size

    # 获取最后一个解码器层的权重
    @property
    def last_dec_layer(self):
        return self.decoder[-2][-1].weight

    # 编码函数
    def encode(self, x):
        return self.encoder(x)

    # 解码函数
    def decode(self, x):
        return self.decoder(x)
# 定义 VitVQGanVAE 类，继承自 nn.Module
class VitVQGanVAE(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,  # 模型维度
        image_size,  # 图像尺寸
        channels = 3,  # 通道数，默认为 3
        layers = 4,  # 层数，默认为 4
        l2_recon_loss = False,  # 是否使用 L2 重建损失，默认为 False
        use_hinge_loss = True,  # 是否使用 Hinge 损失，默认为 True
        vgg = None,  # VGG 模型，默认为 None
        lookup_free_quantization = True,  # 是否使用无查找表量化，默认为 True
        codebook_size = 65536,  # 代码簿大小，默认为 65536
        vq_kwargs: dict = dict(  # VQ 参数字典
            codebook_dim = 64,  # 代码簿维度，默认为 64
            decay = 0.9,  # 衰减率，默认为 0.9
            commitment_weight = 1.,  # 承诺权重，默认为 1.0
            kmeans_init = True  # 是否使用 K-means 初始化，默认为 True
        ),
        lfq_kwargs: dict = dict(  # LFQ 参数字典
            entropy_loss_weight = 0.1,  # 熵损失权重，默认为 0.1
            diversity_gamma = 2.  # 多样性参数，默认为 2.0
        ),
        use_vgg_and_gan = True,  # 是否使用 VGG 和 GAN，默认为 True
        discr_layers = 4,  # 判别器层数，默认为 4
        **kwargs  # 其他参数
    ):
        super().__init__()  # 调用父类初始化函数
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)  # 根据前缀 'vq_' 对参数进行分组
        encdec_kwargs, kwargs = groupby_prefix_and_trim('encdec_', kwargs)  # 根据前缀 'encdec_' 对参数进行分组

        self.image_size = image_size  # 图像尺寸
        self.channels = channels  # 通道数
        self.codebook_size = codebook_size  # 代码簿大小

        # 创建 ViTEncDec 实例
        self.enc_dec = ViTEncDec(
            dim = dim,
            image_size = image_size,
            channels = channels,
            layers = layers,
            **encdec_kwargs
        )

        # 提供无查找表量化
        self.lookup_free_quantization = lookup_free_quantization

        if lookup_free_quantization:
            # 创建 LFQ 实例
            self.quantizer = LFQ(
                dim = self.enc_dec.encoded_dim,
                codebook_size = codebook_size,
                **lfq_kwargs
            )
        else:
            # 创建 VQ 实例
            self.quantizer = VQ(
                dim = self.enc_dec.encoded_dim,
                codebook_size = codebook_size,
                accept_image_fmap = True,
                use_cosine_sim = True,
                **vq_kwargs
            )

        # 重��损失函数
        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss

        # 如果是灰度图像，则关闭 GAN 和感知损失
        self.vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan

        if not use_vgg_and_gan:
            return

        # 感知损失
        if exists(vgg):
            self.vgg = vgg
        else:
            self.vgg = torchvision.models.vgg16(pretrained = True)
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])

        # GAN 相关损失
        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        # 创建判别器实例
        self.discr = Discriminator(dims = dims, channels = channels)

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    @property
    def encoded_dim(self):
        return self.enc_dec.encoded_dim

    # 获取编码特征图大小
    def get_encoded_fmap_size(self, image_size):
        return self.enc_dec.get_encoded_fmap_size(image_size)

    # 用于评估的模型副本
    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    # 状态字典函数
    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    # 加载状态字典函数
    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    # 从代码簿获取特征图
    def get_fmap_from_codebook(self, indices):
        if self.lookup_free_quantization:
            indices, ps = pack([indices], 'b *')
            fmap = self.quantizer.indices_to_codes(indices)
            fmap, = unpack(fmap, ps, 'b * c')
        else:
            codes = self.quantizer.codebook[indices]
            fmap = self.vq.project_out(codes)

        return rearrange(fmap, 'b h w c -> b c h w')
    # 编码输入特征图，返回编码后的特征图、索引和量化器辅助损失
    def encode(self, fmap, return_indices_and_loss = True):
        # 使用编码器对特征图进行编码
        fmap = self.enc_dec.encode(fmap)

        # 对编码后的特征图进行量化
        fmap, indices, quantizer_aux_loss = self.quantizer(fmap)

        # 如果不需要返回索引和损失，则直接返回编码后的特征图
        if not return_indices_and_loss:
            return fmap

        # 返回编码后的特征图、索引和量化器辅助损失
        return fmap, indices, quantizer_aux_loss

    # 解码特征图
    def decode(self, fmap):
        return self.enc_dec.decode(fmap)

    # 前向传播函数
    def forward(
        self,
        img,
        return_loss = False,
        return_discr_loss = False,
        return_recons = False,
        apply_grad_penalty = True
    ):
        # 获取输入图像的批次大小、通道数、高度、宽度和设备信息
        batch, channels, height, width, device = *img.shape, img.device
        # 检查输入图像的高度和宽度是否与设定的图像大小相等
        assert height == self.image_size and width == self.image_size, 'height and width of input image must be equal to {self.image_size}'
        # 检查输入图像的通道数是否与VQGanVAE中设置的通道数相等
        assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'

        # 对输入图像进行编码，返回编码后的特征图、索引和损失
        fmap, indices, commit_loss = self.encode(img, return_indices_and_loss = True)

        # 对编码后的特征图进行解码
        fmap = self.decode(fmap)

        # 如果不需要返回损失和判别器损失，则直接返回解码后的特征图
        if not return_loss and not return_discr_loss:
            return fmap

        # 确保只返回自编码器损失或判别器损失，而不是两者都返回
        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'

        # 是否返回判别器损失
        if return_discr_loss:
            # 确保判别器存在以便训练
            assert exists(self.discr), 'discriminator must exist to train it'

            # 分离编码后的特征图，使其不参与梯度计算
            fmap.detach_()
            # 设置输入图像需要计算梯度
            img.requires_grad_()

            # 获取编码后特征图和输入图像的判别器logits
            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))

            # 计算判别器损失
            discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

            # 如果应用梯度惩罚
            if apply_grad_penalty:
                # 计算梯度惩罚
                gp = gradient_penalty(img, img_discr_logits)
                loss = discr_loss + gp

            # 如果需要返回重构图像
            if return_recons:
                return loss, fmap

            return loss

        # 计算重构损失
        recon_loss = self.recon_loss_fn(fmap, img)

        # 如果不使用VGG和GAN
        if not self.use_vgg_and_gan:
            # 如果需要返回重构图像
            if return_recons:
                return recon_loss, fmap

            return recon_loss

        # 计算感知损失
        img_vgg_input = img
        fmap_vgg_input = fmap

        if img.shape[1] == 1:
            # 处理灰度图像用于VGG
            img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (img_vgg_input, fmap_vgg_input))

        # 获取输入图像和解码后特征图的VGG特征
        img_vgg_feats = self.vgg(img_vgg_input)
        recon_vgg_feats = self.vgg(fmap_vgg_input)
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

        # 生成器损失
        gen_loss = self.gen_loss(self.discr(fmap))

        # 计算自适应权重
        last_dec_layer = self.enc_dec.last_dec_layer

        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max = 1e4)

        # 组合损失
        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss

        # 如果需要返回重构图像
        if return_recons:
            return loss, fmap

        return loss
```