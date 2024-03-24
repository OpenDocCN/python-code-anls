# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\vqgan_vae.py`

```py
# 导入必要的库
import copy
import math
from math import sqrt
from functools import partial, wraps

# 导入自定义模块
from vector_quantize_pytorch import VectorQuantize as VQ

# 导入 PyTorch 库
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torchvision

# 导入 einops 库
from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

# 定义常量
MList = nn.ModuleList

# 辅助函数

# 判断变量是否存在
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

# 移除 VGG 模型装饰器
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

# 稳定的 Softmax 函数
def stable_softmax(t, dim = -1, alpha = 32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim = dim, keepdim = True).detach()
    return (t * alpha).softmax(dim = dim)

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

# 二元交叉熵判别器损失
def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

# 二元交叉熵生成器损失
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

# VQGAN VAE

# 通道层归一化
class LayerNormChan(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

# 判别器

class Discriminator(nn.Module):
    def __init__(
        self,
        dims,
        channels = 3,
        groups = 16,
        init_kernel_size = 5
    # 定义一个继承自 nn.Module 的类，用于构建一个简单的卷积神经网络
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 将输入维度按照前后两两配对，形成一个维度对的列表
        dim_pairs = zip(dims[:-1], dims[1:])

        # 初始化网络的第一层，包括一个卷积层和激活函数
        self.layers = MList([nn.Sequential(nn.Conv2d(channels, dims[0], init_kernel_size, padding = init_kernel_size // 2), leaky_relu())])

        # 遍历维度对列表，构建网络的中间层，每层包括卷积层、归一化层和激活函数
        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                nn.GroupNorm(groups, dim_out),
                leaky_relu()
            ))

        # 获取最后一个维度
        dim = dims[-1]
        # 构建输出层，包括两个卷积层和激活函数，用于生成输出结果
        self.to_logits = nn.Sequential( # return 5 x 5, for PatchGAN-esque training
            nn.Conv2d(dim, dim, 1),
            leaky_relu(),
            nn.Conv2d(dim, 1, 4)
        )

    # 定义前向传播方法，将输入数据通过网络层进行处理，得到输出结果
    def forward(self, x):
        # 遍历网络的每一层，将输入数据依次传递给每一层
        for net in self.layers:
            x = net(x)

        # 返回经过所有网络层处理后的输出结果
        return self.to_logits(x)
# positional encoding

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(self, *, dim, heads, layers = 2):
        super().__init__()
        self.net = MList([])
        self.net.append(nn.Sequential(nn.Linear(2, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads)
        # 初始化一个空的相对位置矩阵
        self.register_buffer('rel_pos', None, persistent = False)

    def forward(self, x):
        n, device = x.shape[-1], x.device
        fmap_size = int(sqrt(n))

        if not exists(self.rel_pos):
            # 生成位置信息
            pos = torch.arange(fmap_size, device = device)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')
            rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            # 将生成的位置信息存储在缓冲区中
            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        bias = rearrange(rel_pos, 'i j h -> h i j')
        return x + bias

# resnet encoder / decoder

class ResnetEncDec(nn.Module):
    def __init__(
        self,
        dim,
        *,
        channels = 3,
        layers = 4,
        layer_mults = None,
        num_resnet_blocks = 1,
        resnet_groups = 16,
        first_conv_kernel_size = 5,
        use_attn = True,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_dropout = 0.,
    ):
        super().__init__()
        assert dim % resnet_groups == 0, f'dimension {dim} must be divisible by {resnet_groups} (groups for the groupnorm)'

        self.layers = layers

        self.encoders = MList([])
        self.decoders = MList([])

        layer_mults = default(layer_mults, list(map(lambda t: 2 ** t, range(layers))))
        assert len(layer_mults) == layers, 'layer multipliers must be equal to designated number of layers'

        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        self.encoded_dim = dims[-1]

        dim_pairs = zip(dims[:-1], dims[1:])

        append = lambda arr, t: arr.append(t)
        prepend = lambda arr, t: arr.insert(0, t)

        if not isinstance(num_resnet_blocks, tuple):
            num_resnet_blocks = (*((0,) * (layers - 1)), num_resnet_blocks)

        if not isinstance(use_attn, tuple):
            use_attn = (*((False,) * (layers - 1)), use_attn)

        assert len(num_resnet_blocks) == layers, 'number of resnet blocks config must be equal to number of layers'
        assert len(use_attn) == layers

        for layer_index, (dim_in, dim_out), layer_num_resnet_blocks, layer_use_attn in zip(range(layers), dim_pairs, num_resnet_blocks, use_attn):
            append(self.encoders, nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1), leaky_relu()))
            prepend(self.decoders, nn.Sequential(nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1), leaky_relu()))

            if layer_use_attn:
                prepend(self.decoders, VQGanAttention(dim = dim_out, heads = attn_heads, dim_head = attn_dim_head, dropout = attn_dropout))

            for _ in range(layer_num_resnet_blocks):
                append(self.encoders, ResBlock(dim_out, groups = resnet_groups))
                prepend(self.decoders, GLUResBlock(dim_out, groups = resnet_groups))

            if layer_use_attn:
                append(self.encoders, VQGanAttention(dim = dim_out, heads = attn_heads, dim_head = attn_dim_head, dropout = attn_dropout))

        prepend(self.encoders, nn.Conv2d(channels, dim, first_conv_kernel_size, padding = first_conv_kernel_size // 2))
        append(self.decoders, nn.Conv2d(dim, channels, 1))

    def get_encoded_fmap_size(self, image_size):
        return image_size // (2 ** self.layers)
    # 定义一个属性，返回最后一个解码器的权重
    @property
    def last_dec_layer(self):
        return self.decoders[-1].weight

    # 编码函数，对输入数据进行编码
    def encode(self, x):
        # 遍历所有编码器，对输入数据进行编码
        for enc in self.encoders:
            x = enc(x)
        # 返回编码后的数据
        return x

    # 解码函数，对输入数据进行解码
    def decode(self, x):
        # 遍历所有解码器，对输入数据进行解码
        for dec in self.decoders:
            x = dec(x)
        # 返回解码后的数据
        return x
# 定义 GLUResBlock 类，继承自 nn.Module
class GLUResBlock(nn.Module):
    # 初始化函数，接受通道数和组数作为参数
    def __init__(self, chan, groups = 16):
        super().__init__()
        # 定义网络结构为一个序列
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan * 2, 3, padding = 1),  # 3x3 卷积层
            nn.GLU(dim = 1),  # GLU 激活函数
            nn.GroupNorm(groups, chan),  # 分组归一化
            nn.Conv2d(chan, chan * 2, 3, padding = 1),  # 3x3 卷积层
            nn.GLU(dim = 1),  # GLU 激活函数
            nn.GroupNorm(groups, chan),  # 分组归一化
            nn.Conv2d(chan, chan, 1)  # 1x1 卷积层
        )

    # 前向传播函数
    def forward(self, x):
        return self.net(x) + x  # 返回网络输出与输入的和

# 定义 ResBlock 类，继承自 nn.Module
class ResBlock(nn.Module):
    # 初始化函数，接受通道数和组数作为参数
    def __init__(self, chan, groups = 16):
        super().__init__()
        # 定义网络结构为一个序列
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),  # 3x3 卷积层
            nn.GroupNorm(groups, chan),  # 分组归一化
            leaky_relu(),  # leaky_relu 激活函数
            nn.Conv2d(chan, chan, 3, padding = 1),  # 3x3 卷积层
            nn.GroupNorm(groups, chan),  # 分组归一化
            leaky_relu(),  # leaky_relu 激活函数
            nn.Conv2d(chan, chan, 1)  # 1x1 卷积层
        )

    # 前向传播函数
    def forward(self, x):
        return self.net(x) + x  # 返回网络输出与输入的和

# 定义 VQGanAttention 类，继承自 nn.Module
class VQGanAttention(nn.Module):
    # 初始化函数，接受维度、头数、头维度和 dropout 等参数
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.dropout = nn.Dropout(dropout)
        self.pre_norm = LayerNormChan(dim)

        self.cpb = ContinuousPositionBias(dim = dim // 4, heads = heads)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias = False)

    # 前向传播函数
    def forward(self, x):
        h = self.heads
        height, width, residual = *x.shape[-2:], x.clone()

        x = self.pre_norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = h), (q, k, v))

        sim = einsum('b h c i, b h c j -> b h i j', q, k) * self.scale

        sim = self.cpb(sim)

        attn = stable_softmax(sim, dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h c j -> b h c i', attn, v)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x = height, y = width)
        out = self.to_out(out)

        return out + residual

# 定义 RearrangeImage 类，继承自 nn.Module
class RearrangeImage(nn.Module):
    # 前向传播函数
    def forward(self, x):
        n = x.shape[1]
        w = h = int(sqrt(n))
        return rearrange(x, 'b (h w) ... -> b h w ...', h = h, w = w)

# 定义 Attention 类，继承自 nn.Module
class Attention(nn.Module):
    # 初始化函数，接受维度、头数和头维度等参数
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 32
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    # 前向传播函数
    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义 FeedForward 函数，返回一个包含层归一化、线性层、GELU 激活函数和线性层的序列
def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult, bias = False),
        nn.GELU(),
        nn.Linear(dim * mult, dim, bias = False)
    )

# 定义 Transformer 类，继承自 nn.Module
class Transformer(nn.Module):
    # 初始化函数，接受维度、层数、头维度、头数和前馈网络倍数等参数
    def __init__(
        self,
        dim,
        *,
        layers,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    ):  
        # 调用父类的构造函数
        super().__init__()
        # 初始化一个空的神经网络模块列表
        self.layers = nn.ModuleList([])
        # 循环创建指定数量的层
        for _ in range(layers):
            # 向神经网络模块列表中添加一个包含注意力和前馈神经网络的模块列表
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        # 初始化一个 LayerNorm 层
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 遍历每一层的注意力和前馈神经网络
        for attn, ff in self.layers:
            # 对输入进行注意力操作并加上原始输入
            x = attn(x) + x
            # 对输入进行前馈神经网络操作并加上原始输入
            x = ff(x) + x

        # 对最终结果进行 LayerNorm 操作
        return self.norm(x)
# 定义 ViTEncDec 类，继承自 nn.Module
class ViTEncDec(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        channels = 3,
        layers = 4,
        patch_size = 8,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置编码后的维度
        self.encoded_dim = dim
        # 设置补丁大小
        self.patch_size = patch_size

        # 计算输入维度
        input_dim = channels * (patch_size ** 2)

        # 定义编码器部分
        self.encoder = nn.Sequential(
            # 重排输入数据形状
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # 线性层
            nn.Linear(input_dim, dim),
            # Transformer 模块
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers
            ),
            # 重排图像数据形状
            RearrangeImage(),
            # 重排输出数据形状
            Rearrange('b h w c -> b c h w')
        )

        # 定义解码器部分
        self.decoder = nn.Sequential(
            # 重排输入数据形状
            Rearrange('b c h w -> b (h w) c'),
            # Transformer 模块
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers
            ),
            # 线性层和激活函数
            nn.Sequential(
                nn.Linear(dim, dim * 4, bias = False),
                nn.Tanh(),
                nn.Linear(dim * 4, input_dim, bias = False),
            ),
            # 重排图像数据形状
            RearrangeImage(),
            # 重排输出数据形状
            Rearrange('b h w (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size)
        )

    # 获取编码后特征图的大小
    def get_encoded_fmap_size(self, image_size):
        return image_size // self.patch_size

    # 返回解码器的最后一层
    @property
    def last_dec_layer(self):
        return self.decoder[-3][-1].weight

    # 编码函数
    def encode(self, x):
        return self.encoder(x)

    # 解码函数
    def decode(self, x):
        return self.decoder(x)

# 定义 NullVQGanVAE 类，继承自 nn.Module
class NullVQGanVAE(nn.Module):
    # 初始化函数��接受 channels 参数
    def __init__(
        self,
        *,
        channels
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置编码后的维度为 channels
        self.encoded_dim = channels
        # 设置层数为 0
        self.layers = 0

    # 获取编码后特征图的大小
    def get_encoded_fmap_size(self, size):
        return size

    # 复制模型用于评估
    def copy_for_eval(self):
        return self

    # 编码函数
    def encode(self, x):
        return x

    # 解码函数
    def decode(self, x):
        return x

# 定义 VQGanVAE 类，继承自 nn.Module
class VQGanVAE(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        layers = 4,
        l2_recon_loss = False,
        use_hinge_loss = True,
        vgg = None,
        vq_codebook_dim = 256,
        vq_codebook_size = 512,
        vq_decay = 0.8,
        vq_commitment_weight = 1.,
        vq_kmeans_init = True,
        vq_use_cosine_sim = True,
        use_vgg_and_gan = True,
        vae_type = 'resnet',
        discr_layers = 4,
        **kwargs
    # 初始化函数，设置各种参数
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 将参数按照前缀分组，提取出以'vq_'开头的参数
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)
        # 将参数按照前缀分组，提取出以'encdec_'开头的参数
        encdec_kwargs, kwargs = groupby_prefix_and_trim('encdec_', kwargs)

        # 设置图像大小、通道数、VQ 编码簇大小
        self.image_size = image_size
        self.channels = channels
        self.codebook_size = vq_codebook_size

        # 根据 VAE 类型选择编码器解码器类
        if vae_type == 'resnet':
            enc_dec_klass = ResnetEncDec
        elif vae_type == 'vit':
            enc_dec_klass = ViTEncDec
        else:
            raise ValueError(f'{vae_type} not valid')

        # 初始化编码器解码器
        self.enc_dec = enc_dec_klass(
            dim = dim,
            channels = channels,
            layers = layers,
            **encdec_kwargs
        )

        # 初始化 VQ 模块
        self.vq = VQ(
            dim = self.enc_dec.encoded_dim,
            codebook_dim = vq_codebook_dim,
            codebook_size = vq_codebook_size,
            decay = vq_decay,
            commitment_weight = vq_commitment_weight,
            accept_image_fmap = True,
            kmeans_init = vq_kmeans_init,
            use_cosine_sim = vq_use_cosine_sim,
            **vq_kwargs
        )

        # 设置重构损失函数
        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss

        # 如果是灰度图像，则关闭 GAN 和感知损失
        self.vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan

        if not use_vgg_and_gan:
            return

        # 初始化感知损失
        if exists(vgg):
            self.vgg = vgg
        else:
            self.vgg = torchvision.models.vgg16(pretrained = True)
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])

        # 初始化 GAN 相关损失
        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        self.discr = Discriminator(dims = dims, channels = channels)

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    # 获取编码后的维度
    @property
    def encoded_dim(self):
        return self.enc_dec.encoded_dim

    # 获取编码后特征图的大小
    def get_encoded_fmap_size(self, image_size):
        return self.enc_dec.get_encoded_fmap_size(image_size)

    # 复制模型用于评估
    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    # 获取模型状态字典
    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    # 加载模型状态字典
    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    # 获取编码簇
    @property
    def codebook(self):
        return self.vq.codebook

    # 编码
    def encode(self, fmap):
        fmap = self.enc_dec.encode(fmap)
        return fmap

    # 解码
    def decode(self, fmap, return_indices_and_loss = False):
        fmap, indices, commit_loss = self.vq(fmap)

        fmap = self.enc_dec.decode(fmap)

        if not return_indices_and_loss:
            return fmap

        return fmap, indices, commit_loss

    # 前向传播
    def forward(
        self,
        img,
        return_loss = False,
        return_discr_loss = False,
        return_recons = False,
        add_gradient_penalty = True
        ):
            # 解构赋值，获取图像的批次、通道数、高度、宽度、设备信息
            batch, channels, height, width, device = *img.shape, img.device
            # 断言输入图像的高度和宽度与设定的图像大小相等
            assert height == self.image_size and width == self.image_size, 'height and width of input image must be equal to {self.image_size}'
            # 断言输入图像的通道数与 VQGanVAE 中设定的通道数相等
            assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'

            # 编码输入图像
            fmap = self.encode(img)

            # 解码编码后的特征图，并返回索引和损失
            fmap, indices, commit_loss = self.decode(fmap, return_indices_and_loss = True)

            if not return_loss and not return_discr_loss:
                return fmap

            # 断言只能返回自编码器损失或鉴别器损失，不能同时返回
            assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'

            # 是否返回鉴别器损失
            if return_discr_loss:
                # 断言鉴别器存在
                assert exists(self.discr), 'discriminator must exist to train it'

                # 分离编码后的特征图，设置输入图像为需要梯度
                fmap.detach_()
                img.requires_grad_()

                # 获取编码后特征图和输入图像的鉴别器输出
                fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))

                # 计算鉴别器损失
                discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

                if add_gradient_penalty:
                    # 添加梯度惩罚项
                    gp = gradient_penalty(img, img_discr_logits)
                    loss = discr_loss + gp

                if return_recons:
                    return loss, fmap

                return loss

            # 重构损失
            recon_loss = self.recon_loss_fn(fmap, img)

            # 若不使用 VGG 和 GAN
            if not self.use_vgg_and_gan:
                if return_recons:
                    return recon_loss, fmap

                return recon_loss

            # 感知损失
            img_vgg_input = img
            fmap_vgg_input = fmap

            if img.shape[1] == 1:
                # 处理灰度图像用于 VGG
                img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (img_vgg_input, fmap_vgg_input))

            # 获取输入图像和重构图像的 VGG 特征
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

            if return_recons:
                return loss, fmap

            return loss
```