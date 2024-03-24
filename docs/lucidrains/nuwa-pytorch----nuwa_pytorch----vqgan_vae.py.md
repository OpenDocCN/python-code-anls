# `.\lucidrains\nuwa-pytorch\nuwa_pytorch\vqgan_vae.py`

```
# 导入必要的库
import copy
import math
from functools import partial, wraps
from math import sqrt

# 导入自定义模块
from vector_quantize_pytorch import VectorQuantize as VQ

# 导入 PyTorch 相关库
import torchvision
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

# 导入 einops 库
from einops import rearrange, reduce, repeat

# 定义常量
MList = nn.ModuleList

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
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
def string_begins_with(prefix, str):
    return str.startswith(prefix)

# 根据前缀将字典分组
def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

# 根据前缀将字典分组并去除前缀
def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs

# 张量辅助函数

# 计算梯度惩罚
def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs = output, inputs = images,
                           grad_outputs = torch.ones(output.size(), device = images.device),
                           create_graph = True, retain_graph = True, only_inputs = True)[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# 计算 L2 范数
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
def safe_div(numer, denom, eps = 1e-6):
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
    return (-log(1 - sigmoid(fake)) - log(sigmoid(real))).mean()

# 二元交叉熵生成器损失
def bce_gen_loss(fake):
    return -log(sigmoid(fake)).mean()

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
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# 判别器模型
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
class ContinuousPositionBias(nn.Module):
    """ 定义一个连续位置偏置的类，参考 https://arxiv.org/abs/2111.09883 """

    def __init__(self, *, dim, heads, layers = 2):
        super().__init__()
        self.net = MList([])
        self.net.append(nn.Sequential(nn.Linear(2, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads)
        self.register_buffer('rel_pos', None, persistent = False)

    def forward(self, x):
        n, device = x.shape[-1], x.device
        fmap_size = int(sqrt(n))

        if not exists(self.rel_pos):
            pos = torch.arange(fmap_size, device = device)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')
            rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        bias = rearrange(rel_pos, 'i j h -> h i j')
        return x + bias

class GLUResBlock(nn.Module):
    """ 定义一个 GLUResBlock 类 """

    def __init__(self, chan, groups = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan * 2, 3, padding = 1),
            nn.GLU(dim = 1),
            nn.GroupNorm(groups, chan),
            nn.Conv2d(chan, chan * 2, 3, padding = 1),
            nn.GLU(dim = 1),
            nn.GroupNorm(groups, chan),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class ResBlock(nn.Module):
    """ 定义一个 ResBlock 类 """

    def __init__(self, chan, groups = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GroupNorm(groups, chan),
            leaky_relu(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GroupNorm(groups, chan),
            leaky_relu(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class VQGanAttention(nn.Module):
    """ 定义一个 VQGanAttention 类 """

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
        self.scale = nn.Parameter(torch.ones(1, heads, 1, 1) * math.log(0.01))
        inner_dim = heads * dim_head

        self.dropout = nn.Dropout(dropout)
        self.post_norm = LayerNormChan(dim)

        self.cpb = ContinuousPositionBias(dim = dim // 4, heads = heads)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        h = self.heads
        height, width, residual = *x.shape[-2:], x.clone()

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = h), (q, k, v))

        q, k = map(l2norm, (q, k))

        sim = einsum('b h c i, b h c j -> b h i j', q, k) * self.scale.exp()

        sim = self.cpb(sim)

        attn = stable_softmax(sim, dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h c j -> b h c i', attn, v)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x = height, y = width)
        out = self.to_out(out)

        return self.post_norm(out) + residual

class VQGanVAE(nn.Module):
    """ 定义一个 VQGanVAE 类 """
    # 初始化函数，设置模型的参数
    def __init__(
        self,
        *,
        dim,  # 模型的维度
        image_size,  # 图像的尺寸
        channels = 3,  # 图像的通道数，默认为3
        num_layers = 4,  # 模型的层数，默认为4
        layer_mults = None,  # 每一层的倍增因子
        l2_recon_loss = False,  # 是否使用L2重建损失，默认为False
        use_hinge_loss = True,  # 是否使用hinge损失，默认为True
        num_resnet_blocks = 1,  # ResNet块的数量，默认为1
        vgg = None,  # VGG模型
        vq_codebook_dim = 256,  # VQ编码簇的维度
        vq_codebook_size = 512,  # VQ编码簇的大小
        vq_decay = 0.8,  # VQ衰减率
        vq_commitment_weight = 1.,  # VQ损失的权重
        vq_kmeans_init = True,  # 是否使用K均值初始化VQ编码簇，默认为True
        vq_use_cosine_sim = True,  # 是否使用余弦相似度计算VQ损失，默认为True
        use_attn = True,  # 是否使用注意力机制，默认为True
        attn_dim_head = 64,  # 注意力机制的头维度
        attn_heads = 8,  # 注意力机制的头数量
        resnet_groups = 16,  # ResNet块的组数
        attn_dropout = 0.,  # 注意力机制的dropout率
        first_conv_kernel_size = 5,  # 第一个卷积层的卷积核大小
        use_vgg_and_gan = True,  # 是否同时使用VGG和GAN，默认为True
        **kwargs  # 其他参数
        ):
        # 调用父类的构造函数
        super().__init__()
        # 断言维度必须能够被 resnet_groups 整除
        assert dim % resnet_groups == 0, f'dimension {dim} must be divisible by {resnet_groups} (groups for the groupnorm)'

        # 将参数中以 'vq_' 开头的参数提取出来
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)

        # 初始化一些属性
        self.image_size = image_size
        self.channels = channels
        self.num_layers = num_layers
        self.fmap_size = image_size // (num_layers ** 2)
        self.codebook_size = vq_codebook_size

        self.encoders = MList([])
        self.decoders = MList([])

        # 计算每一层的维度
        layer_mults = default(layer_mults, list(map(lambda t: 2 ** t, range(num_layers))))
        assert len(layer_mults) == num_layers, 'layer multipliers must be equal to designated number of layers'

        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)
        codebook_dim = layer_dims[-1]

        dim_pairs = zip(dims[:-1], dims[1:])

        append = lambda arr, t: arr.append(t)
        prepend = lambda arr, t: arr.insert(0, t)

        # 如果 num_resnet_blocks 不是元组，则转换为元组
        if not isinstance(num_resnet_blocks, tuple):
            num_resnet_blocks = (*((0,) * (num_layers - 1)), num_resnet_blocks)

        # 如果 use_attn 不是元组，则转换为元组
        if not isinstance(use_attn, tuple):
            use_attn = (*((False,) * (num_layers - 1)), use_attn)

        assert len(num_resnet_blocks) == num_layers, 'number of resnet blocks config must be equal to number of layers'
        assert len(use_attn) == num_layers

        # 遍历每一层，构建编码器和解码器
        for layer_index, (dim_in, dim_out), layer_num_resnet_blocks, layer_use_attn in zip(range(num_layers), dim_pairs, num_resnet_blocks, use_attn):
            append(self.encoders, nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1), leaky_relu()))
            prepend(self.decoders, nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False), nn.Conv2d(dim_out, dim_in, 3, padding = 1), leaky_relu()))

            if layer_use_attn:
                prepend(self.decoders, VQGanAttention(dim = dim_out, heads = attn_heads, dim_head = attn_dim_head, dropout = attn_dropout))

            for _ in range(layer_num_resnet_blocks):
                append(self.encoders, ResBlock(dim_out, groups = resnet_groups))
                prepend(self.decoders, GLUResBlock(dim_out, groups = resnet_groups))

            if layer_use_attn:
                append(self.encoders, VQGanAttention(dim = dim_out, heads = attn_heads, dim_head = attn_dim_head, dropout = attn_dropout))

        prepend(self.encoders, nn.Conv2d(channels, dim, first_conv_kernel_size, padding = first_conv_kernel_size // 2))
        append(self.decoders, nn.Conv2d(dim, channels, 1))

        # 初始化 VQ 模块
        self.vq = VQ(
            dim = layer_dims[-1],
            codebook_dim = vq_codebook_dim,
            codebook_size = vq_codebook_size,
            decay = vq_decay,
            commitment_weight = vq_commitment_weight,
            accept_image_fmap = True,
            kmeans_init = vq_kmeans_init,
            use_cosine_sim = vq_use_cosine_sim,
            **vq_kwargs
        )

        # 重构损失函数
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

        # 初始化GAN相关损失
        self.discr = Discriminator(dims = dims, channels = channels)

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss
    # 创建一个模型的副本用于评估，确保在同一设备上
    def copy_for_eval(self):
        # 获取模型参数的设备信息
        device = next(self.parameters()).device
        # 深度复制模型并将其移动到 CPU
        vae_copy = copy.deepcopy(self.cpu())

        # 如果模型使用 VGG 和 GAN，则删除相关部分
        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        # 将模型设置为评估模式
        vae_copy.eval()
        # 将模型移动回原设备
        return vae_copy.to(device)

    # 重写父类的 state_dict 方法，移除 VGG 相关部分
    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    # 重写父类的 load_state_dict 方法，移除 VGG 相关部分
    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    # 返回模型的 codebook 属性，即 VQ 模块的 codebook
    @property
    def codebook(self):
        return self.vq.codebook

    # 对输入进行编码操作，通过多个编码器层
    def encode(self, fmap):
        for enc in self.encoders:
            fmap = enc(fmap)

        return self.vq(fmap)

    # 对输入进行解码操作，通过多个解码器层
    def decode(self, fmap):
        for dec in self.decoders:
            fmap = dec(fmap)

        return fmap

    # 将 codebook 索引转换为视频数据
    @torch.no_grad()
    @eval_decorator
    def codebook_indices_to_video(self, indices):
        b = indices.shape[0]
        codes = self.codebook[indices]
        codes = rearrange(codes, 'b (f h w) d -> (b f) d h w', h = self.fmap_size, w = self.fmap_size)
        video = self.decode(codes)
        return rearrange(video, '(b f) ... -> b f ...', b = b)

    # 从视频数据中获取 codebook 索引
    @torch.no_grad()
    @eval_decorator
    def get_video_indices(self, video):
        b, f, _, h, w = video.shape
        images = rearrange(video, 'b f ... -> (b f) ...')
        _, indices, _ = self.encode(images)
        return rearrange(indices, '(b f) ... -> b f ...', b = b)

    # 模型的前向传播方法，包括返回损失、重构、梯度惩罚等选项
    def forward(
        self,
        img,
        return_loss = False,
        return_discr_loss = False,
        return_recons = False,
        apply_grad_penalty = False
        ):
        # 解构赋值，获取图像的批次、通道数、高度、宽度和设备信息
        batch, channels, height, width, device = *img.shape, img.device
        # 断言输入图像的高度和宽度与设定的self.image_size相等
        assert height == self.image_size and width == self.image_size, 'height and width of input image must be equal to {self.image_size}'
        # 断言输入图像的通道数与VQGanVAE中设定的通道数相等
        assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'

        # 编码输入图像，获取特征图、索引和commit_loss
        fmap, indices, commit_loss = self.encode(img)

        # 解码特征图
        fmap = self.decode(fmap)

        # 如果不需要返回损失和鉴别器损失，则直接返回解码后的特征图
        if not return_loss and not return_discr_loss:
            return fmap

        # 断言只能返回自编码器损失或鉴别器损失，不能同时返回
        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'

        # 是否返回鉴别器损失
        if return_discr_loss:
            # 断言鉴别器存在
            assert exists(self.discr), 'discriminator must exist to train it'

            # 分离特征图，设置输入图像为可求导
            fmap.detach_()
            img.requires_grad_()

            # 获取特征图和输入图像的鉴别器logits
            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))

            # 计算鉴别器损失
            loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

            # 如果需要应用梯度惩罚
            if apply_grad_penalty:
                gp = gradient_penalty(img, img_discr_logits)
                loss = loss + gp

            # 如果需要返回重构图像
            if return_recons:
                return loss, fmap

            return loss

        # 重构损失
        recon_loss = self.recon_loss_fn(fmap, img)

        # 如果不使用VGG和GAN，则直接返回重构损失
        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, fmap

            return recon_loss

        # 感知损失
        img_vgg_input = img
        fmap_vgg_input = fmap

        # 处理灰度图像用于VGG
        if img.shape[1] == 1:
            img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (img_vgg_input, fmap_vgg_input))

        # 获取输入图像和重构图像的VGG特征
        img_vgg_feats = self.vgg(img_vgg_input)
        recon_vgg_feats = self.vgg(fmap_vgg_input)
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

        # 生成器损失
        gen_loss = self.gen_loss(self.discr(fmap))

        # 计算自适应权重
        last_dec_layer = self.decoders[-1].weight

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