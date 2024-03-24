# `.\lucidrains\muse-maskgit-pytorch\muse_maskgit_pytorch\vqgan_vae.py`

```py
# 导入必要的模块
from pathlib import Path
import copy
import math
from math import sqrt
from functools import partial, wraps

# 导入自定义模块
from vector_quantize_pytorch import VectorQuantize as VQ, LFQ

# 导入 PyTorch 模块
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

# 导入 torchvision 模块
import torchvision

# 导入 einops 模块
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

# 移除 VGG 属性装饰器
def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, '_vgg')
        if has_vgg:
            vgg = self._vgg
            delattr(self, '_vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self._vgg = vgg

        return out
    return inner

# 关键字参数辅助函数

# 选择并弹出指定键的值
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

# 根据前缀分组并修剪
def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs

# 张量辅助函数

# 对数函数
def log(t, eps = 1e-10):
    return torch.log(t + eps)

# 梯度惩罚函数
def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = images.device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

# Leaky ReLU 函数
def leaky_relu(p = 0.1):
    return nn.LeakyReLU(0.1)

# 安全除法函数
def safe_div(numer, denom, eps = 1e-8):
    return numer / denom.clamp(min = eps)

# GAN 损失函数

# Hinge 判别器损失函数
def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

# Hinge 生成器损失函数
def hinge_gen_loss(fake):
    return -fake.mean()

# BCE 判别器损失函数
def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

# BCE 生成器损失函数
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

# 通道层归一化类
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
        return (x - mean) * var.clamp(min = self.eps).rsqrt() * self.gamma

# 判别器类
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
# 定义一个名为 ResnetEncDec 的类，用于实现 ResNet 编码器/解码器
class ResnetEncDec(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        *,
        channels = 3,
        layers = 4,
        layer_mults = None,
        num_resnet_blocks = 1,
        resnet_groups = 16,
        first_conv_kernel_size = 5
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 断言确保维度能够被 resnet_groups 整除
        assert dim % resnet_groups == 0, f'dimension {dim} must be divisible by {resnet_groups} (groups for the groupnorm)'

        # 初始化 layers 属性
        self.layers = layers

        # 初始化 encoders 和 decoders 为 MList 类型的空列表
        self.encoders = MList([])
        self.decoders = MList([])

        # 如果未提供 layer_mults 参数，则使用默认值
        layer_mults = default(layer_mults, list(map(lambda t: 2 ** t, range(layers))))
        # 断言确保 layer_mults 的长度等于 layers
        assert len(layer_mults) == layers, 'layer multipliers must be equal to designated number of layers'

        # 计算每一层的维度
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        # 记录编码后的维度
        self.encoded_dim = dims[-1]

        # 计算每一层的输入输出维度
        dim_pairs = zip(dims[:-1], dims[1:])

        # 定义辅助函数 append 和 prepend
        append = lambda arr, t: arr.append(t)
        prepend = lambda arr, t: arr.insert(0, t)

        # 如果 num_resnet_blocks 不是元组，则转换为元组
        if not isinstance(num_resnet_blocks, tuple):
            num_resnet_blocks = (*((0,) * (layers - 1)), num_resnet_blocks)

        # 断言确保 num_resnet_blocks 的长度等于 layers
        assert len(num_resnet_blocks) == layers, 'number of resnet blocks config must be equal to number of layers'

        # 遍历每一层，构建编码器和解码器
        for layer_index, (dim_in, dim_out), layer_num_resnet_blocks in zip(range(layers), dim_pairs, num_resnet_blocks):
            # 添加卷积层和激活函数到编码器
            append(self.encoders, nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1), leaky_relu()))
            # 添加反卷积层和激活函数到解码器
            prepend(self.decoders, nn.Sequential(nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1), leaky_relu()))

            # 添加 ResBlock 或 GLUResBlock 到编码器和解码器
            for _ in range(layer_num_resnet_blocks):
                append(self.encoders, ResBlock(dim_out, groups = resnet_groups))
                prepend(self.decoders, GLUResBlock(dim_out, groups = resnet_groups))

        # 添加第一层卷积层到编码器
        prepend(self.encoders, nn.Conv2d(channels, dim, first_conv_kernel_size, padding = first_conv_kernel_size // 2))
        # 添加最后一层卷积层到解码器
        append(self.decoders, nn.Conv2d(dim, channels, 1))

    # 获取编码后特征图的大小
    def get_encoded_fmap_size(self, image_size):
        return image_size // (2 ** self.layers)

    # 返回最后一层解码器的权重
    @property
    def last_dec_layer(self):
        return self.decoders[-1].weight

    # 编码函数
    def encode(self, x):
        for enc in self.encoders:
            x = enc(x)
        return x

    # 解码函数
    def decode(self, x):
        for dec in self.decoders:
            x = dec(x)
        return x

# 定义 GLUResBlock 类，继承自 nn.Module
class GLUResBlock(nn.Module):
    # 初始化函数，接受通道数和组数参数
    def __init__(self, chan, groups = 16):
        # 调用父类的初始化函数
        super().__init__()
        # 定义网络结构
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan * 2, 3, padding = 1),
            nn.GLU(dim = 1),
            nn.GroupNorm(groups, chan),
            nn.Conv2d(chan, chan * 2, 3, padding = 1),
            nn.GLU(dim = 1),
            nn.GroupNorm(groups, chan),
            nn.Conv2d(chan, chan, 1)
        )

    # 前向传播函数
    def forward(self, x):
        return self.net(x) + x

# 定义 ResBlock 类，继承自 nn.Module
class ResBlock(nn.Module):
    # 初始化函数，接受通道数和组数参数
    def __init__(self, chan, groups = 16):
        # 调用父类的初始化函数
        super().__init__()
        # 定义网络结构
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GroupNorm(groups, chan),
            leaky_relu(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GroupNorm(groups, chan),
            leaky_relu(),
            nn.Conv2d(chan, chan, 1)
        )

    # 前向传播函数
    def forward(self, x):
        return self.net(x) + x

# 定义 VQGanVAE 类，继承自 nn.Module
class VQGanVAE(nn.Module):
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        *,
        dim,  # 模型的维度
        channels = 3,  # 输入图像的通道数，默认为3
        layers = 4,  # 模型的层数，默认为4
        l2_recon_loss = False,  # 是否使用L2重构损失，默认为False
        use_hinge_loss = True,  # 是否使用hinge loss，默认为True
        vgg = None,  # VGG模型，默认为None
        lookup_free_quantization = True,  # 是否使用无查找表的量化，默认为True
        codebook_size = 65536,  # 量化码书的大小，默认为65536
        vq_kwargs: dict = dict(  # VQ模型的参数，默认为一些参数设置
            codebook_dim = 256,
            decay = 0.8,
            commitment_weight = 1.,
            kmeans_init = True,
            use_cosine_sim = True,
        ),
        lfq_kwargs: dict = dict(  # LFQ模型的参数，默认为一些参数设置
            diversity_gamma = 4.
        ),
        use_vgg_and_gan = True,  # 是否使用VGG和GAN，默认为True
        discr_layers = 4,  # 判别器的层数，默认为4
        **kwargs  # 其他参数
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 将参数按照前缀分组并修剪
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)
        encdec_kwargs, kwargs = groupby_prefix_and_trim('encdec_', kwargs)

        # 设置模型的一些属性
        self.channels = channels
        self.codebook_size = codebook_size
        self.dim_divisor = 2 ** layers

        enc_dec_klass = ResnetEncDec

        # 创建编码器解码器对象
        self.enc_dec = enc_dec_klass(
            dim = dim,
            channels = channels,
            layers = layers,
            **encdec_kwargs
        )

        self.lookup_free_quantization = lookup_free_quantization

        # 根据是否使用无查找表的量化选择量化器类型
        if lookup_free_quantization:
            self.quantizer = LFQ(
                dim = self.enc_dec.encoded_dim,
                codebook_size = codebook_size,
                **lfq_kwargs
            )
        else:
            self.quantizer = VQ(
                dim = self.enc_dec.encoded_dim,
                codebook_size = codebook_size,
                accept_image_fmap = True,
                **vq_kwargs
            )

        # 重构损失函数选择
        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss

        # 如果是灰度图像，则关闭GAN和感知损失
        self._vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan

        if not use_vgg_and_gan:
            return

        # 感知损失
        if exists(vgg):
            self._vgg = vgg

        # GAN相关损失
        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        self.discr = Discriminator(dims = dims, channels = channels)

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    # 获取设备信息
    @property
    def device(self):
        return next(self.parameters()).device

    # 获取VGG模型
    @property
    def vgg(self):
        if exists(self._vgg):
            return self._vgg

        vgg = torchvision.models.vgg16(pretrained = True)
        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        self._vgg = vgg.to(self.device)
        return self._vgg

    # 获取编码后的维度
    @property
    def encoded_dim(self):
        return self.enc_dec.encoded_dim

    # 获取编码特征图的大小
    def get_encoded_fmap_size(self, image_size):
        return self.enc_dec.get_encoded_fmap_size(image_size)

    # 复制模型用于评估
    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy._vgg

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

    # 保存模型
    def save(self, path):
        torch.save(self.state_dict(), path)

    # 加载模型
    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    # 编码函数
    def encode(self, fmap):
        fmap = self.enc_dec.encode(fmap)
        fmap, indices, vq_aux_loss = self.quantizer(fmap)
        return fmap, indices, vq_aux_loss
    # 从编码后的 ids 解码生成图像
    def decode_from_ids(self, ids):
        
        # 如果启用了自由量化查找，则将 ids 打包成字节流
        if self.lookup_free_quantization:
            ids, ps = pack([ids], 'b *')
            # 使用量化器将 ids 转换为 codes
            fmap = self.quantizer.indices_to_codes(ids)
            # 解码 codes 生成 fmap
            fmap, = unpack(fmap, ps, 'b * c')
        else:
            # 根据 ids 获取 codebook 中对应的 codes
            codes = self.codebook[ids]
            # 投影 codes 生成 fmap
            fmap = self.quantizer.project_out(codes)

        # 重新排列 fmap 的维度
        fmap = rearrange(fmap, 'b h w c -> b c h w')
        # 调用 decode 方法生成图像
        return self.decode(fmap)

    # 解码生成图像
    def decode(self, fmap):
        return self.enc_dec.decode(fmap)

    # 前向传播函数
    def forward(
        self,
        img,
        return_loss = False,
        return_discr_loss = False,
        return_recons = False,
        add_gradient_penalty = True
    ):
        # 获取图像的批次、通道数、高度、宽度和设备信息
        batch, channels, height, width, device = *img.shape, img.device

        # 检查高度和宽度是否能被 dim_divisor 整除
        for dim_name, size in (('height', height), ('width', width)):
            assert (size % self.dim_divisor) == 0, f'{dim_name} must be divisible by {self.dim_divisor}'

        # 检查通道数是否与 VQGanVAE 中设置的通道数相等
        assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'

        # 编码输入图像
        fmap, indices, commit_loss = self.encode(img)

        # 解码生成图像
        fmap = self.decode(fmap)

        # 如果不需要返回损失，则直接返回生成图像
        if not return_loss and not return_discr_loss:
            return fmap

        # 确保只返回自编码器损失或鉴别器损失
        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'

        # 如果需要返回鉴别器损失
        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'

            # 分离 fmap 的梯度
            fmap.detach_()
            img.requires_grad_()

            # 获取 fmap 和输入图像的鉴别器 logits
            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))

            # 计算鉴别器损失
            discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

            # 添加梯度惩罚
            if add_gradient_penalty:
                gp = gradient_penalty(img, img_discr_logits)
                loss = discr_loss + gp

            # 如果需要返回重构图像，则返回损失和 fmap
            if return_recons:
                return loss, fmap

            return loss

        # 计算重构损失
        recon_loss = self.recon_loss_fn(fmap, img)

        # 如果不使用 VGG 和 GAN，则直接返回重构损失
        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, fmap

            return recon_loss

        # 计算感知损失
        img_vgg_input = img
        fmap_vgg_input = fmap

        if img.shape[1] == 1:
            # 处理灰度图像用于 VGG
            img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (img_vgg_input, fmap_vgg_input))

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

        # 如果需要返回重构图像，则返回损失和 fmap
        if return_recons:
            return loss, fmap

        return loss
```