# `.\lucidrains\bit-diffusion\bit_diffusion\bit_diffusion.py`

```py
    # 导入所需的库
    import math
    from pathlib import Path
    from functools import partial
    from multiprocessing import cpu_count

    import torch
    from torch import nn, einsum
    from torch.special import expm1
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    from torch.optim import Adam
    from torchvision import transforms as T, utils

    from einops import rearrange, reduce, repeat
    from einops.layers.torch import Rearrange

    from PIL import Image
    from tqdm.auto import tqdm
    from ema_pytorch import EMA

    from accelerate import Accelerator

    # 常量定义
    BITS = 8

    # 辅助函数

    def exists(x):
        return x is not None

    def default(val, d):
        if exists(val):
            return val
        return d() if callable(d) else d

    def cycle(dl):
        while True:
            for data in dl:
                yield data

    def has_int_squareroot(num):
        return (math.sqrt(num) ** 2) == num

    def num_to_groups(num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr

    def convert_image_to(pil_img_type, image):
        if image.mode != pil_img_type:
            return image.convert(pil_img_type)
        return image

    # 小型辅助模块

    class Residual(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x, *args, **kwargs):
            return self.fn(x, *args, **kwargs) + x

    def Upsample(dim, dim_out = None):
        return nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
        )

    def Downsample(dim, dim_out = None):
        return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

    class LayerNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

        def forward(self, x):
            eps = 1e-5 if x.dtype == torch.float32 else 1e-3
            var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
            mean = torch.mean(x, dim = 1, keepdim = True)
            return (x - mean) * (var + eps).rsqrt() * self.g

    class PreNorm(nn.Module):
        def __init__(self, dim, fn):
            super().__init__()
            self.fn = fn
            self.norm = LayerNorm(dim)

        def forward(self, x):
            x = self.norm(x)
            return self.fn(x)

    # 位置嵌入

    class LearnedSinusoidalPosEmb(nn.Module):
        """ following @crowsonkb 's lead with learned sinusoidal pos emb """
        """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

        def __init__(self, dim):
            super().__init__()
            assert (dim % 2) == 0
            half_dim = dim // 2
            self.weights = nn.Parameter(torch.randn(half_dim))

        def forward(self, x):
            x = rearrange(x, 'b -> b 1')
            freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
            fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
            fouriered = torch.cat((x, fouriered), dim = -1)
            return fouriered

    # 构建块模块

    class Block(nn.Module):
        def __init__(self, dim, dim_out, groups = 8):
            super().__init__()
            self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
            self.norm = nn.GroupNorm(groups, dim_out)
            self.act = nn.SiLU()

        def forward(self, x, scale_shift = None):
            x = self.proj(x)
            x = self.norm(x)

            if exists(scale_shift):
                scale, shift = scale_shift
                x = x * (scale + 1) + shift

            x = self.act(x)
            return x

    class ResnetBlock(nn.Module):
    # 初始化函数，定义神经网络结构
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        # 调用父类的初始化函数
        super().__init__()
        # 如果存在时间嵌入维度，则创建包含激活函数和线性层的序列模块
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        # 创建第一个块
        self.block1 = Block(dim, dim_out, groups = groups)
        # 创建第二个块
        self.block2 = Block(dim_out, dim_out, groups = groups)
        # 如果输入维度和输出维度不相等，则使用卷积层进行维度转换，否则使用恒等映射
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # 前向传播函数
    def forward(self, x, time_emb = None):

        scale_shift = None
        # 如果存在时间嵌入模块和时间嵌入向量，则进行处理
        if exists(self.mlp) and exists(time_emb):
            # 对时间嵌入向量进行处理
            time_emb = self.mlp(time_emb)
            # 重新排列时间嵌入向量的维度
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # 将时间嵌入向量分成两部分，用于缩放和平移
            scale_shift = time_emb.chunk(2, dim = 1)

        # 使用第一个块处理输入数据
        h = self.block1(x, scale_shift = scale_shift)

        # 使用第二个块处理第一个块的输出
        h = self.block2(h)

        # 返回块处理后的结果与输入数据经过维度转换后的结果的和
        return h + self.res_conv(x)
# 定义一个线性注意力模块，继承自 nn.Module 类
class LinearAttention(nn.Module):
    # 初始化函数，接受维度 dim、头数 heads 和头维度 dim_head 作为参数
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        # 缩放因子为头维度的倒数
        self.scale = dim_head ** -0.5
        # 头数
        self.heads = heads
        # 隐藏维度为头维度乘以头数
        hidden_dim = dim_head * heads
        # 将输入转换为查询、键、值的形式
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        # 输出转换层，包含一个卷积层和一个 LayerNorm 层
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    # 前向传播函数
    def forward(self, x):
        # 获取输入张量的形状信息
        b, c, h, w = x.shape
        # 将输入通过查询、键、值转换层，并按维度 1 切分为三部分
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # 将查询、键、值按照指定维度重排
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # 对查询和键进行 softmax 操作
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        # 对查询进行缩放
        q = q * self.scale
        # 对值进行归一化
        v = v / (h * w)

        # 计算上下文信息
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # 计算输出
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # 重排输出张量的维度
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# 定义一个注意力模块，继承自 nn.Module 类
class Attention(nn.Module):
    # 初始化函数，接受维度 dim、头数 heads 和头维度 dim_head 作为参数
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        # 缩放因子为头维度的倒数
        self.scale = dim_head ** -0.5
        # 头数
        self.heads = heads
        # 隐藏维度为头维度乘以头数
        hidden_dim = dim_head * heads
        # 将输入转换为查询、键、值的形式
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        # 输出转换层，包含一个卷积层
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    # 前向传播函数
    def forward(self, x):
        # 获取输入张量的形状信息
        b, c, h, w = x.shape
        # 将输入通过查询、键、值转换层，并按维度 1 切分为三部分
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # 将查询、键、值按照指定维度重排
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # 对查询进行缩放
        q = q * self.scale

        # 计算相似度
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        # 对相似度进行 softmax 操作
        attn = sim.softmax(dim = -1)
        # 计算输出
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        # 重排输出张量的维度
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# 定义一个 Unet 模型，继承自 nn.Module 类
class Unet(nn.Module):
    # 初始化函数，接受维度 dim、初始维度 init_dim、维度倍增 dim_mults、通道数 channels、位数 bits、ResNet 块组数 resnet_block_groups 和学习的正弦维度 learned_sinusoidal_dim 作为参数
    def __init__(
        self,
        dim,
        init_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        bits = BITS,
        resnet_block_groups = 8,
        learned_sinusoidal_dim = 16
    ):
        # 调用父类的构造函数
        super().__init__()

        # 确定维度
        channels *= bits
        self.channels = channels

        input_channels = channels * 2

        # 初始化维度
        init_dim = default(init_dim, dim)
        # 创建一个卷积层，输入通道数为input_channels，输出通道数为init_dim，卷积核大小为7，填充为3
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # 计算不同层次的维度
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 使用ResnetBlock类创建一个部分函数block_klass，其中groups参数为resnet_block_groups
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # 时间嵌入
        time_dim = dim * 4

        # 创建一个LearnedSinusoidalPosEmb对象sinu_pos_emb
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        # 创建一个包含线性层和激活函数的神经网络模块time_mlp
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 层
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # 遍历不同层次的维度
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # 向downs列表中添加模块列表
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        # 创建一个ResnetBlock对象mid_block1
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        # 创建一个包含注意力机制的Residual对象mid_attn
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        # 创建一个ResnetBlock对象mid_block2
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        # 反向遍历不同层次的维度
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            # ���ups列表中添加模块列表
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # 创建一个ResnetBlock对象final_res_block
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        # 创建一个卷积层final_conv，输入通道数为dim，输出通道数为channels，卷积核大小为1
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time, x_self_cond = None):

        # 如果x_self_cond为None，则创建一个与x相同形状的全零张量
        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        # 在通道维度上拼接x_self_cond和x
        x = torch.cat((x_self_cond, x), dim = 1)

        # 将输入数据x通过init_conv卷积层
        x = self.init_conv(x)
        r = x.clone()

        # 通过时间嵌入网络计算时间信息t
        t = self.time_mlp(time)

        h = []

        # 遍历downs列表中的模块列表
        for block1, block2, attn, downsample in self.downs:
            # 通过block1进行处理
            x = block1(x, t)
            h.append(x)

            # 通过block2进行处理
            x = block2(x, t)
            # 通过attn进行处理
            x = attn(x)
            h.append(x)

            # 通过downsample进行处理
            x = downsample(x)

        # 通过mid_block1进行处理
        x = self.mid_block1(x, t)
        # 通过mid_attn进行处理
        x = self.mid_attn(x)
        # 通过mid_block2进行处理
        x = self.mid_block2(x, t)

        # 遍历ups列表中的模块列表
        for block1, block2, attn, upsample in self.ups:
            # 在通道维度上拼接x和h中的张量
            x = torch.cat((x, h.pop()), dim = 1)
            # 通过block1进行处理
            x = block1(x, t)

            # 在通道维度上拼接x和h中的张量
            x = torch.cat((x, h.pop()), dim = 1)
            # 通过block2进行处理
            x = block2(x, t)
            # 通过attn进行处理
            x = attn(x)

            # 通过upsample进行处理
            x = upsample(x)

        # 在通道维度上拼接x和r
        x = torch.cat((x, r), dim = 1)

        # 通过final_res_block进行处理
        x = self.final_res_block(x, t)
        return self.final_conv(x)
# 将十进制数转换为位表示，并反向转换

def decimal_to_bits(x, bits = BITS):
    """将范围在0到1之间的图像张量转换为范围在-1到1之间的位张量"""
    device = x.device

    # 将图像张量乘以255并取整，限制在0到255之间
    x = (x * 255).int().clamp(0, 255)

    # 创建位掩码
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b c h w -> b c 1 h w')

    # 将图像张量转换为位张量
    bits = ((x & mask) != 0).float()
    bits = rearrange(bits, 'b c d h w -> b (c d) h w')
    bits = bits * 2 - 1
    return bits

def bits_to_decimal(x, bits = BITS):
    """将范围在-1到1之间的位转换为范围在0到1之间的图像张量"""
    device = x.device

    # 将位张量转换为整数张量
    x = (x > 0).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)

    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b (c d) h w -> b c d h w', d = bits)
    dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
    return (dec / 255).clamp(0., 1.)

# 位扩散类

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # 不确定这是否考虑了在离散版本中将beta剪切为0.999

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class BitDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        use_ddim = False,
        noise_schedule = 'cosine',
        time_difference = 0.,
        bit_scale = 1.
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.bit_scale = bit_scale

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # 在论文中提出���与time_next相加，作为修复自我条件不足和在采样时间步数小于400时降低FID的方法

        self.time_difference = time_difference

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    # 从 DDPM 模型中采样生成图像
    def ddpm_sample(self, shape, time_difference = None):
        # 获取批次大小和设备信息
        batch, device = shape[0], self.device

        # 设置时间差，默认为 self.time_difference
        time_difference = default(time_difference, self.time_difference)

        # 获取采样时间步骤对
        time_pairs = self.get_sampling_timesteps(batch, device = device)

        # 生成随机噪声图像
        img = torch.randn(shape, device=device)

        x_start = None

        # 遍历时间步骤对
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            # 添加时间延迟
            time_next = (time_next - self.time_difference).clamp(min = 0.)

            # 获取噪声条件
            noise_cond = self.log_snr(time)

            # 获取预测的 x0
            x_start = self.model(img, noise_cond, x_start)

            # 限制 x0 的范围
            x_start.clamp_(-self.bit_scale, self.bit_scale)

            # 获取 log(snr)
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            # 获取时间和下一个时间的 alpha 和 sigma
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # 推导后验均值和方差
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # 获取噪声
            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + (0.5 * log_variance).exp() * noise

        return bits_to_decimal(img)

    # 无梯度计算的 DDIM 模型采样函数
    @torch.no_grad()
    def ddim_sample(self, shape, time_difference = None):
        # 获取批次大小和设备信息
        batch, device = shape[0], self.device

        # 设置时间差，默认为 self.time_difference
        time_difference = default(time_difference, self.time_difference)

        # 获取采样时间步骤对
        time_pairs = self.get_sampling_timesteps(batch, device = device)

        # 生成随机噪声图像
        img = torch.randn(shape, device = device)

        x_start = None

        # 遍历时间步骤对
        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # 添加时间延迟
            times_next = (times_next - time_difference).clamp(min = 0.)

            # 获取时间和噪声水平
            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr, padded_log_snr_next = map(partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            # 预测 x0
            x_start = self.model(img, log_snr, x_start)

            # 限制 x0 的范围
            x_start.clamp_(-self.bit_scale, self.bit_scale)

            # 获取预测的噪声
            pred_noise = (img - alpha * x_start) / sigma.clamp(min = 1e-8)

            # 计算下一个 x
            img = x_start * alpha_next + pred_noise * sigma_next

        return bits_to_decimal(img)

    # 采样函数，根据是否使用 DDIM 选择不同的采样方法
    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))
    # 定义前向传播函数，接受图像和其他参数
    def forward(self, img, *args, **kwargs):
        # 解包图像的形状和设备信息
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 断言图像的高度和宽度必须为指定的图像大小
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # 生成随机采样时间

        times = torch.zeros((batch,), device=device).float().uniform_(0, 1.)

        # 将图像转换为比特表示

        img = decimal_to_bits(img) * self.bit_scale

        # 生成噪声样本

        noise = torch.randn_like(img)

        # 计算噪声水平
        noise_level = self.log_snr(times)
        # 将噪声水平填充到与图像相同的维度
        padded_noise_level = right_pad_dims_to(img, noise_level)
        # 将噪声水平转换为 alpha 和 sigma
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)

        # 添加噪声到图像
        noised_img = alpha * img + sigma * noise

        # 如果进行自条件训练，50%的概率从当前时间预测 x_start，并使用 unet 进行条件
        # 这种技术会使训练速度减慢 25%，但似乎显著降低 FID

        self_cond = None
        if torch.rand((1)) < 0.5:
            with torch.no_grad():
                # 使用模型预测 x_start，并分离计算图
                self_cond = self.model(noised_img, noise_level).detach_()

        # 预测并进行梯度下降步骤

        pred = self.model(noised_img, noise_level, self_cond)

        # 返回预测值和真实值的均方误差损失
        return F.mse_loss(pred, img)
# dataset classes

# 定义 Dataset 类，继承自 torch.utils.data.Dataset
class Dataset(Dataset):
    # 初始化函数
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        pil_img_type = None
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置属性
        self.folder = folder
        self.image_size = image_size
        # 获取指定扩展名的所有文件路径
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        # 部分转换函数
        maybe_convert_fn = partial(convert_image_to, pil_img_type) if exists(pil_img_type) else nn.Identity()

        # 数据转换操作
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    # 返回数据集的长度
    def __len__(self):
        return len(self.paths)

    # 获取指定索引的数据
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

# 定义 Trainer 类
class Trainer(object):
    # 初始化函数
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        pil_img_type = None
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 初始化加速器
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # 设置扩散模型
        self.model = diffusion_model

        # 检查样本数量是否有整数平方根
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        # 创建数据集
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, pil_img_type = pil_img_type)
        # 创建数据加载器
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        # 准备数据加载器
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        # 创建优化器
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        # 如果是主进程
        if self.accelerator.is_main_process:
            # 创建指数移动平均模型
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

            # 设置结果文件夹路径
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        # 步数计数器
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        # 使用加速器准备模型、数据加载器和优化器
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    # 保存模型
    def save(self, milestone):
        # 如果不是本地主进程，则返回
        if not self.accelerator.is_local_main_process:
            return

        # 保存模型相关数据
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        # 将数据保存到文件
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    # 加载指定里程碑的模型数据
    def load(self, milestone):
        # 从文件中加载模型数据
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        # 获取未包装的模型对象
        model = self.accelerator.unwrap_model(self.model)
        # 加载模型的状态字典
        model.load_state_dict(data['model'])

        # 设置当前步数为加载的数据中的步数
        self.step = data['step']
        # 加载优化器的状态字典
        self.opt.load_state_dict(data['opt'])
        # 加载指数移动平均模型的状态字典
        self.ema.load_state_dict(data['ema'])

        # 如果加速器的缩放器和加载的数据中的缩放器都存在，则加载缩放器的状态字典
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # 训练模型
    def train(self):
        # 获取加速器和设备
        accelerator = self.accelerator
        device = accelerator.device

        # 使用 tqdm 显示训练进度条
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            # 在未达到训练步数之前循环训练
            while self.step < self.train_num_steps:

                total_loss = 0.

                # 根据梯度累积的次数循环
                for _ in range(self.gradient_accumulate_every):
                    # 从数据加载器中获取数据并移动到设备上
                    data = next(self.dl).to(device)

                    # 使用自动混合精度计算模型的损失
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    # 反向传播计算梯度
                    self.accelerator.backward(loss)

                # 更新进度条显示损失值
                pbar.set_description(f'loss: {total_loss:.4f}')

                # 等待所有进程完成当前步骤
                accelerator.wait_for_everyone()

                # 更新优化器参数
                self.opt.step()
                self.opt.zero_grad()

                # 等待所有进程完成当前步骤
                accelerator.wait_for_everyone()

                # 如果是主进程
                if accelerator.is_main_process:
                    # 将指数移动平均模型移动到设备上并更新
                    self.ema.to(device)
                    self.ema.update()

                    # 如果步数不为0���可以保存和采样
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        # 将指数移动平均模型设置为评估模式
                        self.ema.ema_model.eval()

                        # 使用无梯度计算生成样本图像
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        # 拼接所有生成的图像并保存
                        all_images = torch.cat(all_images_list, dim=0)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                # 更新步数并进度条
                self.step += 1
                pbar.update(1)

        # 打印训练完成信息
        accelerator.print('training complete')
```