# `.\lucidrains\phenaki-pytorch\phenaki_pytorch\cvivit.py`

```
# 导入必要的库
from pathlib import Path
import copy
import math
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.autograd import grad as torch_grad

import torchvision

# 导入 einops 库中的函数
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# 导入自定义的模块
from vector_quantize_pytorch import VectorQuantize, LFQ
from phenaki_pytorch.attention import Attention, Transformer, ContinuousPositionBias

# 定义一些辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 定义 leaky_relu 激活函数
def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

# 移除 vgg 属性的装饰器
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

# 将单个值转换为元组
def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

# 将单个值转换为指定长度的元组
def cast_tuple(val, l = 1):
    return val if isinstance(val, tuple) else (val,) * l

# 计算梯度惩罚
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

# 对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 安全除法，避免分母为零
def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

# 定义 GAN 损失函数

# Hinge 损失函数（判别器）
def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

# Hinge 损失函数（生成器）
def hinge_gen_loss(fake):
    return -fake.mean()

# 二元交叉熵损失函数（判别器）
def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

# 二元交叉熵损失函数（生成器）
def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()

# 计算损失函数对某一层的梯度
def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# 定义判别器模块

class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        downsample = True
    ):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)

        if exists(self.downsample):
            x = self.downsample(x)

        x = (x + res) * (1 / math.sqrt(2))
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        attn_res_layers = (16,),
        max_dim = 512
    # 初始化函数，继承父类的初始化方法
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 将图像大小转换为元组
        image_size = pair(image_size)
        # 计算图像的最小分辨率
        min_image_resolution = min(image_size)

        # 计算层数，根据最小分辨率
        num_layers = int(math.log2(min_image_resolution) - 2)
        # 将注意力层的分辨率转换为元组
        attn_res_layers = cast_tuple(attn_res_layers, num_layers)

        # 初始化块列表
        blocks = []

        # 计算每一层的维度
        layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)]
        # 将每一层的维度限制在最大维度内
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        # 将每一层的输入输出维度组成元组
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        # 初始化块列表和注意力块列表
        blocks = []
        attn_blocks = []

        # 初始化图像分辨率
        image_resolution = min_image_resolution

        # 遍历每一层的输入输出维度
        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            # 计算当前层的编号
            num_layer = ind + 1
            # 判断是否为最后一层
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            # 创建鉴别器块
            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            # 初始化注意力块
            attn_block = None
            if image_resolution in attn_res_layers:
                attn_block = Attention(dim = out_chan)

            attn_blocks.append(attn_block)

            # 更新图像分辨率
            image_resolution //= 2

        # 将块列表和注意力块列表转换为模块列表
        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        # 计算最后一层的维度
        dim_last = layer_dims[-1]

        # 计算下采样因子
        downsample_factor = 2 ** num_layers
        # 计算最后特征图的大小
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        # 计算潜在维度
        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        # 定义输出层
        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding = 1),
            leaky_relu(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )

    # 前向传播函数
    def forward(self, x):

        # 遍历块列表和注意力块列表
        for block, attn_block in zip(self.blocks, self.attn_blocks):
            # 应用块
            x = block(x)

            # 如果存在注意力块
            if exists(attn_block):
                x, ps = pack([x], 'b c *')
                x = rearrange(x, 'b c n -> b n c')
                x = attn_block(x) + x
                x = rearrange(x, 'b n c -> b c n')
                x, = unpack(x, ps, 'b c *')

        # 返回输出结果
        return self.to_logits(x)
# 定义一个函数，用于从视频中选择指定帧的图像
def pick_video_frame(video, frame_indices):
    # 获取视频的批量大小和设备信息
    batch, device = video.shape[0], video.device
    # 重新排列视频张量的维度，将通道维度放在第二个位置
    video = rearrange(video, 'b c f ... -> b f c ...')
    # 创建一个包含批量索引的张量
    batch_indices = torch.arange(batch, device=device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    # 从视频中选择指定帧的图像
    images = video[batch_indices, frame_indices]
    # 重新排列图像张量的维度，将通道维度放在第一个位置
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images

# 定义一个 CViViT 类，实现3D ViT模型，具有分解的空间和时间注意力，并制作成vqgan-vae自动编码器
class CViViT(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 模型维度
        codebook_size,  # 代码簿大小
        image_size,  # 图像大小
        patch_size,  # 图像块大小
        temporal_patch_size,  # 时间块大小
        spatial_depth,  # 空间深度
        temporal_depth,  # 时间深度
        discr_base_dim=16,  # 判别器基础维度
        dim_head=64,  # 头部维度
        heads=8,  # 头部数量
        channels=3,  # 通道数
        use_vgg_and_gan=True,  # 是否使用VGG和GAN
        vgg=None,  # VGG模型
        discr_attn_res_layers=(16,),  # 判别器注意力层分辨率
        use_hinge_loss=True,  # 是否使用hinge损失
        attn_dropout=0.,  # 注意力机制的dropout率
        ff_dropout=0.,  # feed-forward层的dropout率
        lookup_free_quantization=True,  # 是否使用无查找表的量化
        lookup_free_quantization_kwargs: dict = {}  # 无查找表的量化参数
        ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )

        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )

        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)

        # offer look up free quantization
        # https://arxiv.org/abs/2310.05737

        self.lookup_free_quantization = lookup_free_quantization

        if lookup_free_quantization:
            self.vq = LFQ(dim = dim, codebook_size = codebook_size, **lookup_free_quantization_kwargs)
        else:
            self.vq = VectorQuantize(dim = dim, codebook_size = codebook_size, use_cosine_sim = True)

        self.dec_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.dec_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)

        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1 = patch_height, p2 = patch_width)
        )

        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
        )

        # turn off GAN and perceptual loss if grayscale

        self.vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan

        if not use_vgg_and_gan:
            return

        # preceptual loss

        if exists(vgg):
            self.vgg = vgg
        else:
            self.vgg = torchvision.models.vgg16(pretrained = True)
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])

        # gan related losses

        self.discr = Discriminator(
            image_size = self.image_size,
            dim = discr_base_dim,
            channels = channels,
            attn_res_layers = discr_attn_res_layers
        )

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss
    # 计算视频的掩码，用于生成视频的 token
    def calculate_video_token_mask(self, videos, video_frame_mask):
        # 解构赋值，获取视频的高度和宽度
        *_, h, w = videos.shape
        # 获取补丁的高度和宽度
        ph, pw = self.patch_size

        # 断言视频帧掩码的总和减去1必须能被时间补丁大小整除
        assert torch.all(((video_frame_mask.sum(dim = -1) - 1) % self.temporal_patch_size) == 0), 'number of frames must be divisible by temporal patch size, subtracting off the first frame'
        # 将第一帧掩码和其余帧掩码分开
        first_frame_mask, rest_frame_mask = video_frame_mask[:, :1], video_frame_mask[:, 1:]
        # 重新排列其余帧掩码，以适应时间补丁大小
        rest_vq_mask = rearrange(rest_frame_mask, 'b (f p) -> b f p', p = self.temporal_patch_size)
        # 合并第一帧掩码和其余帧掩码的逻辑或结果
        video_mask = torch.cat((first_frame_mask, rest_vq_mask.any(dim = -1)), dim = -1)
        # 重复视频掩码，以匹配视频的高度和宽度
        return repeat(video_mask, 'b f -> b (f hw)', hw = (h // ph) * (w // pw))

    # 获取视频补丁的形状
    def get_video_patch_shape(self, num_frames, include_first_frame = True):
        patch_frames = 0

        if include_first_frame:
            num_frames -= 1
            patch_frames += 1

        patch_frames += (num_frames // self.temporal_patch_size)

        return (patch_frames, *self.patch_height_width)

    # 返回图像 token 的数量
    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    # 根据 token 数量返回帧数
    def frames_per_num_tokens(self, num_tokens):
        tokens_per_frame = self.image_num_tokens

        assert (num_tokens % tokens_per_frame) == 0, f'number of tokens must be divisible by number of tokens per frame {tokens_per_frame}'
        assert (num_tokens > 0)

        pseudo_frames = num_tokens // tokens_per_frames
        return (pseudo_frames - 1) * self.temporal_patch_size + 1

    # 根据帧数返回 token 数量
    def num_tokens_per_frames(self, num_frames, include_first_frame = True):
        image_num_tokens = self.image_num_tokens

        total_tokens = 0

        if include_first_frame:
            num_frames -= 1
            total_tokens += image_num_tokens

        assert (num_frames % self.temporal_patch_size) == 0

        return total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens

    # 用于评估的模型拷贝
    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    # 重写 state_dict 方法
    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    # 重写 load_state_dict 方法
    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    # 加载模型
    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt)

    # 根据 codebook 索引解码
    def decode_from_codebook_indices(self, indices):
        if self.lookup_free_quantization:
            codes = self.vq.indices_to_codes(indices)
        else:
            codes = self.vq.codebook[indices]

        return self.decode(codes)

    # 返回补丁的高度和宽度
    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    # 编码 tokens
    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # encode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        return tokens

    # 解码 tokens
    def decode(
        self,
        tokens
        ):
        # 获取 tokens 的 batch 大小
        b = tokens.shape[0]
        # 获取 patch 的高度和宽度
        h, w = self.patch_height_width

        # 如果 tokens 的维度为 3，则重新排列 tokens 的维度
        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        # 获取视频形状的元组
        video_shape = tuple(tokens.shape[:-1])

        # 解码 - 时间维度

        # 重新排列 tokens 的维度
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        # 对 tokens 进行时间维度的解码
        tokens = self.dec_temporal_transformer(tokens, video_shape = video_shape)

        # 重新排列 tokens 的维度
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        # 解码 - 空间维度

        # 重新排列 tokens 的维度
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        # 获取空间相对位置偏置
        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        # 对 tokens 进行空间维度的解码
        tokens = self.dec_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        # 重新排列 tokens 的维度
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # 转换为像素

        # 获取第一帧 token 和其余帧 tokens
        first_frame_token, rest_frames_tokens = tokens[:, :1], tokens[:, 1:]

        # 将第一帧转换为像素
        first_frame = self.to_pixels_first_frame(first_frame_token)

        # 将其余帧转换为像素
        rest_frames = self.to_pixels(rest_frames_tokens)

        # 拼接重构视频
        recon_video = torch.cat((first_frame, rest_frames), dim = 2)

        # 返回重构视频
        return recon_video

    def forward(
        self,
        video,
        mask = None,
        return_recons = False,
        return_recons_only = False,
        return_discr_loss = False,
        apply_grad_penalty = True,
        return_only_codebook_ids = False
```