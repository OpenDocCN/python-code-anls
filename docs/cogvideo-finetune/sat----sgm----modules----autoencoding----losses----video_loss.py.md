# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\losses\video_loss.py`

```py
# 导入所需的类型提示
from typing import Any, Union
# 导入计算以2为底的对数函数
from math import log2
# 导入类型检查装饰器
from beartype import beartype

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入张量类型
from torch import Tensor
# 导入 PyTorch 的梯度计算功能
from torch.autograd import grad as torch_grad
# 导入自动混合精度的上下文管理器
from torch.cuda.amp import autocast

# 导入 torchvision 库
import torchvision
# 导入 VGG16 权重
from torchvision.models import VGG16_Weights
# 导入 einops 库用于张量重排和操作
from einops import rearrange, einsum, repeat
from einops.layers.torch import Rearrange
# 导入 kornia 库中的 3D 滤波器
from kornia.filters import filter3d

# 导入自定义模块中的类
from ..magvit2_pytorch import Residual, FeedForward, LinearSpaceAttention
# 导入 LPIPS 模块
from .lpips import LPIPS

# 导入 VQVAE 编码器中的类
from sgm.modules.autoencoding.vqvae.movq_enc_3d import CausalConv3d, DownSample3D
# 从配置中实例化对象的工具
from sgm.util import instantiate_from_config


# 检查值是否存在的函数
def exists(v):
    # 返回值是否不为 None
    return v is not None


# 将单个值转换为元组的函数
def pair(t):
    # 如果 t 是元组，直接返回；否则返回 (t, t)
    return t if isinstance(t, tuple) else (t, t)


# 创建 LeakyReLU 激活函数的工厂函数
def leaky_relu(p=0.1):
    # 返回 LeakyReLU 激活函数的实例，负斜率为 p
    return nn.LeakyReLU(p)


# 定义对抗损失的判别器损失函数
def hinge_discr_loss(fake, real):
    # 返回对抗损失，包含真实和伪造样本
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


# 定义对抗损失的生成器损失函数
def hinge_gen_loss(fake):
    # 返回伪造样本的对抗损失
    return -fake.mean()


# 定义计算损失对层的梯度的函数，使用自动混合精度
@autocast(enabled=False)
@beartype
def grad_layer_wrt_loss(loss: Tensor, layer: nn.Parameter):
    # 计算损失相对于指定层的梯度，并返回其张量
    return torch_grad(outputs=loss, inputs=layer, grad_outputs=torch.ones_like(loss), retain_graph=True)[0].detach()


# 从视频中选择特定帧的函数
def pick_video_frame(video, frame_indices):
    # 获取批次大小和设备信息
    batch, device = video.shape[0], video.device
    # 重排视频张量，使帧维度在中间
    video = rearrange(video, "b c f ... -> b f c ...")
    # 创建批次索引
    batch_indices = torch.arange(batch, device=device)
    # 将批次索引重排为列向量
    batch_indices = rearrange(batch_indices, "b -> b 1")
    # 根据帧索引选择特定帧
    images = video[batch_indices, frame_indices]
    # 将选择的帧重排回原始格式
    images = rearrange(images, "b 1 c ... -> b c ...")
    # 返回选择的图像
    return images


# 定义梯度惩罚的计算函数
def gradient_penalty(images, output):
    # 获取批次大小
    batch_size = images.shape[0]

    # 计算输出相对于输入图像的梯度
    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # 将梯度重排为适当的形状
    gradients = rearrange(gradients, "b ... -> b (...)")
    # 返回梯度惩罚的均值
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# 定义带有抗锯齿下采样（模糊池化）的判别器类
class Blur(nn.Module):
    # 初始化函数
    def __init__(self):
        super().__init__()
        # 定义模糊滤波器的权重
        f = torch.Tensor([1, 2, 1])
        # 注册权重为缓冲区
        self.register_buffer("f", f)

    # 前向传播函数
    def forward(self, x, space_only=False, time_only=False):
        # 确保不同时指定空间和时间的选项
        assert not (space_only and time_only)

        # 获取模糊滤波器的权重
        f = self.f

        # 根据参数调整滤波器的形状
        if space_only:
            # 计算空间滤波器的外积
            f = einsum("i, j -> i j", f, f)
            # 重排为适合的形状
            f = rearrange(f, "... -> 1 1 ...")
        elif time_only:
            # 重排为时间维度的形状
            f = rearrange(f, "f -> 1 f 1 1")
        else:
            # 计算三维滤波器的外积
            f = einsum("i, j, k -> i j k", f, f, f)
            # 重排为适合的形状
            f = rearrange(f, "... -> 1 ...")

        # 检查输入是否为图像张量
        is_images = x.ndim == 4

        # 如果输入是图像，重排为合适的形状
        if is_images:
            x = rearrange(x, "b c h w -> b c 1 h w")

        # 使用 3D 滤波器对输入进行滤波
        out = filter3d(x, f, normalized=True)

        # 如果输入是图像，重排输出回原始格式
        if is_images:
            out = rearrange(out, "b c 1 h w -> b c h w")

        # 返回滤波后的输出
        return out


# 定义判别器模块的类
class DiscriminatorBlock(nn.Module):
    # 初始化方法，接收输入通道数、滤波器数量以及下采样和抗锯齿下采样标志
        def __init__(self, input_channels, filters, downsample=True, antialiased_downsample=True):
            # 调用父类初始化方法
            super().__init__()
            # 创建一个卷积层，用于下采样，卷积核大小为1
            self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))
    
            # 构建一个序列网络，包括两个卷积层和激活函数
            self.net = nn.Sequential(
                # 第一个卷积层，卷积核大小为3，添加填充
                nn.Conv2d(input_channels, filters, 3, padding=1),
                # 使用泄漏 ReLU 激活函数
                leaky_relu(),
                # 第二个卷积层，卷积核大小为3，添加填充
                nn.Conv2d(filters, filters, 3, padding=1),
                # 再次使用泄漏 ReLU 激活函数
                leaky_relu(),
            )
    
            # 根据是否需要抗锯齿下采样决定是否创建模糊层
            self.maybe_blur = Blur() if antialiased_downsample else None
    
            # 根据下采样标志构建下采样网络
            self.downsample = (
                # 如果需要下采样，构建一个包括重排和卷积的序列
                nn.Sequential(
                    Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2), nn.Conv2d(filters * 4, filters, 1)
                )
                # 如果不需要下采样，则为 None
                if downsample
                else None
            )
    
        # 前向传播方法，接收输入 x
        def forward(self, x):
            # 通过残差卷积层计算结果
            res = self.conv_res(x)
    
            # 通过网络计算 x 的输出
            x = self.net(x)
    
            # 如果存在下采样网络
            if exists(self.downsample):
                # 如果存在模糊层，则应用模糊处理
                if exists(self.maybe_blur):
                    x = self.maybe_blur(x, space_only=True)
    
                # 应用下采样
                x = self.downsample(x)
    
            # 将 x 和残差相加并缩放
            x = (x + res) * (2**-0.5)
            # 返回最终的输出
            return x
# 定义判别器类，继承自 nn.Module
class Discriminator(nn.Module):
    # 使用 @beartype 装饰器进行参数类型检查
    @beartype
    def __init__(
        self,
        *,
        dim,  # 特征维度
        image_size,  # 输入图像尺寸
        channels=3,  # 输入通道数，默认为3（RGB图像）
        max_dim=512,  # 最大特征维度
        attn_heads=8,  # 注意力头的数量
        attn_dim_head=32,  # 每个注意力头的维度
        linear_attn_dim_head=8,  # 线性注意力头的维度
        linear_attn_heads=16,  # 线性注意力头的数量
        ff_mult=4,  # 前馈网络的扩展倍数
        antialiased_downsample=False,  # 是否使用抗锯齿下采样
    ):
        # 调用父类构造函数
        super().__init__()
        # 将图像尺寸转换为元组形式
        image_size = pair(image_size)
        # 获取最小图像分辨率
        min_image_resolution = min(image_size)

        # 计算网络层数
        num_layers = int(log2(min_image_resolution) - 2)

        # 初始化层块列表
        blocks = []

        # 计算每层的特征维度
        layer_dims = [channels] + [(dim * 4) * (2**i) for i in range(num_layers + 1)]
        # 限制特征维度不超过最大值
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        # 创建每层输入和输出特征维度的元组
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        # 初始化层块和注意力块列表
        blocks = []
        attn_blocks = []

        # 当前图像分辨率
        image_resolution = min_image_resolution

        # 为每层创建判别器块和注意力块
        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1  # 当前层的序号
            is_not_last = ind != (len(layer_dims_in_out) - 1)  # 是否为最后一层

            # 创建判别器块
            block = DiscriminatorBlock(
                in_chan, out_chan, downsample=is_not_last, antialiased_downsample=antialiased_downsample
            )

            # 创建注意力块
            attn_block = nn.Sequential(
                Residual(LinearSpaceAttention(dim=out_chan, heads=linear_attn_heads, dim_head=linear_attn_dim_head)),
                Residual(FeedForward(dim=out_chan, mult=ff_mult, images=True)),
            )

            # 将判别器块和注意力块添加到层块列表中
            blocks.append(nn.ModuleList([block, attn_block]))

            # 更新图像分辨率
            image_resolution //= 2

        # 将所有块组成一个模块列表
        self.blocks = nn.ModuleList(blocks)

        # 获取最后一层的特征维度
        dim_last = layer_dims[-1]

        # 计算下采样因子
        downsample_factor = 2**num_layers
        # 计算最后特征图的尺寸
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        # 计算潜在维度
        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        # 构建输出层
        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),  # 卷积层，保持特征图尺寸
            leaky_relu(),  # 激活函数
            Rearrange("b ... -> b (...)"),  # 重排特征图形状
            nn.Linear(latent_dim, 1),  # 全连接层，输出一个值
            Rearrange("b 1 -> b"),  # 重排输出形状
        )

    # 定义前向传播函数
    def forward(self, x):
        # 遍历所有块，进行前向传播
        for block, attn_block in self.blocks:
            x = block(x)  # 通过判别器块
            x = attn_block(x)  # 通过注意力块

        # 返回最终输出
        return self.to_logits(x)


# 定义三维判别器块类，继承自 nn.Module
class DiscriminatorBlock3D(nn.Module):
    def __init__(
        self,
        input_channels,  # 输入通道数
        filters,  # 输出滤波器数量
        antialiased_downsample=True,  # 是否使用抗锯齿下采样
    ):
        # 调用父类构造函数
        super().__init__()
        # 创建一维卷积层，用于下采样
        self.conv_res = nn.Conv3d(input_channels, filters, 1, stride=2)

        # 定义主网络结构
        self.net = nn.Sequential(
            nn.Conv3d(input_channels, filters, 3, padding=1),  # 卷积层，保持尺寸
            leaky_relu(),  # 激活函数
            nn.Conv3d(filters, filters, 3, padding=1),  # 继续卷积
            leaky_relu(),  # 激活函数
        )

        # 根据是否使用抗锯齿下采样选择模糊操作
        self.maybe_blur = Blur() if antialiased_downsample else None

        # 定义下采样网络
        self.downsample = nn.Sequential(
            Rearrange("b c (f p1) (h p2) (w p3) -> b (c p1 p2 p3) f h w", p1=2, p2=2, p3=2),  # 重排输入形状
            nn.Conv3d(filters * 8, filters, 1),  # 1x1卷积层，调整通道数
        )
    # 定义前向传播函数，接收输入 x
        def forward(self, x):
            # 通过卷积残差网络处理输入 x，得到残差结果
            res = self.conv_res(x)
    
            # 通过主网络处理输入 x
            x = self.net(x)
    
            # 如果存在下采样操作
            if exists(self.downsample):
                # 如果存在可能的模糊处理
                if exists(self.maybe_blur):
                    # 对 x 进行模糊处理，只在空间维度上应用
                    x = self.maybe_blur(x, space_only=True)
    
                # 对 x 进行下采样处理
                x = self.downsample(x)
    
            # 将 x 和残差相加并进行归一化处理
            x = (x + res) * (2**-0.5)
            # 返回处理后的结果 x
            return x
# 定义一个3D鉴别器模块，继承自nn.Module
class DiscriminatorBlock3DWithfirstframe(nn.Module):
    # 初始化方法，定义输入通道数、滤波器等参数
    def __init__(
        self,
        input_channels,
        filters,
        antialiased_downsample=True,
        pad_mode="first",
    ):
        # 调用父类构造函数
        super().__init__()
        # 创建一个3D下采样层，带有卷积和时间压缩
        self.downsample_res = DownSample3D(
            in_channels=input_channels,
            out_channels=filters,
            with_conv=True,
            compress_time=True,
        )

        # 定义网络结构，包括两个因果卷积和激活函数
        self.net = nn.Sequential(
            CausalConv3d(input_channels, filters, kernel_size=3, pad_mode=pad_mode),
            leaky_relu(),
            CausalConv3d(filters, filters, kernel_size=3, pad_mode=pad_mode),
            leaky_relu(),
        )

        # 如果启用抗锯齿下采样，则初始化模糊层
        self.maybe_blur = Blur() if antialiased_downsample else None

        # 创建第二个3D下采样层
        self.downsample = DownSample3D(
            in_channels=filters,
            out_channels=filters,
            with_conv=True,
            compress_time=True,
        )

    # 前向传播方法
    def forward(self, x):
        # 通过下采样层处理输入
        res = self.downsample_res(x)

        # 通过网络结构处理输入
        x = self.net(x)

        # 如果下采样层存在
        if exists(self.downsample):
            # 如果模糊层存在，则对输入进行模糊处理
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only=True)

            # 对处理后的输入进行下采样
            x = self.downsample(x)

        # 将处理结果与残差相加并进行归一化
        x = (x + res) * (2**-0.5)
        return x


# 定义一个3D鉴别器类，继承自nn.Module
class Discriminator3D(nn.Module):
    # 使用类型注释进行参数检查
    @beartype
    def __init__(
        self,
        *,
        dim,
        image_size,
        frame_num,
        channels=3,
        max_dim=512,
        linear_attn_dim_head=8,
        linear_attn_heads=16,
        ff_mult=4,
        antialiased_downsample=False,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 将图像大小转换为一对（宽，高）的形式
        image_size = pair(image_size)
        # 获取图像尺寸中的最小值
        min_image_resolution = min(image_size)

        # 计算层数，基于最小图像分辨率
        num_layers = int(log2(min_image_resolution) - 2)
        # 计算时间层数，基于帧数
        temporal_num_layers = int(log2(frame_num))
        # 将时间层数存储为类属性
        self.temporal_num_layers = temporal_num_layers

        # 生成每层的维度，第一层为通道数，后续层基于当前层的维度和数量
        layer_dims = [channels] + [(dim * 4) * (2**i) for i in range(num_layers + 1)]
        # 确保每层维度不超过最大维度
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        # 创建层的输入输出维度元组
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        # 初始化块列表
        blocks = []

        # 最小图像分辨率
        image_resolution = min_image_resolution
        # 帧分辨率
        frame_resolution = frame_num

        # 遍历每一对输入输出通道
        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            # 当前层数
            num_layer = ind + 1
            # 是否不是最后一层
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            # 根据时间层数决定使用的块类型
            if ind < temporal_num_layers:
                # 创建3D判别块
                block = DiscriminatorBlock3D(
                    in_chan,
                    out_chan,
                    antialiased_downsample=antialiased_downsample,
                )

                # 将块添加到列表
                blocks.append(block)

                # 将帧分辨率减半
                frame_resolution //= 2
            else:
                # 创建2D判别块
                block = DiscriminatorBlock(
                    in_chan,
                    out_chan,
                    downsample=is_not_last,
                    antialiased_downsample=antialiased_downsample,
                )
                # 创建注意力块
                attn_block = nn.Sequential(
                    Residual(
                        LinearSpaceAttention(dim=out_chan, heads=linear_attn_heads, dim_head=linear_attn_dim_head)
                    ),
                    Residual(FeedForward(dim=out_chan, mult=ff_mult, images=True)),
                )

                # 将块和注意力块作为模块列表添加
                blocks.append(nn.ModuleList([block, attn_block]))

            # 将图像分辨率减半
            image_resolution //= 2

        # 将所有块转为模块列表
        self.blocks = nn.ModuleList(blocks)

        # 获取最后一层的维度
        dim_last = layer_dims[-1]

        # 计算下采样因子
        downsample_factor = 2**num_layers
        # 计算最后特征图的大小
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        # 计算潜在维度
        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        # 创建最后的输出序列
        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),
            leaky_relu(),
            Rearrange("b ... -> b (...)"),
            nn.Linear(latent_dim, 1),
            Rearrange("b 1 -> b"),
        )

    # 前向传播方法
    def forward(self, x):
        # 遍历每个块
        for i, layer in enumerate(self.blocks):
            # 如果是时间层数内
            if i < self.temporal_num_layers:
                x = layer(x)  # 通过块传递数据
                # 如果是最后一层，将数据重排
                if i == self.temporal_num_layers - 1:
                    x = rearrange(x, "b c f h w -> (b f) c h w")
            else:
                # 拆分块和注意力块
                block, attn_block = layer
                x = block(x)  # 通过判别块传递数据
                x = attn_block(x)  # 通过注意力块传递数据

        # 返回最终的逻辑输出
        return self.to_logits(x)
# 定义一个 3D 判别器类，继承自 nn.Module
class Discriminator3DWithfirstframe(nn.Module):
    # 使用 beartype 进行类型检查
    @beartype
    def __init__(
        self,
        *,
        dim,  # 特征维度
        image_size,  # 输入图像的大小
        frame_num,  # 帧的数量
        channels=3,  # 输入图像的通道数，默认值为 3
        max_dim=512,  # 最大维度限制
        linear_attn_dim_head=8,  # 线性注意力的每个头的维度
        linear_attn_heads=16,  # 线性注意力的头数
        ff_mult=4,  # 前馈网络的倍数因子
        antialiased_downsample=False,  # 是否使用抗锯齿下采样
    ):
        # 调用父类构造函数
        super().__init__()
        # 将输入图像大小转换为一对 (height, width)
        image_size = pair(image_size)
        # 计算输入图像的最小分辨率
        min_image_resolution = min(image_size)

        # 计算网络的层数
        num_layers = int(log2(min_image_resolution) - 2)
        # 计算时间维度的层数
        temporal_num_layers = int(log2(frame_num))
        # 保存时间维度的层数
        self.temporal_num_layers = temporal_num_layers

        # 创建层的维度列表，包含输入通道和每一层的输出通道
        layer_dims = [channels] + [(dim * 4) * (2**i) for i in range(num_layers + 1)]
        # 限制每一层的维度不超过 max_dim
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        # 形成层输入和输出的元组列表
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        # 初始化块的列表
        blocks = []

        # 初始化图像和帧的分辨率
        image_resolution = min_image_resolution
        frame_resolution = frame_num

        # 遍历每一层的输入和输出通道
        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            # 当前层的索引
            num_layer = ind + 1
            # 判断当前层是否为最后一层
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            # 如果当前层在时间层数内
            if ind < temporal_num_layers:
                # 创建 3D 判别块
                block = DiscriminatorBlock3DWithfirstframe(
                    in_chan,  # 输入通道数
                    out_chan,  # 输出通道数
                    antialiased_downsample=antialiased_downsample,  # 是否使用抗锯齿下采样
                )

                # 将块添加到块列表中
                blocks.append(block)

                # 更新帧分辨率
                frame_resolution //= 2
            else:
                # 创建 2D 判别块
                block = DiscriminatorBlock(
                    in_chan,  # 输入通道数
                    out_chan,  # 输出通道数
                    downsample=is_not_last,  # 是否进行下采样
                    antialiased_downsample=antialiased_downsample,  # 是否使用抗锯齿下采样
                )
                # 创建注意力块
                attn_block = nn.Sequential(
                    Residual(
                        LinearSpaceAttention(dim=out_chan, heads=linear_attn_heads, dim_head=linear_attn_dim_head)  # 线性空间注意力
                    ),
                    Residual(FeedForward(dim=out_chan, mult=ff_mult, images=True)),  # 前馈网络
                )

                # 将块和注意力块一起添加到块列表中
                blocks.append(nn.ModuleList([block, attn_block]))

            # 更新图像分辨率
            image_resolution //= 2

        # 将块列表转换为 nn.ModuleList
        self.blocks = nn.ModuleList(blocks)

        # 获取最后一层的维度
        dim_last = layer_dims[-1]

        # 计算下采样因子
        downsample_factor = 2**num_layers
        # 计算最后特征图的大小
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        # 计算潜在维度
        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        # 创建用于生成 logits 的序列
        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),  # 卷积层
            leaky_relu(),  # 激活函数
            Rearrange("b ... -> b (...)"),  # 重排维度
            nn.Linear(latent_dim, 1),  # 线性层
            Rearrange("b 1 -> b"),  # 重排维度
        )
    # 定义前向传播函数，接收输入 x
        def forward(self, x):
            # 遍历每个层，使用枚举获取索引 i 和层 layer
            for i, layer in enumerate(self.blocks):
                # 如果索引小于 temporal_num_layers，处理时间层
                if i < self.temporal_num_layers:
                    # 将输入 x 传递给当前层进行计算
                    x = layer(x)
                    # 如果是最后一个时间层，计算沿时间维度的平均值
                    if i == self.temporal_num_layers - 1:
                        x = x.mean(dim=2)
                        # 将 x 重新排列为指定的形状，注释掉的代码未被使用
                        # x = rearrange(x, "b c f h w -> (b f) c h w")
                else:
                    # 将当前层分为 block 和 attn_block
                    block, attn_block = layer
                    # 将输入 x 传递给 block 进行计算
                    x = block(x)
                    # 将计算结果传递给 attn_block 进行进一步处理
                    x = attn_block(x)
    
            # 将最终结果传递给 logits 层进行输出
            return self.to_logits(x)
# 定义视频自编码器损失类，继承自 nn.Module
class VideoAutoencoderLoss(nn.Module):
    # 初始化方法，设置各项损失权重及参数
    def __init__(
        self,
        disc_start,  # 对抗损失开始的迭代步数
        perceptual_weight=1,  # 感知损失的权重
        adversarial_loss_weight=0,  # 对抗损失的权重
        multiscale_adversarial_loss_weight=0,  # 多尺度对抗损失的权重
        grad_penalty_loss_weight=0,  # 梯度惩罚损失的权重
        quantizer_aux_loss_weight=0,  # 量化辅助损失的权重
        vgg_weights=VGG16_Weights.DEFAULT,  # VGG权重的默认设置
        discr_kwargs=None,  # 判别器的参数
        discr_3d_kwargs=None,  # 三维判别器的参数
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 保存对抗损失开始的迭代步数
        self.disc_start = disc_start
        # 保存各项损失的权重
        self.perceptual_weight = perceptual_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.multiscale_adversarial_loss_weight = multiscale_adversarial_loss_weight
        self.grad_penalty_loss_weight = grad_penalty_loss_weight
        self.quantizer_aux_loss_weight = quantizer_aux_loss_weight

        # 如果感知损失权重大于0，初始化感知模型
        if self.perceptual_weight > 0:
            self.perceptual_model = LPIPS().eval()  # 使用 LPIPS 作为感知损失模型
            # self.vgg = torchvision.models.vgg16(pretrained = True)  # 可选：加载预训练的 VGG 模型
            # self.vgg.requires_grad_(False)  # 可选：冻结 VGG 模型参数
        # if self.adversarial_loss_weight > 0:  # 可选：初始化对抗损失判别器
        #     self.discr = Discriminator(**discr_kwargs)
        # else:
        #     self.discr = None
        # if self.multiscale_adversarial_loss_weight > 0:  # 可选：初始化多尺度判别器
        #     self.multiscale_discrs = nn.ModuleList([*multiscale_discrs])
        # else:
        #     self.multiscale_discrs = None
        # 根据 discr_kwargs 初始化判别器，如果没有则为 None
        if discr_kwargs is not None:
            self.discr = Discriminator(**discr_kwargs)  # 初始化判别器
        else:
            self.discr = None  # 没有判别器
        # 根据 discr_3d_kwargs 初始化三维判别器，如果没有则为 None
        if discr_3d_kwargs is not None:
            # self.discr_3d = Discriminator3D(**discr_3d_kwargs)  # 可选：初始化三维判别器
            self.discr_3d = instantiate_from_config(discr_3d_kwargs)  # 从配置实例化三维判别器
        else:
            self.discr_3d = None  # 没有三维判别器
        # self.multiscale_discrs = nn.ModuleList([*multiscale_discrs])  # 可选：初始化多尺度判别器列表

        # 注册一个持久性为 False 的缓冲区 zero，值为 0.0
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    # 获取可训练参数的方法
    def get_trainable_params(self) -> Any:
        params = []  # 初始化参数列表
        # 如果有判别器，添加其可训练参数
        if self.discr is not None:
            params += list(self.discr.parameters())
        # 如果有三维判别器，添加其可训练参数
        if self.discr_3d is not None:
            params += list(self.discr_3d.parameters())
        # if self.multiscale_discrs is not None:  # 可选：如果有多尺度判别器，添加其参数
        #     for discr in self.multiscale_discrs:
        #         params += list(discr.parameters())
        return params  # 返回可训练参数列表

    # 获取可训练参数的另一个方法
    def get_trainable_parameters(self) -> Any:
        return self.get_trainable_params()  # 调用获取可训练参数的方法

    # 前向传播方法
    def forward(
        self,
        inputs,  # 输入数据
        reconstructions,  # 重建数据
        optimizer_idx,  # 优化器索引
        global_step,  # 全局步骤计数
        aux_losses=None,  # 辅助损失，默认为 None
        last_layer=None,  # 最后一层，默认为 None
        split="train",  # 数据集划分，默认为训练集
```