# CogVideo & CogVideoX 微调代码源码解析（六）



# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\losses\discriminator_loss.py`

```py
# 导入所需的类型定义
from typing import Dict, Iterator, List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 torchvision 库
import torchvision
# 从 einops 库导入 rearrange 函数
from einops import rearrange
# 从 matplotlib 库导入 colormaps
from matplotlib import colormaps
# 从 matplotlib 库导入 pyplot
from matplotlib import pyplot as plt

# 导入自定义的工具函数
from ....util import default, instantiate_from_config
# 导入 LPIPS 损失函数
from ..lpips.loss.lpips import LPIPS
# 导入模型的权重初始化函数
from ..lpips.model.model import weights_init
# 导入两种感知损失函数
from ..lpips.vqperceptual import hinge_d_loss, vanilla_d_loss


# 定义一个带有鉴别器的通用 LPIPS 类
class GeneralLPIPSWithDiscriminator(nn.Module):
    # 初始化方法，接受多个参数进行配置
    def __init__(
        self,
        disc_start: int,
        logvar_init: float = 0.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_loss: str = "hinge",
        scale_input_to_tgt_size: bool = False,
        dims: int = 2,
        learn_logvar: bool = False,
        regularization_weights: Union[None, Dict[str, float]] = None,
        additional_log_keys: Optional[List[str]] = None,
        discriminator_config: Optional[Dict] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存维度信息
        self.dims = dims
        # 如果维度大于2，打印相关信息
        if self.dims > 2:
            print(
                f"running with dims={dims}. This means that for perceptual loss "
                f"calculation, the LPIPS loss will be applied to each frame "
                f"independently."
            )
        # 保存是否缩放输入至目标大小
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        # 确保鉴别器损失是有效的
        assert disc_loss in ["hinge", "vanilla"]
        # 初始化感知损失为 LPIPS 模型并设置为评估模式
        self.perceptual_loss = LPIPS().eval()
        # 保存感知损失的权重
        self.perceptual_weight = perceptual_weight
        # 输出对数方差，设置为可训练参数
        self.logvar = nn.Parameter(torch.full((), logvar_init), requires_grad=learn_logvar)
        # 保存是否学习对数方差
        self.learn_logvar = learn_logvar

        # 使用默认配置创建鉴别器配置
        discriminator_config = default(
            discriminator_config,
            {
                "target": "sgm.modules.autoencoding.lpips.model.model.NLayerDiscriminator",
                "params": {
                    "input_nc": disc_in_channels,
                    "n_layers": disc_num_layers,
                    "use_actnorm": False,
                },
            },
        )

        # 实例化鉴别器并应用权重初始化
        self.discriminator = instantiate_from_config(discriminator_config).apply(weights_init)
        # 保存鉴别器开始训练的迭代次数
        self.discriminator_iter_start = disc_start
        # 根据损失类型选择相应的损失函数
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        # 保存鉴别器的因子和权重
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        # 设置正则化权重
        self.regularization_weights = default(regularization_weights, {})

        # 定义前向传播时需要的键
        self.forward_keys = [
            "optimizer_idx",
            "global_step",
            "last_layer",
            "split",
            "regularization_log",
        ]

        # 创建额外的日志键集
        self.additional_log_keys = set(default(additional_log_keys, []))
        # 更新日志键集合，包含正则化权重的键
        self.additional_log_keys.update(set(self.regularization_weights.keys()))

    # 获取可训练参数的迭代器
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.discriminator.parameters()
    # 获取可训练的自编码器参数的生成器
    def get_trainable_autoencoder_parameters(self) -> Iterator[nn.Parameter]:
        # 如果需要学习对数方差，则生成 logvar 参数
        if self.learn_logvar:
            yield self.logvar
        # 生成器为空，表示没有其他可训练参数
        yield from ()

    # 计算自适应权重，使用 torch.no_grad() 防止计算梯度
    @torch.no_grad()
    def calculate_adaptive_weight(
        self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: torch.Tensor
    ) -> torch.Tensor:
        # 计算负对数似然损失相对于最后一层输出的梯度
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        # 计算生成损失相对于最后一层输出的梯度
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        # 计算自适应权重，使用负对数似然梯度的范数与生成损失梯度的范数的比值
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        # 将权重限制在 0.0 到 1e4 之间，并分离梯度
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # 将权重乘以判别器权重
        d_weight = d_weight * self.discriminator_weight
        # 返回计算得到的自适应权重
        return d_weight

    # 定义前向传播方法，处理输入和重构数据
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        *,  # 用于确保参数顺序正确，强制使用关键字参数
        regularization_log: Dict[str, torch.Tensor],
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.Tensor,
        split: str = "train",
        weights: Union[None, float, torch.Tensor] = None,
    ):
        # 前向传播的具体实现逻辑（未提供）

    # 计算负对数似然损失
    def get_nll_loss(
        self,
        rec_loss: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算基本的负对数似然损失
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # 初始化加权负对数似然损失为基本损失
        weighted_nll_loss = nll_loss
        # 如果提供了权重，则计算加权损失
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        # 计算加权负对数似然损失的平均值
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        # 计算负对数似然损失的平均值
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # 返回负对数似然损失和加权负对数似然损失
        return nll_loss, weighted_nll_loss
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\losses\lpips.py`

```py
# 导入 PyTorch 和相关模块
import torch
import torch.nn as nn

# 从 util 模块导入默认值和配置实例化函数
from ....util import default, instantiate_from_config
# 从 lpips 模块导入 LPIPS 损失函数
from ..lpips.loss.lpips import LPIPS


# 定义 LatentLPIPS 类，继承自 nn.Module
class LatentLPIPS(nn.Module):
    # 初始化函数，接收多个参数进行配置
    def __init__(
        self,
        decoder_config,
        perceptual_weight=1.0,
        latent_weight=1.0,
        scale_input_to_tgt_size=False,
        scale_tgt_to_input_size=False,
        perceptual_weight_on_inputs=0.0,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置输入参数以控制输入大小缩放
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        self.scale_tgt_to_input_size = scale_tgt_to_input_size
        # 初始化解码器
        self.init_decoder(decoder_config)
        # 创建 LPIPS 实例并设置为评估模式
        self.perceptual_loss = LPIPS().eval()
        # 设置感知损失权重
        self.perceptual_weight = perceptual_weight
        # 设置潜在损失权重
        self.latent_weight = latent_weight
        # 设置输入的感知损失权重
        self.perceptual_weight_on_inputs = perceptual_weight_on_inputs

    # 初始化解码器的函数
    def init_decoder(self, config):
        # 根据配置实例化解码器
        self.decoder = instantiate_from_config(config)
        # 如果解码器有编码器，则将其删除
        if hasattr(self.decoder, "encoder"):
            del self.decoder.encoder

    # 前向传播函数，接收潜在输入和预测，以及图像输入
    def forward(self, latent_inputs, latent_predictions, image_inputs, split="train"):
        # 初始化日志字典
        log = dict()
        # 计算潜在输入和预测之间的均方损失
        loss = (latent_inputs - latent_predictions) ** 2
        # 记录潜在损失到日志
        log[f"{split}/latent_l2_loss"] = loss.mean().detach()
        # 初始化图像重建变量
        image_reconstructions = None
        # 如果感知权重大于 0，进行感知损失计算
        if self.perceptual_weight > 0.0:
            # 解码潜在预测得到图像重建
            image_reconstructions = self.decoder.decode(latent_predictions)
            # 解码潜在输入得到目标图像
            image_targets = self.decoder.decode(latent_inputs)
            # 计算感知损失
            perceptual_loss = self.perceptual_loss(image_targets.contiguous(), image_reconstructions.contiguous())
            # 结合潜在损失和感知损失
            loss = self.latent_weight * loss.mean() + self.perceptual_weight * perceptual_loss.mean()
            # 记录感知损失到日志
            log[f"{split}/perceptual_loss"] = perceptual_loss.mean().detach()

        # 如果输入的感知损失权重大于 0
        if self.perceptual_weight_on_inputs > 0.0:
            # 如果没有重建图像，重新解码潜在预测
            image_reconstructions = default(image_reconstructions, self.decoder.decode(latent_predictions))
            # 根据配置缩放输入图像到目标大小
            if self.scale_input_to_tgt_size:
                image_inputs = torch.nn.functional.interpolate(
                    image_inputs,
                    image_reconstructions.shape[2:],
                    mode="bicubic",
                    antialias=True,
                )
            # 根据配置缩放重建图像到输入大小
            elif self.scale_tgt_to_input_size:
                image_reconstructions = torch.nn.functional.interpolate(
                    image_reconstructions,
                    image_inputs.shape[2:],
                    mode="bicubic",
                    antialias=True,
                )

            # 计算第二次感知损失
            perceptual_loss2 = self.perceptual_loss(image_inputs.contiguous(), image_reconstructions.contiguous())
            # 将第二次感知损失加入总损失
            loss = loss + self.perceptual_weight_on_inputs * perceptual_loss2.mean()
            # 记录第二次感知损失到日志
            log[f"{split}/perceptual_loss_on_inputs"] = perceptual_loss2.mean().detach()
        # 返回最终损失和日志
        return loss, log
```

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

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\losses\__init__.py`

```py
# 定义模块的公开接口，包含可导出的类名
__all__ = [
    "GeneralLPIPSWithDiscriminator",  # 引入通用的 LPIPS 计算类，带有判别器
    "LatentLPIPS",                     # 引入潜在空间的 LPIPS 计算类
]

# 从 discriminator_loss 模块中导入 GeneralLPIPSWithDiscriminator 类
from .discriminator_loss import GeneralLPIPSWithDiscriminator
# 从 lpips 模块中导入 LatentLPIPS 类
from .lpips import LatentLPIPS
# 从 video_loss 模块中导入 VideoAutoencoderLoss 类
from .video_loss import VideoAutoencoderLoss
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\loss\lpips.py`

```py
"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""  # 引用的模块链接说明

from collections import namedtuple  # 从 collections 模块导入 namedtuple，用于创建可命名的元组

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torchvision import models  # 从 torchvision 导入模型模块

from ..util import get_ckpt_path  # 从父目录的 util 模块导入获取检查点路径的函数


class LPIPS(nn.Module):  # 定义 LPIPS 类，继承自 nn.Module
    # Learned perceptual metric  # 学习到的感知度量
    def __init__(self, use_dropout=True):  # 初始化方法，接受使用 dropout 的参数
        super().__init__()  # 调用父类的初始化方法
        self.scaling_layer = ScalingLayer()  # 实例化 ScalingLayer
        self.chns = [64, 128, 256, 512, 512]  # 定义 VGG16 的特征通道数量
        self.net = vgg16(pretrained=True, requires_grad=False)  # 加载预训练的 VGG16 网络，不更新参数
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)  # 创建第一个线性层
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)  # 创建第二个线性层
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)  # 创建第三个线性层
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)  # 创建第四个线性层
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)  # 创建第五个线性层
        self.load_from_pretrained()  # 加载预训练的权重
        for param in self.parameters():  # 遍历模型参数
            param.requires_grad = False  # 不更新任何参数的梯度


    def load_from_pretrained(self, name="vgg_lpips"):  # 从预训练模型加载参数的方法
        ckpt = get_ckpt_path(name, "sgm/modules/autoencoding/lpips/loss")  # 获取检查点路径
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)  # 加载权重
        print("loaded pretrained LPIPS loss from {}".format(ckpt))  # 打印加载信息


    @classmethod  # 将该方法定义为类方法
    def from_pretrained(cls, name="vgg_lpips"):  # 通过预训练权重创建模型的类方法
        if name != "vgg_lpips":  # 检查模型名称
            raise NotImplementedError  # 抛出未实现错误
        model = cls()  # 实例化模型
        ckpt = get_ckpt_path(name)  # 获取检查点路径
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)  # 加载权重
        return model  # 返回模型


    def forward(self, input, target):  # 前向传播方法，接受输入和目标
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))  # 对输入和目标应用缩放层
        outs0, outs1 = self.net(in0_input), self.net(in1_input)  # 通过网络计算输出
        feats0, feats1, diffs = {}, {}, {}  # 初始化特征和差异字典
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]  # 收集线性层
        for kk in range(len(self.chns)):  # 遍历通道数量
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])  # 规范化输出特征
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2  # 计算特征差的平方

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]  # 计算每个通道的空间平均
        val = res[0]  # 初始化结果为第一个通道的结果
        for l in range(1, len(self.chns)):  # 遍历其余通道
            val += res[l]  # 累加结果
        return val  # 返回最终结果


class ScalingLayer(nn.Module):  # 定义缩放层类
    def __init__(self):  # 初始化方法
        super(ScalingLayer, self).__init__()  # 调用父类初始化方法
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])  # 注册偏移量缓冲区
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])  # 注册缩放量缓冲区

    def forward(self, inp):  # 前向传播方法
        return (inp - self.shift) / self.scale  # 返回缩放后的输入


class NetLinLayer(nn.Module):  # 定义线性层类
    """A single linear layer which does a 1x1 conv"""  # 单个线性层，执行 1x1 卷积
    # 初始化 NetLinLayer 类的构造函数
        def __init__(self, chn_in, chn_out=1, use_dropout=False):
            # 调用父类构造函数初始化
            super(NetLinLayer, self).__init__()
            # 根据是否使用 dropout 创建层列表
            layers = (
                [
                    nn.Dropout(),  # 添加 dropout 层以防止过拟合
                ]
                if (use_dropout)  # 检查是否需要使用 dropout
                else []  # 如果不需要，返回空列表
            )
            # 在层列表中添加卷积层
            layers += [
                nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),  # 添加卷积层，卷积核大小为1x1
            ]
            # 将层列表封装为顺序模型
            self.model = nn.Sequential(*layers)  # 使用 nn.Sequential 组合所有层
# 定义一个名为 vgg16 的类，继承自 PyTorch 的 nn.Module
class vgg16(torch.nn.Module):
    # 初始化方法，接受是否需要梯度和是否使用预训练模型的参数
    def __init__(self, requires_grad=False, pretrained=True):
        # 调用父类的初始化方法
        super(vgg16, self).__init__()
        # 获取 VGG16 的预训练特征部分
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        # 创建五个序列容器，用于存放不同层的特征提取
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        # 定义切片的数量
        self.N_slices = 5
        # 将前4层特征添加到 slice1
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # 将第4到第8层特征添加到 slice2
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # 将第9到第15层特征添加到 slice3
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # 将第16到第22层特征添加到 slice4
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # 将第23到第29层特征添加到 slice5
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # 如果不需要计算梯度，则将所有参数的 requires_grad 属性设置为 False
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # 定义前向传播方法
    def forward(self, X):
        # 通过 slice1 处理输入 X
        h = self.slice1(X)
        # 保存 slice1 的输出
        h_relu1_2 = h
        # 通过 slice2 处理 h
        h = self.slice2(h)
        # 保存 slice2 的输出
        h_relu2_2 = h
        # 通过 slice3 处理 h
        h = self.slice3(h)
        # 保存 slice3 的输出
        h_relu3_3 = h
        # 通过 slice4 处理 h
        h = self.slice4(h)
        # 保存 slice4 的输出
        h_relu4_3 = h
        # 通过 slice5 处理 h
        h = self.slice5(h)
        # 保存 slice5 的输出
        h_relu5_3 = h
        # 创建一个命名元组，包含各个 ReLU 层的输出
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        # 将各层的输出组合成一个命名元组
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        # 返回结果
        return out


# 定义一个归一化张量的函数，接受一个张量和一个小的 epsilon 值
def normalize_tensor(x, eps=1e-10):
    # 计算张量 x 在第1维的 L2 范数
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    # 返回归一化后的张量，避免除以零
    return x / (norm_factor + eps)


# 定义一个空间平均的函数，接受一个张量和一个布尔值 keepdim
def spatial_average(x, keepdim=True):
    # 在空间维度（高度和宽度）上计算均值，保留维度选项
    return x.mean([2, 3], keepdim=keepdim)
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\loss\__init__.py`

```py
请提供需要注释的代码。
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\model\model.py`

```py
# 导入 functools 模块以使用函数工具
import functools

# 导入 PyTorch 的神经网络模块
import torch.nn as nn

# 从上级目录导入 ActNorm 实用程序
from ..util import ActNorm


# 权重初始化函数
def weights_init(m):
    # 获取模块的类名
    classname = m.__class__.__name__
    # 如果类名中包含 "Conv"，进行卷积层的权重初始化
    if classname.find("Conv") != -1:
        try:
            # 初始化权重为均值 0，标准差 0.02 的正态分布
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            # 如果出错，尝试初始化卷积层的权重
            nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
    # 如果类名中包含 "BatchNorm"，进行批量归一化层的初始化
    elif classname.find("BatchNorm") != -1:
        # 初始化权重为均值 1，标准差 0.02 的正态分布
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # 将偏置初始化为 0
        nn.init.constant_(m.bias.data, 0)


# 定义一个 NLayerDiscriminator 类，继承自 nn.Module
class NLayerDiscriminator(nn.Module):
    """定义一个 PatchGAN 判别器，参照 Pix2Pix
    --> 参考链接：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    # 初始化函数
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """构造一个 PatchGAN 判别器
        参数：
            input_nc (int)  -- 输入图像的通道数
            ndf (int)       -- 最后一层卷积的过滤器数量
            n_layers (int)  -- 判别器中的卷积层数量
            norm_layer      -- 归一化层
        """
        # 调用父类初始化方法
        super(NLayerDiscriminator, self).__init__()
        # 根据是否使用 ActNorm 选择归一化层
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        # 检查归一化层是否需要偏置
        if type(norm_layer) == functools.partial:  # 如果使用偏函数，BatchNorm2d 具有仿射参数
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # 定义卷积核大小和填充
        kw = 4
        padw = 1
        # 初始化网络序列
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),  # 输入层卷积
            nn.LeakyReLU(0.2, True),  # 激活函数
        ]
        # 初始化滤波器数量的倍数
        nf_mult = 1
        nf_mult_prev = 1
        # 循环添加卷积层和归一化层
        for n in range(1, n_layers):  # 逐渐增加过滤器的数量
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)  # 最大值限制为 8
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,  # 输入通道数
                    ndf * nf_mult,  # 输出通道数
                    kernel_size=kw,  # 卷积核大小
                    stride=2,  # 步幅
                    padding=padw,  # 填充
                    bias=use_bias,  # 是否使用偏置
                ),
                norm_layer(ndf * nf_mult),  # 添加归一化层
                nn.LeakyReLU(0.2, True),  # 激活函数
            ]

        # 最后一层卷积的参数设置
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)  # 最大值限制为 8
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,  # 输入通道数
                ndf * nf_mult,  # 输出通道数
                kernel_size=kw,  # 卷积核大小
                stride=1,  # 步幅
                padding=padw,  # 填充
                bias=use_bias,  # 是否使用偏置
            ),
            norm_layer(ndf * nf_mult),  # 添加归一化层
            nn.LeakyReLU(0.2, True),  # 激活函数
        ]

        # 添加输出层
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)  # 输出 1 通道预测图
        ]  
        # 将序列转换为顺序容器
        self.main = nn.Sequential(*sequence)

    # 定义前向传播方法
    def forward(self, input):
        """标准前向传播。"""
        return self.main(input)  # 返回主网络的输出
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\model\__init__.py`

```py
# 这里是一个空的代码块，没有提供任何代码。
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\util.py`

```py
# 导入所需的库
import hashlib  # 用于计算 MD5 哈希
import os  # 用于操作文件和目录

import requests  # 用于发送 HTTP 请求
import torch  # 用于深度学习框架
import torch.nn as nn  # 用于构建神经网络模块
from tqdm import tqdm  # 用于显示进度条

# 定义 URL 映射字典，包含模型名称及其下载链接
URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

# 定义检查点映射字典，包含模型名称及其本地文件名
CKPT_MAP = {"vgg_lpips": "vgg.pth"}

# 定义 MD5 哈希映射字典，包含模型名称及其对应的哈希值
MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}

# 下载指定 URL 的文件到本地路径
def download(url, local_path, chunk_size=1024):
    # 创建存储文件的目录（如果不存在的话）
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    # 发送 GET 请求以流式方式下载文件
    with requests.get(url, stream=True) as r:
        # 获取响应头中的内容长度
        total_size = int(r.headers.get("content-length", 0))
        # 使用 tqdm 显示下载进度
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            # 以二进制写模式打开本地文件
            with open(local_path, "wb") as f:
                # 逐块读取内容并写入文件
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:  # 如果读取到数据
                        f.write(data)  # 将数据写入文件
                        pbar.update(chunk_size)  # 更新进度条

# 计算指定文件的 MD5 哈希值
def md5_hash(path):
    # 以二进制读取模式打开文件
    with open(path, "rb") as f:
        content = f.read()  # 读取文件内容
    # 返回内容的 MD5 哈希值
    return hashlib.md5(content).hexdigest()

# 获取检查点路径，如果需要则下载模型
def get_ckpt_path(name, root, check=False):
    # 确保模型名称在 URL 映射中
    assert name in URL_MAP
    # 组合根目录和检查点文件名生成完整路径
    path = os.path.join(root, CKPT_MAP[name])
    # 如果文件不存在或需要检查 MD5 值
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        # 打印下载信息
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        # 下载文件
        download(URL_MAP[name], path)
        # 计算下载后的文件 MD5 哈希
        md5 = md5_hash(path)
        # 确保 MD5 哈希匹配
        assert md5 == MD5_MAP[name], md5
    # 返回检查点文件的路径
    return path

# 定义一个标准化的神经网络模块
class ActNorm(nn.Module):
    # 初始化模块，设置参数和属性
    def __init__(self, num_features, logdet=False, affine=True, allow_reverse_init=False):
        assert affine  # 确保启用仿射变换
        super().__init__()  # 调用父类初始化
        self.logdet = logdet  # 是否计算对数行列式
        # 定义位置参数
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        # 定义缩放参数
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init  # 是否允许反向初始化

        # 注册一个用于标记初始化状态的缓冲区
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    # 初始化函数，接受输入并计算位置和缩放参数
    def initialize(self, input):
        with torch.no_grad():  # 禁用梯度计算
            # 重排输入并扁平化
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            # 计算每个特征的均值
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            # 计算每个特征的标准差
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)

            # 更新位置参数为负均值
            self.loc.data.copy_(-mean)
            # 更新缩放参数为标准差的倒数
            self.scale.data.copy_(1 / (std + 1e-6))
    # 定义前向传播函数，接受输入和是否反向的参数
    def forward(self, input, reverse=False):
        # 如果需要反向传播，调用反向函数处理输入
        if reverse:
            return self.reverse(input)
        # 如果输入是二维数组，扩展维度以适应后续处理
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True  # 标记为需要挤压的状态
        else:
            squeeze = False  # 标记为不需要挤压的状态
    
        # 获取输入的高度和宽度
        _, _, height, width = input.shape
    
        # 如果处于训练模式且未初始化，进行初始化
        if self.training and self.initialized.item() == 0:
            self.initialize(input)  # 初始化
            self.initialized.fill_(1)  # 标记为已初始化
    
        # 计算h，考虑缩放和偏移
        h = self.scale * (input + self.loc)
    
        # 如果需要挤压，去掉多余的维度
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
    
        # 如果需要计算对数行列式，执行相应的计算
        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))  # 计算缩放的对数绝对值
            logdet = height * width * torch.sum(log_abs)  # 计算对数行列式
            logdet = logdet * torch.ones(input.shape[0]).to(input)  # 创建与输入批次相同大小的张量
            return h, logdet  # 返回h和对数行列式
    
        return h  # 返回计算结果h
    
    # 定义反向传播函数，接受输出作为输入
    def reverse(self, output):
        # 如果处于训练模式且未初始化，进行初始化
        if self.training and self.initialized.item() == 0:
            # 如果不允许反向初始化，抛出错误
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)  # 初始化
                self.initialized.fill_(1)  # 标记为已初始化
    
        # 如果输出是二维数组，扩展维度以适应后续处理
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True  # 标记为需要挤压的状态
        else:
            squeeze = False  # 标记为不需要挤压的状态
    
        # 根据缩放和偏移计算h
        h = output / self.scale - self.loc
    
        # 如果需要挤压，去掉多余的维度
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h  # 返回计算结果h
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\vqperceptual.py`

```py
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能模块
import torch.nn.functional as F


# 定义一个函数，计算对抗网络中判别器的铰链损失
def hinge_d_loss(logits_real, logits_fake):
    # 计算真实样本的损失，使用 ReLU 激活函数
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    # 计算假样本的损失，使用 ReLU 激活函数
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    # 计算总的判别器损失，取真实和假样本损失的平均
    d_loss = 0.5 * (loss_real + loss_fake)
    # 返回判别器的铰链损失
    return d_loss


# 定义一个函数，计算判别器的经典损失
def vanilla_d_loss(logits_real, logits_fake):
    # 计算判别器损失，使用 Softplus 函数处理真实和假样本
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    # 返回判别器的经典损失
    return d_loss
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\__init__.py`

```py
请提供需要注释的代码段。
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\magvit2_pytorch.py`

```py
# 导入所需的模块和库
import copy  # 导入复制对象的库
from pathlib import Path  # 导入路径操作库
from math import log2, ceil, sqrt  # 导入数学相关函数
from functools import wraps, partial  # 导入函数式编程工具

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
from torch.cuda.amp import autocast  # 导入自动混合精度训练
from torch import nn, einsum, Tensor  # 导入神经网络模块、爱因斯坦求和和张量类
from torch.nn import Module, ModuleList  # 导入 PyTorch 模块和模块列表
from torch.autograd import grad as torch_grad  # 导入自动梯度计算

import torchvision  # 导入计算机视觉库
from torchvision.models import VGG16_Weights  # 导入 VGG16 模型权重

from collections import namedtuple  # 导入命名元组

# from vector_quantize_pytorch import LFQ, FSQ  # 注释掉的导入，表示不再使用的模块
from .regularizers.finite_scalar_quantization import FSQ  # 从正则化模块导入有限标量量化
from .regularizers.lookup_free_quantization import LFQ  # 从正则化模块导入查找自由量化

from einops import rearrange, repeat, reduce, pack, unpack  # 导入 einops 的操作函数
from einops.layers.torch import Rearrange  # 导入 PyTorch 特有的 einops 重排列层

from beartype import beartype  # 导入类型检查库
from beartype.typing import Union, Tuple, Optional, List  # 导入类型注解

from magvit2_pytorch.attend import Attend  # 从 magvit2 导入注意力模块
from magvit2_pytorch.version import __version__  # 导入当前版本信息

from gateloop_transformer import SimpleGateLoopLayer  # 从 gateloop_transformer 导入简单门控循环层

from taylor_series_linear_attention import TaylorSeriesLinearAttn  # 从泰勒级数线性注意力导入相应模块

from kornia.filters import filter3d  # 从 Kornia 导入三维滤波器

import pickle  # 导入序列化和反序列化模块

# 辅助函数


def exists(v):  # 检查变量是否存在
    return v is not None  # 如果变量不为 None，返回 True


def default(v, d):  # 返回默认值
    return v if exists(v) else d  # 如果 v 存在则返回 v，否则返回 d


def safe_get_index(it, ind, default=None):  # 安全获取索引
    if ind < len(it):  # 检查索引是否在范围内
        return it[ind]  # 返回指定索引的元素
    return default  # 如果索引超出范围，则返回默认值


def pair(t):  # 将输入转换为元组
    return t if isinstance(t, tuple) else (t, t)  # 如果 t 是元组则返回，否则返回重复的元组


def identity(t, *args, **kwargs):  # 身份函数
    return t  # 返回原输入


def divisible_by(num, den):  # 检查 num 是否能被 den 整除
    return (num % den) == 0  # 返回除法余数是否为零


def pack_one(t, pattern):  # 打包一个张量
    return pack([t], pattern)  # 将张量打包成指定模式


def unpack_one(t, ps, pattern):  # 解包一个张量
    return unpack(t, ps, pattern)[0]  # 将张量解包并返回第一个元素


def append_dims(t, ndims: int):  # 向张量追加维度
    return t.reshape(*t.shape, *((1,) * ndims))  # 调整张量形状，追加指定数量的维度


def is_odd(n):  # 检查数字是否为奇数
    return not divisible_by(n, 2)  # 使用可整除函数检查


def maybe_del_attr_(o, attr):  # 有条件地删除对象属性
    if hasattr(o, attr):  # 检查对象是否具有该属性
        delattr(o, attr)  # 删除该属性


def cast_tuple(t, length=1):  # 将输入转换为元组
    return t if isinstance(t, tuple) else ((t,) * length)  # 如果 t 是元组则返回，否则创建指定长度的元组


# 张量辅助函数


def l2norm(t):  # 计算 L2 范数
    return F.normalize(t, dim=-1, p=2)  # 在最后一个维度上归一化


def pad_at_dim(t, pad, dim=-1, value=0.0):  # 在指定维度上填充张量
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)  # 计算从右边开始的维度索引
    zeros = (0, 0) * dims_from_right  # 创建零填充的元组
    return F.pad(t, (*zeros, *pad), value=value)  # 填充张量并返回


def pick_video_frame(video, frame_indices):  # 从视频中选择帧
    batch, device = video.shape[0], video.device  # 获取批次大小和设备
    video = rearrange(video, "b c f ... -> b f c ...")  # 调整视频张量的形状
    batch_indices = torch.arange(batch, device=device)  # 创建批次索引
    batch_indices = rearrange(batch_indices, "b -> b 1")  # 调整批次索引的形状
    images = video[batch_indices, frame_indices]  # 根据索引选择图像
    images = rearrange(images, "b 1 c ... -> b c ...")  # 调整图像张量的形状
    return images  # 返回选择的图像


# GAN 相关函数


def gradient_penalty(images, output):  # 计算梯度惩罚
    batch_size = images.shape[0]  # 获取批次大小

    gradients = torch_grad(  # 计算输出关于输入的梯度
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),  # 设置梯度输出
        create_graph=True,  # 创建计算图以便后续计算
        retain_graph=True,  # 保留计算图以便多次使用
        only_inputs=True,  # 仅对输入计算梯度
    )[0]  # 只取第一个返回值

    gradients = rearrange(gradients, "b ... -> b (...)")  # 调整梯度张量的形状
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # 计算并返回平均梯度惩罚


def leaky_relu(p=0.1):  # 创建带泄漏的 ReLU 激活函数
    return nn.LeakyReLU(p)  # 返回带泄漏的 ReLU 实例
# 定义一个函数来计算铰链判别损失
def hinge_discr_loss(fake, real):
    # 计算并返回铰链损失的均值
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


# 定义一个函数来计算铰链生成损失
def hinge_gen_loss(fake):
    # 返回生成样本的负均值作为损失
    return -fake.mean()


# 自动混合精度上下文装饰器，禁用混合精度
@autocast(enabled=False)
# 类型检查装饰器
@beartype
# 定义一个函数计算损失相对于某一层的梯度
def grad_layer_wrt_loss(loss: Tensor, layer: nn.Parameter):
    # 使用反向传播计算损失相对于层参数的梯度，并返回其张量
    return torch_grad(outputs=loss, inputs=layer, grad_outputs=torch.ones_like(loss), retain_graph=True)[0].detach()


# 装饰器帮助函数


# 定义一个装饰器，用于移除 VGG 属性
def remove_vgg(fn):
    # 装饰器内部函数
    @wraps(fn)
    def inner(self, *args, **kwargs):
        # 检查对象是否有 VGG 属性
        has_vgg = hasattr(self, "vgg")
        if has_vgg:
            # 保存 VGG 属性并删除它
            vgg = self.vgg
            delattr(self, "vgg")

        # 调用原始函数并获取输出
        out = fn(self, *args, **kwargs)

        # 如果有 VGG 属性，则将其恢复
        if has_vgg:
            self.vgg = vgg

        # 返回函数输出
        return out

    return inner


# 帮助类


# 定义一个顺序模块的构造函数
def Sequential(*modules):
    # 过滤出有效的模块
    modules = [*filter(exists, modules)]

    # 如果没有有效模块，则返回身份映射
    if len(modules) == 0:
        return nn.Identity()

    # 返回一个顺序容器
    return nn.Sequential(*modules)


# 定义残差模块类
class Residual(Module):
    # 类型检查装饰器
    @beartype
    def __init__(self, fn: Module):
        # 调用父类构造函数
        super().__init__()
        # 保存传入的模块函数
        self.fn = fn

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 返回残差输出，即函数输出加上输入
        return self.fn(x, **kwargs) + x


# 定义一个用于张量操作的类，将张量转换为时间序列格式
class ToTimeSequence(Module):
    # 类型检查装饰器
    @beartype
    def __init__(self, fn: Module):
        # 调用父类构造函数
        super().__init__()
        # 保存传入的模块函数
        self.fn = fn

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 重新排列输入张量的维度
        x = rearrange(x, "b c f ... -> b ... f c")
        # 将张量打包以便处理
        x, ps = pack_one(x, "* n c")

        # 通过模块函数处理张量
        o = self.fn(x, **kwargs)

        # 解包处理后的张量
        o = unpack_one(o, ps, "* n c")
        # 重新排列输出张量的维度
        return rearrange(o, "b ... f c -> b c f ...")


# 定义一个 squeeze-excite 模块类
class SqueezeExcite(Module):
    # 全局上下文网络 - 类似注意力机制的 squeeze-excite 变体
    def __init__(self, dim, *, dim_out=None, dim_hidden_min=16, init_bias=-10):
        # 调用父类构造函数
        super().__init__()
        # 如果未指定输出维度，则默认为输入维度
        dim_out = default(dim_out, dim)

        # 创建卷积层用于计算键
        self.to_k = nn.Conv2d(dim, 1, 1)
        # 计算隐藏层维度
        dim_hidden = max(dim_hidden_min, dim_out // 2)

        # 定义网络结构
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_hidden, 1), nn.LeakyReLU(0.1), nn.Conv2d(dim_hidden, dim_out, 1), nn.Sigmoid()
        )

        # 初始化网络的权重和偏置
        nn.init.zeros_(self.net[-2].weight)
        nn.init.constant_(self.net[-2].bias, init_bias)

    # 前向传播函数
    def forward(self, x):
        # 保存原始输入和批量大小
        orig_input, batch = x, x.shape[0]
        # 检查输入是否为视频格式
        is_video = x.ndim == 5

        # 如果是视频格式，重新排列输入张量的维度
        if is_video:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        # 计算上下文信息
        context = self.to_k(x)

        # 重新排列上下文信息，并应用 softmax
        context = rearrange(context, "b c h w -> b c (h w)").softmax(dim=-1)
        # 展平输入张量
        spatial_flattened_input = rearrange(x, "b c h w -> b c (h w)")

        # 计算输出
        out = einsum("b i n, b c n -> b c i", context, spatial_flattened_input)
        # 重新排列输出
        out = rearrange(out, "... -> ... 1")
        # 通过网络计算门控值
        gates = self.net(out)

        # 如果是视频格式，重新排列门控值的维度
        if is_video:
            gates = rearrange(gates, "(b f) c h w -> b c f h w", b=batch)

        # 返回门控值与原始输入的乘积
        return gates * orig_input


# 定义一个 token shifting 模块类
class TokenShift(Module):
    # 类型检查装饰器
    @beartype
    def __init__(self, fn: Module):
        # 调用父类构造函数
        super().__init__()
        # 保存传入的模块函数
        self.fn = fn
    # 定义前向传播函数，接受输入 x 和其他可选参数
        def forward(self, x, **kwargs):
            # 将输入 x 在第 1 维上分成两部分，分别赋值给 x 和 x_shift
            x, x_shift = x.chunk(2, dim=1)
            # 在时间维度上对 x_shift 进行填充，使其适应后续操作
            x_shift = pad_at_dim(x_shift, (1, -1), dim=2)  # shift time dimension
            # 将 x 和填充后的 x_shift 在第 1 维上拼接
            x = torch.cat((x, x_shift), dim=1)
            # 调用 self.fn 函数处理拼接后的 x，并传入其他参数
            return self.fn(x, **kwargs)
# rmsnorm

# 定义 RMSNorm 类，继承自 Module 类
class RMSNorm(Module):
    # 初始化函数，接受多个参数
    def __init__(self, dim, channel_first=False, images=False, bias=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据是否为图像确定可广播维度
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        # 根据通道顺序和维度定义形状
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        # 保存通道顺序
        self.channel_first = channel_first
        # 计算缩放因子
        self.scale = dim**0.5
        # 定义可学习的参数 gamma，初始化为全1
        self.gamma = nn.Parameter(torch.ones(shape))
        # 定义偏置参数，若 bias 为真则初始化为全0，否则设为0.0
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    # 前向传播函数
    def forward(self, x):
        # 归一化输入 x，应用缩放因子和 gamma，然后加上偏置
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


# 定义自适应 RMSNorm 类，继承自 Module 类
class AdaptiveRMSNorm(Module):
    # 初始化函数，接受多个参数
    def __init__(self, dim, *, dim_cond, channel_first=False, images=False, bias=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据是否为图像确定可广播维度
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        # 根据通道顺序和维度定义形状
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        # 保存条件维度和通道顺序
        self.dim_cond = dim_cond
        self.channel_first = channel_first
        # 计算缩放因子
        self.scale = dim**0.5

        # 定义线性层用于计算 gamma
        self.to_gamma = nn.Linear(dim_cond, dim)
        # 若需要偏置，则定义相应的线性层
        self.to_bias = nn.Linear(dim_cond, dim) if bias else None

        # 初始化 gamma 的权重为零，偏置为一
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        # 若需要偏置，则初始化偏置层的权重和偏置为零
        if bias:
            nn.init.zeros_(self.to_bias.weight)
            nn.init.zeros_(self.to_bias.bias)

    # 前向传播函数，带有条件输入
    @beartype
    def forward(self, x: Tensor, *, cond: Tensor):
        # 获取输入的批大小
        batch = x.shape[0]
        # 确保条件张量的形状与批大小匹配
        assert cond.shape == (batch, self.dim_cond)

        # 计算 gamma 值
        gamma = self.to_gamma(cond)

        # 初始化偏置为 0
        bias = 0.0
        # 若存在偏置层，则计算偏置
        if exists(self.to_bias):
            bias = self.to_bias(cond)

        # 若通道顺序为先，则扩展 gamma 的维度
        if self.channel_first:
            gamma = append_dims(gamma, x.ndim - 2)

            # 若存在偏置层，则扩展偏置的维度
            if exists(self.to_bias):
                bias = append_dims(bias, x.ndim - 2)

        # 归一化输入 x，应用缩放因子和 gamma，然后加上偏置
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * gamma + bias


# attention

# 定义 Attention 类，继承自 Module 类
class Attention(Module):
    # 初始化函数，接受多个参数
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_cond: Optional[int] = None,
        causal=False,
        dim_head=32,
        heads=8,
        flash=False,
        dropout=0.0,
        num_memory_kv=4,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 计算内部维度
        dim_inner = dim_head * heads

        # 检查是否需要条件维度
        self.need_cond = exists(dim_cond)

        # 根据是否需要条件维度选择归一化方式
        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond=dim_cond)
        else:
            self.norm = RMSNorm(dim)

        # 定义线性层以计算查询、键、值
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False), 
            # 重排张量维度
            Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=heads)
        )

        # 确保记忆键值对数量大于零
        assert num_memory_kv > 0
        # 定义可学习的参数用于存储记忆键值对
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_memory_kv, dim_head))

        # 定义注意力机制
        self.attend = Attend(causal=causal, dropout=dropout, flash=flash)

        # 定义输出层
        self.to_out = nn.Sequential(
            # 重排张量维度
            Rearrange("b h n d -> b n (h d)"), 
            nn.Linear(dim_inner, dim, bias=False)
        )

    # 继续定义其他方法...
    # 前向传播函数，接受输入张量 x 和可选的掩码与条件张量
    def forward(self, x, mask: Optional[Tensor] = None, cond: Optional[Tensor] = None):
        # 根据是否需要条件，构建条件参数字典
        maybe_cond_kwargs = dict(cond=cond) if self.need_cond else dict()
    
        # 对输入 x 进行归一化处理，可能包含条件参数
        x = self.norm(x, **maybe_cond_kwargs)
    
        # 将输入 x 转换为查询（q）、键（k）和值（v）三种张量
        q, k, v = self.to_qkv(x)
    
        # 将记忆中的键（mk）和值（mv）重复以匹配批次大小，并保持原有形状
        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=q.shape[0]), self.mem_kv)
        # 将新的键张量和记忆中的键张量沿最后一个维度拼接
        k = torch.cat((mk, k), dim=-2)
        # 将新的值张量和记忆中的值张量沿最后一个维度拼接
        v = torch.cat((mv, v), dim=-2)
    
        # 根据查询、键和值以及掩码计算注意力输出
        out = self.attend(q, k, v, mask=mask)
        # 将注意力输出转换为最终输出格式
        return self.to_out(out)
# 定义一个线性注意力类，继承自 Module
class LinearAttention(Module):
    """
    使用特定的线性注意力，参考 https://arxiv.org/abs/2106.09681
    """

    @beartype
    # 初始化方法，接收多个参数
    def __init__(self, *, dim, dim_cond: Optional[int] = None, dim_head=8, heads=8, dropout=0.0):
        # 调用父类的初始化方法
        super().__init__()
        # 计算内部维度
        dim_inner = dim_head * heads

        # 检查条件维度是否存在
        self.need_cond = exists(dim_cond)

        # 如果需要条件，则使用自适应 RMSNorm
        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond=dim_cond)
        # 否则使用 RMSNorm
        else:
            self.norm = RMSNorm(dim)

        # 创建 TaylorSeriesLinearAttn 对象
        self.attn = TaylorSeriesLinearAttn(dim=dim, dim_head=dim_head, heads=heads)

    # 前向传播方法
    def forward(self, x, cond: Optional[Tensor] = None):
        # 根据是否需要条件来设置可选参数
        maybe_cond_kwargs = dict(cond=cond) if self.need_cond else dict()

        # 通过规范化处理输入数据
        x = self.norm(x, **maybe_cond_kwargs)

        # 返回注意力计算的结果
        return self.attn(x)


# 定义一个线性空间注意力类，继承自 LinearAttention
class LinearSpaceAttention(LinearAttention):
    # 前向传播方法
    def forward(self, x, *args, **kwargs):
        # 重新排列张量维度
        x = rearrange(x, "b c ... h w -> b ... h w c")
        # 将张量打包成一个新的格式
        x, batch_ps = pack_one(x, "* h w c")
        # 再次打包张量以准备进行注意力计算
        x, seq_ps = pack_one(x, "b * c")

        # 调用父类的前向传播方法
        x = super().forward(x, *args, **kwargs)

        # 解包张量以恢复原始格式
        x = unpack_one(x, seq_ps, "b * c")
        x = unpack_one(x, batch_ps, "* h w c")
        # 重新排列输出张量维度
        return rearrange(x, "b ... h w c -> b c ... h w")


# 定义空间注意力类，继承自 Attention
class SpaceAttention(Attention):
    # 前向传播方法
    def forward(self, x, *args, **kwargs):
        # 重新排列张量维度
        x = rearrange(x, "b c t h w -> b t h w c")
        # 将张量打包以便处理
        x, batch_ps = pack_one(x, "* h w c")
        # 再次打包以准备进行注意力计算
        x, seq_ps = pack_one(x, "b * c")

        # 调用父类的前向传播方法
        x = super().forward(x, *args, **kwargs)

        # 解包张量以恢复原始格式
        x = unpack_one(x, seq_ps, "b * c")
        x = unpack_one(x, batch_ps, "* h w c")
        # 重新排列输出张量维度
        return rearrange(x, "b t h w c -> b c t h w")


# 定义时间注意力类，继承自 Attention
class TimeAttention(Attention):
    # 前向传播方法
    def forward(self, x, *args, **kwargs):
        # 重新排列张量维度
        x = rearrange(x, "b c t h w -> b h w t c")
        # 将张量打包以便处理
        x, batch_ps = pack_one(x, "* t c")

        # 调用父类的前向传播方法
        x = super().forward(x, *args, **kwargs)

        # 解包张量以恢复原始格式
        x = unpack_one(x, batch_ps, "* t c")
        # 重新排列输出张量维度
        return rearrange(x, "b h w t c -> b c t h w")


# 定义 GEGLU 类，继承自 Module
class GEGLU(Module):
    # 前向传播方法
    def forward(self, x):
        # 将输入张量分成两部分：x 和 gate
        x, gate = x.chunk(2, dim=1)
        # 返回激活函数 gelu 的结果乘以 x
        return F.gelu(gate) * x


# 定义前馈神经网络类，继承自 Module
class FeedForward(Module):
    @beartype
    # 初始化方法，接收多个参数
    def __init__(self, dim, *, dim_cond: Optional[int] = None, mult=4, images=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据 images 参数选择卷积类型
        conv_klass = nn.Conv2d if images else nn.Conv3d

        # 根据条件维度选择 RMSNorm 类型
        rmsnorm_klass = RMSNorm if not exists(dim_cond) else partial(AdaptiveRMSNorm, dim_cond=dim_cond)

        # 创建适应性规范化类的部分函数
        maybe_adaptive_norm_klass = partial(rmsnorm_klass, channel_first=True, images=images)

        # 计算内部维度
        dim_inner = int(dim * mult * 2 / 3)

        # 创建规范化层
        self.norm = maybe_adaptive_norm_klass(dim)

        # 构建前馈神经网络，包括卷积层和 GEGLU 激活
        self.net = Sequential(conv_klass(dim, dim_inner * 2, 1), GEGLU(), conv_klass(dim_inner, dim, 1))

    @beartype
    # 前向传播方法
    def forward(self, x: Tensor, *, cond: Optional[Tensor] = None):
        # 根据条件是否存在设置可选参数
        maybe_cond_kwargs = dict(cond=cond) if exists(cond) else dict()

        # 通过规范化处理输入数据
        x = self.norm(x, **maybe_cond_kwargs)
        # 返回前馈网络的结果
        return self.net(x)


# 注释: 使用反锯齿下采样（blurpool Zhang 等人的方法）构建的判别器
# 定义一个模糊处理模块，继承自 Module 类
class Blur(Module):
    # 初始化方法
    def __init__(self):
        # 调用父类构造函数
        super().__init__()
        # 创建一个一维张量 f，包含模糊核的值
        f = torch.Tensor([1, 2, 1])
        # 注册一个缓冲区以存储模糊核
        self.register_buffer("f", f)

    # 前向传播方法
    def forward(self, x, space_only=False, time_only=False):
        # 确保不同时使用空间模糊和时间模糊
        assert not (space_only and time_only)

        # 获取模糊核
        f = self.f

        # 如果只进行空间模糊
        if space_only:
            # 计算外积以生成二维模糊核
            f = einsum("i, j -> i j", f, f)
            # 调整维度为 1x1xN
            f = rearrange(f, "... -> 1 1 ...")
        # 如果只进行时间模糊
        elif time_only:
            # 调整维度为 1xNx1x1
            f = rearrange(f, "f -> 1 f 1 1")
        else:
            # 如果同时进行空间和时间模糊
            # 计算三维外积以生成三维模糊核
            f = einsum("i, j, k -> i j k", f, f, f)
            # 调整维度为 1xNxN
            f = rearrange(f, "... -> 1 ...")

        # 检查输入是否为图像格式（4维）
        is_images = x.ndim == 4

        # 如果是图像格式，则调整维度以适应后续处理
        if is_images:
            x = rearrange(x, "b c h w -> b c 1 h w")

        # 使用三维滤波器处理输入
        out = filter3d(x, f, normalized=True)

        # 如果是图像格式，则恢复到原始维度
        if is_images:
            out = rearrange(out, "b c 1 h w -> b c h w")

        # 返回处理后的输出
        return out


# 定义一个判别器块，继承自 Module 类
class DiscriminatorBlock(Module):
    # 初始化方法，接受输入通道数、过滤器数量等参数
    def __init__(self, input_channels, filters, downsample=True, antialiased_downsample=True):
        # 调用父类构造函数
        super().__init__()
        # 定义残差卷积层，决定是否下采样
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        # 定义一个序列网络，包括两个卷积层和激活函数
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(),
        )

        # 根据条件选择是否使用模糊处理
        self.maybe_blur = Blur() if antialiased_downsample else None

        # 定义下采样操作，如果需要下采样
        self.downsample = (
            nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2), nn.Conv2d(filters * 4, filters, 1)
            )
            if downsample
            else None
        )

    # 前向传播方法
    def forward(self, x):
        # 计算残差
        res = self.conv_res(x)

        # 通过网络处理输入
        x = self.net(x)

        # 如果有下采样操作
        if exists(self.downsample):
            # 如果有模糊处理
            if exists(self.maybe_blur):
                # 进行空间模糊处理
                x = self.maybe_blur(x, space_only=True)

            # 执行下采样操作
            x = self.downsample(x)

        # 将处理后的输出与残差相加并进行归一化
        x = (x + res) * (2**-0.5)
        # 返回处理后的输出
        return x


# 定义一个判别器类，继承自 Module 类
class Discriminator(Module):
    # 类型注释的方法
    @beartype
    # 初始化方法，接受多个参数
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels=3,
        max_dim=512,
        attn_heads=8,
        attn_dim_head=32,
        linear_attn_dim_head=8,
        linear_attn_heads=16,
        ff_mult=4,
        antialiased_downsample=False,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 将输入的图像大小转换为元组形式
        image_size = pair(image_size)
        # 获取图像最小分辨率
        min_image_resolution = min(image_size)

        # 计算网络层数，最小分辨率减少2的对数取整
        num_layers = int(log2(min_image_resolution) - 2)

        # 初始化块列表
        blocks = []

        # 定义每层的维度，第一层为通道数，后续层根据指数增长
        layer_dims = [channels] + [(dim * 4) * (2**i) for i in range(num_layers + 1)]
        # 确保每层的维度不超过最大维度
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        # 创建输入和输出维度的元组
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        # 重新初始化块列表和注意力块列表
        blocks = []
        attn_blocks = []

        # 设置图像分辨率为最小分辨率
        image_resolution = min_image_resolution

        # 遍历每对输入和输出通道
        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            # 计算当前层数
            num_layer = ind + 1
            # 判断当前层是否为最后一层
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            # 创建判别器块，包含输入和输出通道，是否下采样的标志
            block = DiscriminatorBlock(
                in_chan, out_chan, downsample=is_not_last, antialiased_downsample=antialiased_downsample
            )

            # 创建注意力块，包含残差连接和前馈层
            attn_block = Sequential(
                Residual(LinearSpaceAttention(dim=out_chan, heads=linear_attn_heads, dim_head=linear_attn_dim_head)),
                Residual(FeedForward(dim=out_chan, mult=ff_mult, images=True)),
            )

            # 将块和注意力块添加到块列表中
            blocks.append(ModuleList([block, attn_block]))

            # 每次迭代将图像分辨率减半
            image_resolution //= 2

        # 将所有块转换为模块列表
        self.blocks = ModuleList(blocks)

        # 获取最后一层的维度
        dim_last = layer_dims[-1]

        # 计算下采样因子
        downsample_factor = 2**num_layers
        # 计算最后特征图的大小
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        # 计算潜在维度
        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        # 创建最终输出的层，包含卷积、激活、重排列和线性层
        self.to_logits = Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),
            leaky_relu(),
            Rearrange("b ... -> b (...)"),
            nn.Linear(latent_dim, 1),
            Rearrange("b 1 -> b"),
        )

    # 定义前向传播方法
    def forward(self, x):
        # 遍历每个块和注意力块进行前向传播
        for block, attn_block in self.blocks:
            x = block(x)  # 通过判别器块
            x = attn_block(x)  # 通过注意力块

        # 返回最后的 logits 结果
        return self.to_logits(x)
# 可调节卷积，来自 Karras 等人的 Stylegan2
# 用于对潜在变量进行条件化


class Conv3DMod(Module):
    @beartype
    # 初始化函数，设置卷积的维度、核大小等参数
    def __init__(
        self, dim, *, spatial_kernel, time_kernel, causal=True, dim_out=None, demod=True, eps=1e-8, pad_mode="zeros"
    ):
        super().__init__()  # 调用父类的初始化函数
        dim_out = default(dim_out, dim)  # 如果未指定 dim_out，则默认与 dim 相同

        self.eps = eps  # 设置一个小常数用于数值稳定性

        # 确保空间核和时间核都是奇数
        assert is_odd(spatial_kernel) and is_odd(time_kernel)

        self.spatial_kernel = spatial_kernel  # 保存空间核的大小
        self.time_kernel = time_kernel  # 保存时间核的大小

        # 根据是否为因果卷积计算时间填充
        time_padding = (time_kernel - 1, 0) if causal else ((time_kernel // 2,) * 2)

        self.pad_mode = pad_mode  # 设置填充模式
        # 计算总的填充大小，包含空间和时间的填充
        self.padding = (*((spatial_kernel // 2,) * 4), *time_padding)
        # 初始化卷积核权重为随机值，并作为可学习参数
        self.weights = nn.Parameter(torch.randn((dim_out, dim, time_kernel, spatial_kernel, spatial_kernel)))

        self.demod = demod  # 是否进行去调制的标志

        # 使用 Kaiming 正态分布初始化卷积权重
        nn.init.kaiming_normal_(self.weights, a=0, mode="fan_in", nonlinearity="selu")

    @beartype
    # 前向传播函数，定义数据的流动
    def forward(self, fmap, cond: Tensor):
        """
        符号说明

        b - 批量
        n - 卷积
        o - 输出
        i - 输入
        k - 核
        """

        b = fmap.shape[0]  # 获取批量大小

        # 准备用于调制的权重

        weights = self.weights  # 获取当前权重

        # 执行调制和去调制，参考 stylegan2 的实现

        cond = rearrange(cond, "b i -> b 1 i 1 1 1")  # 调整条件张量的形状以适配权重

        weights = weights * (cond + 1)  # 对权重进行调制

        if self.demod:  # 如果需要去调制
            # 计算权重的逆归一化因子
            inv_norm = reduce(weights**2, "b o i k0 k1 k2 -> b o 1 1 1 1", "sum").clamp(min=self.eps).rsqrt()
            weights = weights * inv_norm  # 对权重进行去调制

        # 调整 fmap 的形状以适配卷积操作
        fmap = rearrange(fmap, "b c t h w -> 1 (b c) t h w")

        # 调整权重的形状
        weights = rearrange(weights, "b o ... -> (b o) ...")

        # 对 fmap 进行填充
        fmap = F.pad(fmap, self.padding, mode=self.pad_mode)
        # 进行 3D 卷积操作
        fmap = F.conv3d(fmap, weights, groups=b)

        # 调整输出的形状为 (b, o, ...)
        return rearrange(fmap, "1 (b o) ... -> b o ...", b=b)


# 进行步幅卷积以降采样


class SpatialDownsample2x(Module):
    # 初始化函数，设置降采样的维度和卷积参数
    def __init__(self, dim, dim_out=None, kernel_size=3, antialias=False):
        super().__init__()  # 调用父类的初始化函数
        dim_out = default(dim_out, dim)  # 如果未指定 dim_out，则默认与 dim 相同
        # 根据是否启用抗混叠设置可能的模糊操作
        self.maybe_blur = Blur() if antialias else identity
        # 初始化 2D 卷积，步幅为 2，填充为核大小的一半
        self.conv = nn.Conv2d(dim, dim_out, kernel_size, stride=2, padding=kernel_size // 2)

    # 前向传播函数
    def forward(self, x):
        # 进行模糊处理（如果需要）
        x = self.maybe_blur(x, space_only=True)

        # 调整输入的形状
        x = rearrange(x, "b c t h w -> b t c h w")
        x, ps = pack_one(x, "* c h w")  # 将数据打包以便处理

        out = self.conv(x)  # 进行卷积操作

        out = unpack_one(out, ps, "* c h w")  # 解包数据
        # 调整输出的形状为 (b, c, t, h, w)
        out = rearrange(out, "b t c h w -> b c t h w")
        return out


# 时间维度的降采样


class TimeDownsample2x(Module):
    # 初始化函数，设置降采样的维度和卷积参数
    def __init__(self, dim, dim_out=None, kernel_size=3, antialias=False):
        super().__init__()  # 调用父类的初始化函数
        dim_out = default(dim_out, dim)  # 如果未指定 dim_out，则默认与 dim 相同
        # 根据是否启用抗混叠设置可能的模糊操作
        self.maybe_blur = Blur() if antialias else identity
        self.time_causal_padding = (kernel_size - 1, 0)  # 设置时间因果填充
        # 初始化 1D 卷积，步幅为 2
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, stride=2)
    # 前向传播方法，接收输入 x
        def forward(self, x):
            # 根据时间维度可能模糊处理输入 x
            x = self.maybe_blur(x, time_only=True)
    
            # 重排张量维度，从 (batch, channels, time, height, width) 到 (batch, height, width, channels, time)
            x = rearrange(x, "b c t h w -> b h w c t")
            # 将重排后的张量打包，返回新张量和打包信息 ps
            x, ps = pack_one(x, "* c t")
    
            # 对张量 x 进行填充，添加时间因果填充
            x = F.pad(x, self.time_causal_padding)
            # 使用卷积层处理填充后的张量
            out = self.conv(x)
    
            # 解包卷积输出，恢复到原始张量形状
            out = unpack_one(out, ps, "* c t")
            # 再次重排维度，从 (batch, height, width, channels, time) 到 (batch, channels, time, height, width)
            out = rearrange(out, "b h w c t -> b c t h w")
            # 返回最终输出
            return out
# 深度到空间的上采样


class SpatialUpsample2x(Module):
    # 初始化上采样模块，指定输入和输出通道
    def __init__(self, dim, dim_out=None):
        # 调用父类构造函数
        super().__init__()
        # 如果未指定输出维度，则将其设置为输入维度
        dim_out = default(dim_out, dim)
        # 创建一个卷积层，将输入通道扩展为输出通道的四倍
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        # 定义网络结构，包括卷积、激活和重排列
        self.net = nn.Sequential(conv, nn.SiLU(), Rearrange("b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2))

        # 初始化卷积层的权重
        self.init_conv_(conv)

    # 初始化卷积层的权重
    def init_conv_(self, conv):
        # 获取卷积层的输出通道、输入通道、高度和宽度
        o, i, h, w = conv.weight.shape
        # 创建一个新的权重张量，形状为输出通道的四分之一
        conv_weight = torch.empty(o // 4, i, h, w)
        # 使用 He 均匀初始化卷积权重
        nn.init.kaiming_uniform_(conv_weight)
        # 扩展权重张量，使输出通道数量恢复到四倍
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")

        # 将新的权重复制到卷积层
        conv.weight.data.copy_(conv_weight)
        # 将卷积层的偏置初始化为零
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(self, x):
        # 重排列输入张量，使维度顺序为 (batch, time, channel, height, width)
        x = rearrange(x, "b c t h w -> b t c h w")
        # 打包张量，将其维度压缩
        x, ps = pack_one(x, "* c h w")

        # 通过网络进行前向传播
        out = self.net(x)

        # 解包输出张量，恢复维度
        out = unpack_one(out, ps, "* c h w")
        # 重排列输出张量
        out = rearrange(out, "b t c h w -> b c t h w")
        return out


class TimeUpsample2x(Module):
    # 初始化时间上采样模块，指定输入和输出通道
    def __init__(self, dim, dim_out=None):
        # 调用父类构造函数
        super().__init__()
        # 如果未指定输出维度，则将其设置为输入维度
        dim_out = default(dim_out, dim)
        # 创建一个一维卷积层，将输入通道扩展为输出通道的两倍
        conv = nn.Conv1d(dim, dim_out * 2, 1)

        # 定义网络结构，包括卷积、激活和重排列
        self.net = nn.Sequential(conv, nn.SiLU(), Rearrange("b (c p) t -> b c (t p)", p=2))

        # 初始化卷积层的权重
        self.init_conv_(conv)

    # 初始化卷积层的权重
    def init_conv_(self, conv):
        # 获取卷积层的输出通道、输入通道和时间维度
        o, i, t = conv.weight.shape
        # 创建一个新的权重张量，形状为输出通道的二分之一
        conv_weight = torch.empty(o // 2, i, t)
        # 使用 He 均匀初始化卷积权重
        nn.init.kaiming_uniform_(conv_weight)
        # 扩展权重张量，使输出通道数量恢复到两倍
        conv_weight = repeat(conv_weight, "o ... -> (o 2) ...")

        # 将新的权重复制到卷积层
        conv.weight.data.copy_(conv_weight)
        # 将卷积层的偏置初始化为零
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(self, x):
        # 重排列输入张量，使维度顺序为 (batch, height, width, channel, time)
        x = rearrange(x, "b c t h w -> b h w c t")
        # 打包张量，将其维度压缩
        x, ps = pack_one(x, "* c t")

        # 通过网络进行前向传播
        out = self.net(x)

        # 解包输出张量，恢复维度
        out = unpack_one(out, ps, "* c t")
        # 重排列输出张量
        out = rearrange(out, "b h w c t -> b c t h w")
        return out


# 自编码器 - 这里只提供最佳变体，使用因果卷积 3D


# 创建一个带有填充的卷积层，保持输入和输出维度相同
def SameConv2d(dim_in, dim_out, kernel_size):
    # 将核大小转换为元组，如果不是的话
    kernel_size = cast_tuple(kernel_size, 2)
    # 计算填充，以保持卷积后尺寸不变
    padding = [k // 2 for k in kernel_size]
    # 返回具有指定参数的卷积层
    return nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding)


class CausalConv3d(Module):
    # 定义因果卷积 3D 的构造函数
    @beartype
    def __init__(
        # 输入通道、输出通道、核大小及填充模式
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], pad_mode="constant", **kwargs
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 将 kernel_size 转换为包含 3 个元素的元组
        kernel_size = cast_tuple(kernel_size, 3)

        # 解包 kernel_size 为时间、 altura 和宽度的大小
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # 确保高度和宽度的内核大小都是奇数
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        # 从关键字参数中弹出膨胀和步幅的值，默认值为 1
        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

        # 设置填充模式
        self.pad_mode = pad_mode
        # 计算时间维度的填充大小
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        # 计算高度和宽度的填充大小
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        # 保存时间填充大小
        self.time_pad = time_pad
        # 设置时间因果填充，包含宽度、高度和时间的填充大小
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        # 设置步幅为 (步幅, 1, 1) 的元组
        stride = (stride, 1, 1)
        # 设置膨胀为 (膨胀, 1, 1) 的元组
        dilation = (dilation, 1, 1)
        # 创建 3D 卷积层
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        # 根据输入 x 的形状和时间填充确定填充模式
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        # 对输入 x 进行填充
        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        # 返回经过卷积层处理后的输出
        return self.conv(x)
# 装饰器，用于类型检查
@beartype
# 定义残差单元，包含卷积操作和激活函数
def ResidualUnit(dim, kernel_size: Union[int, Tuple[int, int, int]], pad_mode: str = "constant"):
    # 创建一个顺序模型，包含一系列层
    net = Sequential(
        # 使用因果卷积进行3D卷积，指定输入和输出通道、卷积核大小及填充方式
        CausalConv3d(dim, dim, kernel_size, pad_mode=pad_mode),
        # 应用ELU激活函数
        nn.ELU(),
        # 进行1x1x1卷积
        nn.Conv3d(dim, dim, 1),
        # 再次应用ELU激活函数
        nn.ELU(),
        # 使用Squeeze and Excitation模块
        SqueezeExcite(dim),
    )

    # 返回残差模块
    return Residual(net)


# 装饰器，用于类型检查
@beartype
# 定义带条件输入的残差单元模块
class ResidualUnitMod(Module):
    # 初始化方法，定义模块参数
    def __init__(
        self, dim, kernel_size: Union[int, Tuple[int, int, int]], *, dim_cond, pad_mode: str = "constant", demod=True
    ):
        # 调用父类构造函数
        super().__init__()
        # 将卷积核大小转换为元组，确保为三维
        kernel_size = cast_tuple(kernel_size, 3)
        # 解包卷积核的时间、高度和宽度
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        # 断言高度和宽度的卷积核大小相同
        assert height_kernel_size == width_kernel_size

        # 定义条件输入的线性层
        self.to_cond = nn.Linear(dim_cond, dim)

        # 定义条件卷积层
        self.conv = Conv3DMod(
            dim=dim,
            spatial_kernel=height_kernel_size,
            time_kernel=time_kernel_size,
            causal=True,
            demod=demod,
            pad_mode=pad_mode,
        )

        # 定义输出卷积层
        self.conv_out = nn.Conv3d(dim, dim, 1)

    # 装饰器，用于类型检查
    @beartype
    # 前向传播方法，定义模块的计算流程
    def forward(
        self,
        x,
        cond: Tensor,
    ):
        # 保存输入以便于后续相加
        res = x
        # 将条件输入通过线性层转换
        cond = self.to_cond(cond)

        # 使用条件卷积处理输入
        x = self.conv(x, cond=cond)
        # 应用ELU激活函数
        x = F.elu(x)
        # 通过输出卷积层处理
        x = self.conv_out(x)
        # 再次应用ELU激活函数
        x = F.elu(x)
        # 返回残差连接的结果
        return x + res


# 定义因果卷积转置模块
class CausalConvTranspose3d(Module):
    # 初始化方法，定义模块参数
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], *, time_stride, **kwargs):
        # 调用父类构造函数
        super().__init__()
        # 将卷积核大小转换为元组，确保为三维
        kernel_size = cast_tuple(kernel_size, 3)

        # 解包卷积核的时间、高度和宽度
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # 断言高度和宽度的卷积核大小为奇数
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        # 设置上采样因子
        self.upsample_factor = time_stride

        # 计算高度和宽度的填充大小
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        # 定义步幅和填充
        stride = (time_stride, 1, 1)
        padding = (0, height_pad, width_pad)

        # 定义转置卷积层
        self.conv = nn.ConvTranspose3d(chan_in, chan_out, kernel_size, stride, padding=padding, **kwargs)

    # 前向传播方法，定义模块的计算流程
    def forward(self, x):
        # 确保输入为5维
        assert x.ndim == 5
        # 获取时间维度的大小
        t = x.shape[2]

        # 通过转置卷积层处理输入
        out = self.conv(x)

        # 切片以匹配上采样后的时间维度
        out = out[..., : (t * self.upsample_factor), :, :]
        # 返回处理后的输出
        return out


# 定义损失分解的命名元组
LossBreakdown = namedtuple(
    "LossBreakdown",
    [
        # 重构损失
        "recon_loss",
        # 辅助损失
        "lfq_aux_loss",
        # 量化器损失分解
        "quantizer_loss_breakdown",
        # 感知损失
        "perceptual_loss",
        # 对抗生成损失
        "adversarial_gen_loss",
        # 自适应对抗权重
        "adaptive_adversarial_weight",
        # 多尺度生成损失
        "multiscale_gen_losses",
        # 多尺度生成自适应权重
        "multiscale_gen_adaptive_weights",
    ],
)

# 定义鉴别器损失分解的命名元组
DiscrLossBreakdown = namedtuple("DiscrLossBreakdown", ["discr_loss", "multiscale_discr_losses", "gradient_penalty"])


# 定义视频分词器模块
class VideoTokenizer(Module):
    # 装饰器，用于类型检查
    @beartype
    # 初始化方法，用于创建类的实例，接受多个参数
        def __init__(
            self,
            *,  # 使用关键字参数
            image_size,  # 输入图像的尺寸
            layers: Tuple[Union[str, Tuple[str, int]], ...] = ("residual", "residual", "residual"),  # 网络层的类型与配置
            residual_conv_kernel_size=3,  # 残差卷积核的大小
            num_codebooks=1,  # 代码本的数量
            codebook_size: Optional[int] = None,  # 代码本的大小（可选）
            channels=3,  # 输入图像的通道数（如RGB）
            init_dim=64,  # 初始化维度
            max_dim=float("inf"),  # 最大维度，默认为无穷大
            dim_cond=None,  # 条件维度（可选）
            dim_cond_expansion_factor=4.0,  # 条件维度扩展因子
            input_conv_kernel_size: Tuple[int, int, int] = (7, 7, 7),  # 输入卷积核的大小
            output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),  # 输出卷积核的大小
            pad_mode: str = "constant",  # 填充模式，默认为常数填充
            lfq_entropy_loss_weight=0.1,  # LFQ熵损失权重
            lfq_commitment_loss_weight=1.0,  # LFQ承诺损失权重
            lfq_diversity_gamma=2.5,  # LFQ多样性超参数
            quantizer_aux_loss_weight=1.0,  # 量化辅助损失权重
            lfq_activation=nn.Identity(),  # LFQ激活函数，默认为恒等函数
            use_fsq=False,  # 是否使用FSQ
            fsq_levels: Optional[List[int]] = None,  # FSQ的级别（可选）
            attn_dim_head=32,  # 注意力维度头大小
            attn_heads=8,  # 注意力头的数量
            attn_dropout=0.0,  # 注意力的丢弃率
            linear_attn_dim_head=8,  # 线性注意力维度头大小
            linear_attn_heads=16,  # 线性注意力头的数量
            vgg: Optional[Module] = None,  # VGG模型（可选）
            vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,  # VGG权重
            perceptual_loss_weight=1e-1,  # 感知损失权重
            discr_kwargs: Optional[dict] = None,  # 判别器参数（可选）
            multiscale_discrs: Tuple[Module, ...] = tuple(),  # 多尺度判别器
            use_gan=True,  # 是否使用GAN
            adversarial_loss_weight=1.0,  # 对抗损失权重
            grad_penalty_loss_weight=10.0,  # 梯度惩罚损失权重
            multiscale_adversarial_loss_weight=1.0,  # 多尺度对抗损失权重
            flash_attn=True,  # 是否使用闪存注意力
            separate_first_frame_encoding=False,  # 是否分开第一帧编码
        @property
        def device(self):  # 属性方法，返回设备信息
            return self.zero.device
    
        @classmethod
        def init_and_load_from(cls, path, strict=True):  # 类方法，用于初始化并从指定路径加载模型
            path = Path(path)  # 将路径转换为Path对象
            assert path.exists()  # 确保路径存在
            pkg = torch.load(str(path), map_location="cpu")  # 从指定路径加载模型，映射到CPU
    
            assert "config" in pkg, "model configs were not found in this saved checkpoint"  # 确保配置存在
    
            config = pickle.loads(pkg["config"])  # 反序列化配置
            tokenizer = cls(**config)  # 使用配置创建类的实例
            tokenizer.load(path, strict=strict)  # 加载模型权重
            return tokenizer  # 返回初始化的tokenizer实例
    
        def parameters(self):  # 返回模型的所有可训练参数
            return [
                *self.conv_in.parameters(),  # 输入卷积层的参数
                *self.conv_in_first_frame.parameters(),  # 第一帧输入卷积层的参数
                *self.conv_out_first_frame.parameters(),  # 第一帧输出卷积层的参数
                *self.conv_out.parameters(),  # 输出卷积层的参数
                *self.encoder_layers.parameters(),  # 编码层的参数
                *self.decoder_layers.parameters(),  # 解码层的参数
                *self.encoder_cond_in.parameters(),  # 编码条件输入的参数
                *self.decoder_cond_in.parameters(),  # 解码条件输入的参数
                *self.quantizers.parameters(),  # 量化器的参数
            ]
    
        def discr_parameters(self):  # 返回判别器的参数
            return self.discr.parameters()  # 获取判别器的可训练参数
    
        def copy_for_eval(self):  # 创建用于评估的模型副本
            device = self.device  # 获取当前设备
            vae_copy = copy.deepcopy(self.cpu())  # 深拷贝模型并转到CPU
    
            maybe_del_attr_(vae_copy, "discr")  # 删除判别器属性（如果存在）
            maybe_del_attr_(vae_copy, "vgg")  # 删除VGG属性（如果存在）
            maybe_del_attr_(vae_copy, "multiscale_discrs")  # 删除多尺度判别器属性（如果存在）
    
            vae_copy.eval()  # 设置模型为评估模式
            return vae_copy.to(device)  # 将模型移动到原设备并返回
    
        @remove_vgg  # 装饰器，用于去掉VGG的相关内容
        def state_dict(self, *args, **kwargs):  # 返回模型的状态字典
            return super().state_dict(*args, **kwargs)  # 调用父类方法获取状态字典
    
        @remove_vgg  # 装饰器，用于去掉VGG的相关内容
        def load_state_dict(self, *args, **kwargs):  # 加载状态字典
            return super().load_state_dict(*args, **kwargs)  # 调用父类方法加载状态字典
    # 保存模型参数到指定路径
    def save(self, path, overwrite=True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 如果 overwrite 为 False 且路径已存在，则抛出异常
        assert overwrite or not path.exists(), f"{str(path)} already exists"

        # 创建包含模型参数、版本和配置的字典
        pkg = dict(model_state_dict=self.state_dict(), version=__version__, config=self._configs)

        # 将字典保存到指定路径
        torch.save(pkg, str(path))

    # 从指定路径加载模型参数
    def load(self, path, strict=True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 如果路径不存在，则抛出异常
        assert path.exists()

        # 加载保存的模型参数
        pkg = torch.load(str(path))
        state_dict = pkg.get("model_state_dict")
        version = pkg.get("version")

        # 断言模型参数存在
        assert exists(state_dict)

        # 如果存在版本信息，则打印加载的版本信息
        if exists(version):
            print(f"loading checkpointed tokenizer from version {version}")

        # 加载模型参数到当前模型
        self.load_state_dict(state_dict, strict=strict)

    # 编码视频
    @beartype
    def encode(self, video: Tensor, quantize=False, cond: Optional[Tensor] = None, video_contains_first_frame=True):
        # 是否分开编码第一帧
        encode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        # 是否对视频进行填充
        if video_contains_first_frame:
            video_len = video.shape[2]
            video = pad_at_dim(video, (self.time_padding, 0), value=0.0, dim=2)
            video_packed_shape = [torch.Size([self.time_padding]), torch.Size([]), torch.Size([video_len - 1])]

        # 如果需要条件编码，则对条件进行处理
        assert (not self.has_cond) or exists(
            cond
        ), "`cond` must be passed into tokenizer forward method since conditionable layers were specified"

        if exists(cond):
            assert cond.shape == (video.shape[0], self.dim_cond)
            cond = self.encoder_cond_in(cond)
            cond_kwargs = dict(cond=cond)

        # 初始卷积
        if encode_first_frame_separately:
            pad, first_frame, video = unpack(video, video_packed_shape, "b c * h w")
            first_frame = self.conv_in_first_frame(first_frame)

        video = self.conv_in(video)

        if encode_first_frame_separately:
            video, _ = pack([first_frame, video], "b c * h w")
            video = pad_at_dim(video, (self.time_padding, 0), dim=2)

        # 编码器层
        for fn, has_cond in zip(self.encoder_layers, self.has_cond_across_layers):
            layer_kwargs = dict()

            if has_cond:
                layer_kwargs = cond_kwargs

            video = fn(video, **layer_kwargs)

        # 是否进行量化
        maybe_quantize = identity if not quantize else self.quantizers

        return maybe_quantize(video)

    @beartype
    # 从编码索引解码，将编码转换为原始数据
    def decode_from_code_indices(self, codes: Tensor, cond: Optional[Tensor] = None, video_contains_first_frame=True):
        # 断言编码的数据类型为 long 或 int32
        assert codes.dtype in (torch.long, torch.int32)

        # 如果编码的维度为2，则重新排列成视频编码的形状
        if codes.ndim == 2:
            video_code_len = codes.shape[-1]
            assert divisible_by(
                video_code_len, self.fmap_size**2
            ), f"flattened video ids must have a length ({video_code_len}) that is divisible by the fmap size ({self.fmap_size}) squared ({self.fmap_size ** 2})"
            codes = rearrange(codes, "b (f h w) -> b f h w", h=self.fmap_size, w=self.fmap_size)

        # 将索引编码转换为量化编码
        quantized = self.quantizers.indices_to_codes(codes)

        # 调用解码方法，返回解码后的视频
        return self.decode(quantized, cond=cond, video_contains_first_frame=video_contains_first_frame)

    # 解码方法
    @beartype
    def decode(self, quantized: Tensor, cond: Optional[Tensor] = None, video_contains_first_frame=True):
        # 如果需要单独解码第一帧，则设置为True
        decode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        batch = quantized.shape[0]

        # 如果需要条件编码，则进行条件编码
        assert (not self.has_cond) or exists(
            cond
        ), "`cond` must be passed into tokenizer forward method since conditionable layers were specified"
        if exists(cond):
            assert cond.shape == (batch, self.dim_cond)
            cond = self.decoder_cond_in(cond)
            cond_kwargs = dict(cond=cond)

        # 解码层
        x = quantized
        for fn, has_cond in zip(self.decoder_layers, reversed(self.has_cond_across_layers)):
            layer_kwargs = dict()
            if has_cond:
                layer_kwargs = cond_kwargs
            x = fn(x, **layer_kwargs)

        # 转换为像素
        if decode_first_frame_separately:
            left_pad, xff, x = (
                x[:, :, : self.time_padding],
                x[:, :, self.time_padding],
                x[:, :, (self.time_padding + 1) :],
            )
            out = self.conv_out(x)
            outff = self.conv_out_first_frame(xff)
            video, _ = pack([outff, out], "b c * h w")
        else:
            video = self.conv_out(x)
            # 如果视频有填充，则移除填充
            if video_contains_first_frame:
                video = video[:, :, self.time_padding :]

        return video

    # 无梯度的标记方法，用于标记不需要梯度的操作
    @torch.no_grad()
    def tokenize(self, video):
        self.eval()
        return self.forward(video, return_codes=True)

    # 前向传播方法
    @beartype
    def forward(
        self,
        video_or_images: Tensor,
        cond: Optional[Tensor] = None,
        return_loss=False,
        return_codes=False,
        return_recon=False,
        return_discr_loss=False,
        return_recon_loss_only=False,
        apply_gradient_penalty=True,
        video_contains_first_frame=True,
        adversarial_loss_weight=None,
        multiscale_adversarial_loss_weight=None,
# 主类定义
class MagViT2(Module):
    # 构造函数，初始化类
    def __init__(self):
        # 调用父类的构造函数
        super().__init__()

    # 前向传播函数，处理输入数据
    def forward(self, x):
        # 返回输入数据 x
        return x
```