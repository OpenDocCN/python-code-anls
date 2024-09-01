# `.\flux\src\flux\modules\autoencoder.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass

# 导入 PyTorch 库
import torch
# 从 einops 模块导入 rearrange 函数
from einops import rearrange
# 从 torch 库导入 Tensor 和 nn 模块
from torch import Tensor, nn


# 定义 AutoEncoder 的参数数据类
@dataclass
class AutoEncoderParams:
    resolution: int  # 图像分辨率
    in_channels: int  # 输入通道数
    ch: int  # 基本通道数
    out_ch: int  # 输出通道数
    ch_mult: list[int]  # 通道数的增减比例
    num_res_blocks: int  # 残差块数量
    z_channels: int  # 潜在通道数
    scale_factor: float  # 缩放因子
    shift_factor: float  # 偏移因子


# 定义 swish 激活函数
def swish(x: Tensor) -> Tensor:
    # 使用 sigmoid 函数调节 x 的激活值
    return x * torch.sigmoid(x)


# 定义注意力块类
class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # 初始化归一化层
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        # 初始化用于计算注意力的卷积层
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    # 注意力机制函数
    def attention(self, h_: Tensor) -> Tensor:
        # 归一化输入
        h_ = self.norm(h_)
        # 计算 q, k, v
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 获取 q, k, v 的维度
        b, c, h, w = q.shape
        # 重排列 q, k, v
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        # 应用缩放点积注意力
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        # 将输出重排列为原始维度
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    # 前向传播函数
    def forward(self, x: Tensor) -> Tensor:
        # 添加注意力机制后的输出到原始输入
        return x + self.proj_out(self.attention(x))


# 定义残差块类
class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # 初始化归一化层和卷积层
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入和输出通道数不同，初始化快捷连接
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播函数
    def forward(self, x):
        h = x
        # 通过第一层归一化、激活和卷积
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        # 通过第二层归一化、激活和卷积
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        # 如果输入和输出通道数不同，应用快捷连接
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        # 返回残差连接的结果
        return x + h


# 定义下采样类
class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # 在 torch conv 中没有非对称填充，必须手动处理
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    # 前向传播函数，接受一个 Tensor 作为输入
        def forward(self, x: Tensor):
            # 定义 padding 的大小，分别是右边 1、下边 1
            pad = (0, 1, 0, 1)
            # 对输入 Tensor 进行 padding，填充值为 0
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            # 将 padding 过的 Tensor 通过卷积层
            x = self.conv(x)
            # 返回卷积后的结果
            return x
# 定义上采样模块，继承自 nn.Module
class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # 创建卷积层，用于对输入特征图进行卷积操作
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        # 对输入特征图进行双线性插值上采样，扩大尺寸为原来的2倍
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 对上采样后的特征图应用卷积层
        x = self.conv(x)
        # 返回处理后的特征图
        return x


# 定义编码器模块，继承自 nn.Module
class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # 输入层卷积，用于初始化特征图
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # 设置每层的输入和输出通道数
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                # 添加残差块到当前层
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                # 添加下采样层
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # 中间层，包括两个残差块和一个注意力块
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # 输出层，包括归一化和卷积层
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # 对输入特征图进行下采样
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # 中间处理
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # 输出处理
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        # 返回最终处理后的特征图
        return h


# 定义解码器模块，继承自 nn.Module
class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.ch = ch
        # 保存多分辨率通道数的数量
        self.num_resolutions = len(ch_mult)
        # 保存残差块的数量
        self.num_res_blocks = num_res_blocks
        # 保存图像分辨率
        self.resolution = resolution
        # 保存输入通道数
        self.in_channels = in_channels
        # 计算最终分辨率的缩放因子
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # 计算最低分辨率下的输入通道数和分辨率
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # 定义潜在变量 z 的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z 到 block_in 的卷积层
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间层模块
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # 上采样模块
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # 当前分辨率下的输出通道数
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                # 添加残差块
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                # 添加上采样层
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            # 将上采样模块插入列表开头，保持顺序一致
            self.up.insert(0, up)

        # 输出归一化层
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # 输出卷积层
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # 将 z 传入 conv_in 层
        h = self.conv_in(z)

        # 通过中间层
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                # 上采样
                h = self.up[i_level].upsample(h)

        # 结束层
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        # 返回最终输出
        return h
# 定义对角高斯分布的神经网络模块
class DiagonalGaussian(nn.Module):
    # 初始化方法，定义是否采样及分块维度
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        # 是否进行采样
        self.sample = sample
        # 进行分块操作的维度
        self.chunk_dim = chunk_dim

    # 前向传播方法
    def forward(self, z: Tensor) -> Tensor:
        # 将输入张量 z 按指定维度 chunk_dim 划分为两个张量 mean 和 logvar
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            # 如果需要采样，计算标准差并从标准正态分布中生成随机样本
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            # 否则只返回均值
            return mean


# 定义自编码器的神经网络模块
class AutoEncoder(nn.Module):
    # 初始化方法，定义编码器、解码器及高斯分布
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        # 创建编码器实例，传入相应参数
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        # 创建解码器实例，传入相应参数
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        # 创建对角高斯分布实例
        self.reg = DiagonalGaussian()

        # 设置缩放因子和偏移因子
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    # 编码方法，将输入 x 进行编码并调整缩放和偏移
    def encode(self, x: Tensor) -> Tensor:
        # 通过编码器获取 z，随后通过对角高斯分布进行处理
        z = self.reg(self.encoder(x))
        # 对 z 进行缩放和偏移
        z = self.scale_factor * (z - self.shift_factor)
        return z

    # 解码方法，将 z 解码为输出
    def decode(self, z: Tensor) -> Tensor:
        # 对 z 进行逆操作，恢复到编码前的尺度
        z = z / self.scale_factor + self.shift_factor
        # 使用解码器进行解码
        return self.decoder(z)

    # 前向传播方法，执行编码和解码
    def forward(self, x: Tensor) -> Tensor:
        # 先编码再解码
        return self.decode(self.encode(x))
```