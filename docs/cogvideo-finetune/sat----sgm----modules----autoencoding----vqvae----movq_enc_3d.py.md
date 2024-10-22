# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\movq_enc_3d.py`

```py
# pytorch_diffusion + derived encoder decoder
import math  # 导入数学库，提供数学函数
import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的功能性操作模块
import numpy as np  # 导入 NumPy 库，用于数组和数学操作

from beartype import beartype  # 导入 beartype 库，用于类型检查
from beartype.typing import Union, Tuple, Optional, List  # 导入类型注解
from einops import rearrange  # 导入 einops 库，用于数组重排


def cast_tuple(t, length=1):  # 定义将输入转换为元组的函数
    return t if isinstance(t, tuple) else ((t,) * length)  # 如果是元组，返回原值，否则返回长度为 length 的元组


def divisible_by(num, den):  # 定义判断是否整除的函数
    return (num % den) == 0  # 返回 num 是否能被 den 整除


def is_odd(n):  # 定义判断数字是否为奇数的函数
    return not divisible_by(n, 2)  # 返回 n 是否为奇数


def get_timestep_embedding(timesteps, embedding_dim):  # 定义生成时间步嵌入的函数
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1  # 确保 timesteps 是一维的

    half_dim = embedding_dim // 2  # 计算嵌入维度的一半
    emb = math.log(10000) / (half_dim - 1)  # 计算正弦嵌入的缩放因子
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)  # 生成指数衰减的嵌入
    emb = emb.to(device=timesteps.device)  # 将嵌入移动到 timesteps 的设备上
    emb = timesteps.float()[:, None] * emb[None, :]  # 扩展 timesteps 和嵌入的维度以便相乘
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # 将正弦和余弦嵌入在一起
    if embedding_dim % 2 == 1:  # 如果嵌入维度为奇数，进行零填充
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))  # 在最后填充一个零
    return emb  # 返回生成的嵌入


def nonlinearity(x):  # 定义非线性激活函数
    # swish
    return x * torch.sigmoid(x)  # 返回 Swish 激活值


class CausalConv3d(nn.Module):  # 定义因果三维卷积层的类
    @beartype  # 应用类型检查装饰器
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], pad_mode="constant", **kwargs):  # 初始化函数
        super().__init__()  # 调用父类构造函数
        kernel_size = cast_tuple(kernel_size, 3)  # 将 kernel_size 转换为三元组

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size  # 解包卷积核的大小

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)  # 确保高度和宽度的卷积核为奇数

        dilation = kwargs.pop("dilation", 1)  # 从参数中弹出膨胀值，默认为1
        stride = kwargs.pop("stride", 1)  # 从参数中弹出步幅值，默认为1

        self.pad_mode = pad_mode  # 保存填充模式
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)  # 计算时间维度的填充量
        height_pad = height_kernel_size // 2  # 计算高度维度的填充量
        width_pad = width_kernel_size // 2  # 计算宽度维度的填充量

        self.height_pad = height_pad  # 保存高度填充量
        self.width_pad = width_pad  # 保存宽度填充量
        self.time_pad = time_pad  # 保存时间填充量
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)  # 定义因果卷积的填充格式

        stride = (stride, 1, 1)  # 将步幅转换为三维元组
        dilation = (dilation, 1, 1)  # 将膨胀转换为三维元组
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)  # 初始化三维卷积层
    # 定义前向传播函数，接收输入张量 x
    def forward(self, x):
        # 检查填充模式是否为常数填充
        if self.pad_mode == "constant":
            # 设置三维的因果填充参数
            causal_padding_3d = (self.time_pad, 0, self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            # 对输入张量 x 进行常数填充
            x = F.pad(x, causal_padding_3d, mode="constant", value=0)
        # 检查填充模式是否为首元素填充
        elif self.pad_mode == "first":
            # 复制输入张量的首元素并在时间维度上进行填充
            pad_x = torch.cat([x[:, :, :1]] * self.time_pad, dim=2)
            # 将填充的张量与原张量拼接
            x = torch.cat([pad_x, x], dim=2)
            # 设置二维的因果填充参数
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            # 对拼接后的张量进行常数填充
            x = F.pad(x, causal_padding_2d, mode="constant", value=0)
        # 检查填充模式是否为反射填充
        elif self.pad_mode == "reflect":
            # 进行反射填充，获取输入张量的反向切片
            reflect_x = x[:, :, 1 : self.time_pad + 1, :, :].flip(dims=[2])
            # 如果反射张量的时间维度小于所需的填充时间，则进行零填充
            if reflect_x.shape[2] < self.time_pad:
                reflect_x = torch.cat(
                    [torch.zeros_like(x[:, :, :1, :, :])] * (self.time_pad - reflect_x.shape[2]) + [reflect_x], dim=2
                )
            # 将反射填充的张量与原张量拼接
            x = torch.cat([reflect_x, x], dim=2)
            # 设置二维的因果填充参数
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            # 对拼接后的张量进行常数填充
            x = F.pad(x, causal_padding_2d, mode="constant", value=0)
        # 如果填充模式无效，则引发错误
        else:
            raise ValueError("Invalid pad mode")
        # 返回经过卷积操作的结果
        return self.conv(x)
# 定义一个用于归一化的函数，适用于3D和2D数据
def Normalize3D(in_channels):  # same for 3D and 2D
    # 返回一个GroupNorm层，用于归一化输入通道
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# 定义一个3D上采样的类，继承自nn.Module
class Upsample3D(nn.Module):
    # 初始化方法，设置输入通道、是否使用卷积和时间压缩参数
    def __init__(self, in_channels, with_conv, compress_time=False):
        super().__init__()  # 调用父类初始化方法
        self.with_conv = with_conv  # 保存是否使用卷积的标志
        if self.with_conv:
            # 如果使用卷积，创建一个卷积层，输入输出通道相同
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.compress_time = compress_time  # 保存时间压缩的标志

    # 前向传播方法
    def forward(self, x):
        if self.compress_time:  # 如果启用时间压缩
            if x.shape[2] > 1:  # 如果时间维度大于1
                # 分离第一帧和其余帧
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]

                # 将第一帧上采样到原来的2倍
                x_first = torch.nn.functional.interpolate(x_first, scale_factor=2.0, mode="nearest")
                # 将其余帧上采样到原来的2倍
                x_rest = torch.nn.functional.interpolate(x_rest, scale_factor=2.0, mode="nearest")
                # 将第一帧和其余帧在时间维度上连接
                x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2)
            else:  # 如果时间维度为1
                x = x.squeeze(2)  # 去除时间维度
                # 将数据上采样到原来的2倍
                x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
                x = x[:, :, None, :, :]  # 添加时间维度
        else:  # 如果不进行时间压缩
            # 只对2D进行上采样
            t = x.shape[2]  # 获取时间维度大小
            # 重新排列数据，合并批次和时间维度
            x = rearrange(x, "b c t h w -> (b t) c h w")
            # 将数据上采样到原来的2倍
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            # 恢复数据排列
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.with_conv:  # 如果使用卷积
            t = x.shape[2]  # 获取时间维度大小
            # 重新排列数据，合并批次和时间维度
            x = rearrange(x, "b c t h w -> (b t) c h w")
            # 通过卷积层处理数据
            x = self.conv(x)
            # 恢复数据排列
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x  # 返回处理后的数据


# 定义一个3D下采样的类，继承自nn.Module
class DownSample3D(nn.Module):
    # 初始化方法，设置输入通道、是否使用卷积、时间压缩和输出通道
    def __init__(self, in_channels, with_conv, compress_time=False, out_channels=None):
        super().__init__()  # 调用父类初始化方法
        self.with_conv = with_conv  # 保存是否使用卷积的标志
        if out_channels is None:  # 如果没有指定输出通道
            out_channels = in_channels  # 设置输出通道为输入通道
        if self.with_conv:  # 如果使用卷积
            # 在torch的卷积中没有不对称填充，必须手动处理
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.compress_time = compress_time  # 保存时间压缩的标志
    # 定义前向传播函数，接受输入 x
        def forward(self, x):
            # 如果设置了 compress_time 为 True
            if self.compress_time:
                # 获取输入张量 x 的高度和宽度
                h, w = x.shape[-2:]
                # 重新排列输入张量 x 的维度，将 (batch, channel, time, height, width) 转换为 ((batch*height*width), channel, time)
                x = rearrange(x, "b c t h w -> (b h w) c t")
    
                # 分离出第一帧和其余帧
                x_first, x_rest = x[..., 0], x[..., 1:]
    
                # 如果剩余帧存在
                if x_rest.shape[-1] > 0:
                    # 对剩余帧进行平均池化，减少时间维度
                    x_rest = torch.nn.functional.avg_pool1d(x_rest, kernel_size=2, stride=2)
                # 将第一帧和池化后的剩余帧拼接在一起
                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                # 重新排列回原始维度，恢复为 (batch, channel, time, height, width)
                x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)
    
            # 如果设置了 with_conv 为 True
            if self.with_conv:
                # 定义填充参数，确保维度匹配
                pad = (0, 1, 0, 1)
                # 对输入张量 x 进行零填充
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                # 获取时间维度的大小
                t = x.shape[2]
                # 重新排列输入张量 x 的维度，为卷积操作准备
                x = rearrange(x, "b c t h w -> (b t) c h w")
                # 通过卷积层处理输入张量
                x = self.conv(x)
                # 重新排列输出张量，恢复维度
                x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            else:
                # 获取时间维度的大小
                t = x.shape[2]
                # 重新排列输入张量 x 的维度，准备进行池化操作
                x = rearrange(x, "b c t h w -> (b t) c h w")
                # 对输入张量进行平均池化，减小空间维度
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
                # 重新排列输出张量，恢复维度
                x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            # 返回处理后的输出张量
            return x
# 定义一个 3D 残差块，继承自 nn.Module
class ResnetBlock3D(nn.Module):
    # 初始化方法，设置输入、输出通道和其他参数
    def __init__(
        self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512, pad_mode="constant"
    ):
        # 调用父类构造函数
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels
        # 如果未指定输出通道，则与输入通道数相同
        out_channels = in_channels if out_channels is None else out_channels
        # 保存输出通道数
        self.out_channels = out_channels
        # 保存是否使用卷积捷径的标志
        self.use_conv_shortcut = conv_shortcut

        # 初始化输入通道的归一化层
        self.norm1 = Normalize3D(in_channels)
        # 使用因果卷积初始化第一层卷积
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        # 如果时间嵌入通道数大于0，初始化时间嵌入投影层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化输出通道的归一化层
        self.norm2 = Normalize3D(out_channels)
        # 初始化 dropout 层
        self.dropout = torch.nn.Dropout(dropout)
        # 使用因果卷积初始化第二层卷积
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        # 如果输入和输出通道不相同
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                # 使用因果卷积初始化捷径卷积
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
            else:
                # 初始化 1x1 卷积作为捷径
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法
    def forward(self, x, temb):
        # 将输入赋值给 h
        h = x
        # 对 h 进行归一化
        h = self.norm1(h)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过第一层卷积处理 h
        h = self.conv1(h)

        # 如果时间嵌入存在，将其投影并添加到 h
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        # 对 h 进行归一化
        h = self.norm2(h)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 应用 dropout
        h = self.dropout(h)
        # 通过第二层卷积处理 h
        h = self.conv2(h)

        # 如果输入和输出通道不相同
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                # 使用捷径卷积处理 x
                x = self.conv_shortcut(x)
            else:
                # 使用 1x1 卷积处理 x
                x = self.nin_shortcut(x)

        # 返回输入和处理后的 h 的和
        return x + h
    # 初始化方法，接收输入通道数
        def __init__(self, in_channels):
            # 调用父类的初始化方法
            super().__init__()
            # 保存输入通道数
            self.in_channels = in_channels
    
            # 创建 3D 归一化层
            self.norm = Normalize3D(in_channels)
            # 创建查询卷积层，使用 1x1 卷积
            self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            # 创建键卷积层，使用 1x1 卷积
            self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            # 创建值卷积层，使用 1x1 卷积
            self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            # 创建输出投影卷积层，使用 1x1 卷积
            self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
        # 前向传播方法，接收输入 x
        def forward(self, x):
            # 将输入赋值给 h_
            h_ = x
            # 对输入进行归一化
            h_ = self.norm(h_)
    
            # 获取时间维度的大小
            t = h_.shape[2]
            # 调整 h_ 的形状以便于后续计算
            h_ = rearrange(h_, "b c t h w -> (b t) c h w")
    
            # 计算查询向量
            q = self.q(h_)
            # 计算键向量
            k = self.k(h_)
            # 计算值向量
            v = self.v(h_)
    
            # 计算注意力
            b, c, h, w = q.shape
            # 调整查询向量的形状
            q = q.reshape(b, c, h * w)
            # 调整查询向量的维度顺序
            q = q.permute(0, 2, 1)  # b,hw,c
            # 调整键向量的形状
            k = k.reshape(b, c, h * w)  # b,c,hw
    
            # # 原始版本，fp16 中出现 nan
            # w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            # w_ = w_ * (int(c)**(-0.5))
            # 在查询向量上实现 c**-0.5 的缩放
            q = q * (int(c) ** (-0.5))
            # 计算注意力权重
            w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    
            # 对注意力权重进行 softmax 归一化
            w_ = torch.nn.functional.softmax(w_, dim=2)
    
            # 处理值向量
            v = v.reshape(b, c, h * w)
            # 调整注意力权重的维度顺序
            w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
            # 计算最终输出
            h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            # 将输出调整回原始形状
            h_ = h_.reshape(b, c, h, w)
    
            # 通过输出投影层处理结果
            h_ = self.proj_out(h_)
    
            # 调整输出的形状以匹配输入
            h_ = rearrange(h_, "(b t) c h w -> b c t h w", t=t)
    
            # 返回输入与输出的相加结果
            return x + h_
# 定义一个3D编码器类，继承自PyTorch的nn.Module
class Encoder3D(nn.Module):
    # 初始化方法，接收多个参数用于配置编码器
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数倍增因子
        num_res_blocks,  # 每个分辨率的残差块数量
        attn_resolutions,  # 注意力分辨率
        dropout=0.0,  # dropout比率
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入数据的通道数
        resolution,  # 输入数据的分辨率
        z_channels,  # 潜在变量的通道数
        double_z=True,  # 是否使用双重潜在变量
        pad_mode="first",  # 填充模式
        temporal_compress_times=4,  # 时间压缩次数
        **ignore_kwargs,  # 额外参数，忽略处理
    # 前向传播方法，定义数据如何通过网络流动
    def forward(self, x, use_cp=False):
        # 确保输入x的高度和宽度等于预期分辨率
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
        # 初始化时间步嵌入为None
        temb = None

        # 下采样过程，初始化特征列表
        hs = [self.conv_in(x)]
        # 遍历每个分辨率级别
        for i_level in range(self.num_resolutions):
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks):
                # 通过下采样块处理当前特征图
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # 如果当前级别有注意力层，则应用该层
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # 将处理后的特征图添加到特征列表中
                hs.append(h)
            # 如果不是最后一个分辨率级别，则进行下采样
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # 处理最后的特征图
        h = hs[-1]
        # 通过中间块1处理特征图
        h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)  # 注意力层（已注释）
        # 通过中间块2处理特征图
        h = self.mid.block_2(h, temb)

        # 最终处理步骤
        h = self.norm_out(h)  # 归一化输出
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.conv_out(h)  # 最终卷积操作
        # 返回处理后的特征图
        return h
```