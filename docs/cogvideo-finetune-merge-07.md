# CogVideo & CogVideoX 微调代码源码解析（八）



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

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\movq_modules.py`

```py
# pytorch_diffusion + derived encoder decoder
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 导入 NumPy 库
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    """
    该函数实现了去噪扩散概率模型中的时间步嵌入构建。
    来自 Fairseq。
    构建正弦嵌入。
    该实现与 tensor2tensor 中的实现匹配，但与“Attention Is All You Need”第 3.5 节中的描述略有不同。
    """
    # 确保 timesteps 是一维张量
    assert len(timesteps.shape) == 1

    # 计算嵌入维度的一半
    half_dim = embedding_dim // 2
    # 计算嵌入的指数缩放因子
    emb = math.log(10000) / (half_dim - 1)
    # 创建半维度的指数衰减张量
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将嵌入移动到与 timesteps 相同的设备上
    emb = emb.to(device=timesteps.device)
    # 计算时间步嵌入，使用广播机制
    emb = timesteps.float()[:, None] * emb[None, :]
    # 将正弦和余弦嵌入连接在一起
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度是奇数，则进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回生成的嵌入
    return emb


def nonlinearity(x):
    # 定义非线性激活函数：swish
    return x * torch.sigmoid(x)


class SpatialNorm(nn.Module):
    # 定义空间归一化模块
    def __init__(
        self,
        f_channels,
        zq_channels,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=False,
        **norm_layer_params,
    ):
        # 初始化父类
        super().__init__()
        # 创建归一化层
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        # 如果需要冻结归一化层的参数
        if freeze_norm_layer:
            # 将归一化层的所有参数设置为不需要梯度
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        # 是否添加卷积层
        self.add_conv = add_conv
        # 如果添加卷积层，定义卷积层
        if self.add_conv:
            self.conv = nn.Conv2d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
        # 定义用于处理 zq 的卷积层
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        # 定义另一个用于处理 zq 的卷积层
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f, zq):
        # 获取 f 的空间尺寸
        f_size = f.shape[-2:]
        # 将 zq 进行上采样以匹配 f 的尺寸
        zq = torch.nn.functional.interpolate(zq, size=f_size, mode="nearest")
        # 如果需要添加卷积层，则对 zq 进行卷积处理
        if self.add_conv:
            zq = self.conv(zq)
        # 对 f 应用归一化层
        norm_f = self.norm_layer(f)
        # 计算新的 f，结合归一化后的 f 和 zq 的卷积结果
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        # 返回新的 f
        return new_f


def Normalize(in_channels, zq_ch, add_conv):
    # 创建并返回一个 SpatialNorm 实例
    return SpatialNorm(
        in_channels,
        zq_ch,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True,
    )


class Upsample(nn.Module):
    # 定义上采样模块
    def __init__(self, in_channels, with_conv):
        # 初始化父类
        super().__init__()
        # 是否使用卷积层
        self.with_conv = with_conv
        # 如果使用卷积层，定义卷积层
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 将输入 x 上采样，放大两倍
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 如果使用卷积层，对上采样后的 x 进行卷积处理
        if self.with_conv:
            x = self.conv(x)
        # 返回处理后的 x
        return x


class Downsample(nn.Module):
    # 该类的定义未完成
    # 初始化方法，设置输入通道和是否使用卷积
        def __init__(self, in_channels, with_conv):
            # 调用父类初始化方法
            super().__init__()
            # 存储是否使用卷积的标志
            self.with_conv = with_conv
            # 如果选择使用卷积
            if self.with_conv:
                # 创建一个 2D 卷积层，卷积核大小为 3，步幅为 2，填充为 0
                # 注意：在 torch 卷积中没有非对称填充，必须手动处理
                self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    
        # 前向传播方法，接收输入张量 x
        def forward(self, x):
            # 如果选择使用卷积
            if self.with_conv:
                # 定义填充参数，添加零填充以匹配卷积输入要求
                pad = (0, 1, 0, 1)
                # 对输入 x 进行常量填充
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                # 通过卷积层处理输入 x
                x = self.conv(x)
            else:
                # 如果不使用卷积，执行平均池化操作，降低维度
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            # 返回处理后的输出 x
            return x
# 定义一个残差块类，继承自 nn.Module
class ResnetBlock(nn.Module):
    # 初始化方法，接受多个参数配置残差块
    def __init__(
        self,
        *,
        in_channels,  # 输入通道数
        out_channels=None,  # 输出通道数，默认为 None
        conv_shortcut=False,  # 是否使用卷积短接
        dropout,  # dropout 的比率
        temb_channels=512,  # 时间嵌入的通道数，默认值为 512
        zq_ch=None,  # zq 的通道数，默认为 None
        add_conv=False,  # 是否添加卷积
    ):
        # 调用父类初始化方法
        super().__init__()
        self.in_channels = in_channels  # 设置输入通道数
        # 确定输出通道数，如果未提供，则等于输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels  # 设置输出通道数
        self.use_conv_shortcut = conv_shortcut  # 设置是否使用卷积短接

        # 初始化归一化层
        self.norm1 = Normalize(in_channels, zq_ch, add_conv=add_conv)
        # 初始化第一个卷积层
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果时间嵌入通道数大于 0，初始化时间嵌入投影层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化第二个归一化层
        self.norm2 = Normalize(out_channels, zq_ch, add_conv=add_conv)
        # 初始化 dropout 层
        self.dropout = torch.nn.Dropout(dropout)
        # 初始化第二个卷积层
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入通道数与输出通道数不同，设置短接卷积
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                # 使用卷积短接
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                # 使用 1x1 卷积短接
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法，接受输入 x、时间嵌入 temb 和 zq
    def forward(self, x, temb, zq):
        h = x  # 将输入赋值给 h
        h = self.norm1(h, zq)  # 对 h 进行第一次归一化
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.conv1(h)  # 通过第一个卷积层

        # 如果时间嵌入存在，则进行相应处理
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]  # 加入时间嵌入投影

        h = self.norm2(h, zq)  # 对 h 进行第二次归一化
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.dropout(h)  # 应用 dropout
        h = self.conv2(h)  # 通过第二个卷积层

        # 如果输入和输出通道数不同，进行短接处理
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)  # 使用卷积短接
            else:
                x = self.nin_shortcut(x)  # 使用 1x1 卷积短接

        # 返回输入和 h 的和
        return x + h  # 残差连接


# 定义一个注意力块类，继承自 nn.Module
class AttnBlock(nn.Module):
    # 初始化方法，接受输入通道数和可选参数
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        # 调用父类初始化方法
        super().__init__()
        self.in_channels = in_channels  # 设置输入通道数

        # 初始化归一化层
        self.norm = Normalize(in_channels, zq_ch, add_conv=add_conv)
        # 初始化查询卷积层
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化键卷积层
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化值卷积层
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化输出卷积层
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    # 定义前向传播函数，接受输入 x 和 zq
        def forward(self, x, zq):
            # 将输入 x 赋值给 h_
            h_ = x
            # 对 h_ 进行归一化处理，依据 zq
            h_ = self.norm(h_, zq)
            # 通过 q 层对 h_ 进行变换，得到查询 q
            q = self.q(h_)
            # 通过 k 层对 h_ 进行变换，得到键 k
            k = self.k(h_)
            # 通过 v 层对 h_ 进行变换，得到值 v
            v = self.v(h_)
    
            # 计算注意力
            # 获取 q 的形状，b 是批大小，c 是通道数，h 和 w 是高和宽
            b, c, h, w = q.shape
            # 将 q 重塑为 (b, c, h*w) 的形状
            q = q.reshape(b, c, h * w)
            # 变换 q 的维度顺序为 (b, hw, c)
            q = q.permute(0, 2, 1)  # b,hw,c
            # 将 k 重塑为 (b, c, h*w) 的形状
            k = k.reshape(b, c, h * w)  # b,c,hw
            # 计算 q 和 k 的乘积，得到注意力权重 w_
            w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            # 对权重进行缩放
            w_ = w_ * (int(c) ** (-0.5))
            # 对权重应用 softmax 函数，进行归一化
            w_ = torch.nn.functional.softmax(w_, dim=2)
    
            # 对值进行注意力计算
            # 将 v 重塑为 (b, c, h*w) 的形状
            v = v.reshape(b, c, h * w)
            # 变换 w_ 的维度顺序为 (b, hw, hw)
            w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
            # 计算 v 和 w_ 的乘积，得到最终的 h_
            h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            # 将 h_ 重塑为 (b, c, h, w) 的形状
            h_ = h_.reshape(b, c, h, w)
    
            # 对 h_ 进行投影处理
            h_ = self.proj_out(h_)
    
            # 返回输入 x 和 h_ 的和
            return x + h_
# 定义一个名为 MOVQDecoder 的类，继承自 nn.Module
class MOVQDecoder(nn.Module):
    # 初始化方法，用于创建 MOVQDecoder 的实例
    def __init__(
        # 接受关键字参数 ch，表示输入通道数
        *,
        ch,
        # 接受关键字参数 out_ch，表示输出通道数
        out_ch,
        # 接受关键字参数 ch_mult，表示通道倍增的因子，默认为 (1, 2, 4, 8)
        ch_mult=(1, 2, 4, 8),
        # 接受关键字参数 num_res_blocks，表示残差块的数量
        num_res_blocks,
        # 接受关键字参数 attn_resolutions，表示注意力分辨率
        attn_resolutions,
        # 接受关键字参数 dropout，表示 dropout 的比例，默认为 0.0
        dropout=0.0,
        # 接受关键字参数 resamp_with_conv，表示是否使用卷积进行重采样，默认为 True
        resamp_with_conv=True,
        # 接受关键字参数 in_channels，表示输入数据的通道数
        in_channels,
        # 接受关键字参数 resolution，表示输入数据的分辨率
        resolution,
        # 接受关键字参数 z_channels，表示潜在空间的通道数
        z_channels,
        # 接受关键字参数 give_pre_end，表示是否提供前置结束标志，默认为 False
        give_pre_end=False,
        # 接受关键字参数 zq_ch，表示潜在空间的量化通道数，默认为 None
        zq_ch=None,
        # 接受关键字参数 add_conv，表示是否添加额外的卷积层，默认为 False
        add_conv=False,
        # 接受其他未指定的关键字参数，使用 **ignorekwargs 收集
        **ignorekwargs,
    # 定义构造函数的结束部分
        ):
            # 调用父类构造函数
            super().__init__()
            # 存储输入参数 ch
            self.ch = ch
            # 初始化 temb_ch 为 0
            self.temb_ch = 0
            # 计算分辨率的数量
            self.num_resolutions = len(ch_mult)
            # 存储残差块的数量
            self.num_res_blocks = num_res_blocks
            # 存储分辨率
            self.resolution = resolution
            # 存储输入通道数
            self.in_channels = in_channels
            # 存储是否给出预处理结束标志
            self.give_pre_end = give_pre_end
    
            # 计算输入通道数乘法，块输入和当前分辨率
            in_ch_mult = (1,) + tuple(ch_mult)
            # 计算当前块的输入通道数
            block_in = ch * ch_mult[self.num_resolutions - 1]
            # 计算当前分辨率
            curr_res = resolution // 2 ** (self.num_resolutions - 1)
            # 定义 z_shape，表示 z 的形状
            self.z_shape = (1, z_channels, curr_res, curr_res)
            # 打印 z 的形状信息
            print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))
    
            # 定义从 z 到块输入的卷积层
            self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
    
            # 创建中间模块
            self.mid = nn.Module()
            # 定义中间块 1
            self.mid.block_1 = ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
                zq_ch=zq_ch,
                add_conv=add_conv,
            )
            # 定义中间注意力块 1
            self.mid.attn_1 = AttnBlock(block_in, zq_ch, add_conv=add_conv)
            # 定义中间块 2
            self.mid.block_2 = ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
                zq_ch=zq_ch,
                add_conv=add_conv,
            )
    
            # 初始化上采样模块列表
            self.up = nn.ModuleList()
            # 遍历分辨率，从高到低
            for i_level in reversed(range(self.num_resolutions)):
                # 初始化块和注意力模块列表
                block = nn.ModuleList()
                attn = nn.ModuleList()
                # 计算当前输出块的通道数
                block_out = ch * ch_mult[i_level]
                # 遍历每个残差块
                for i_block in range(self.num_res_blocks + 1):
                    # 添加残差块到块列表
                    block.append(
                        ResnetBlock(
                            in_channels=block_in,
                            out_channels=block_out,
                            temb_channels=self.temb_ch,
                            dropout=dropout,
                            zq_ch=zq_ch,
                            add_conv=add_conv,
                        )
                    )
                    # 更新块输入通道数
                    block_in = block_out
                    # 如果当前分辨率需要注意力模块，添加注意力模块
                    if curr_res in attn_resolutions:
                        attn.append(AttnBlock(block_in, zq_ch, add_conv=add_conv))
                # 创建上采样模块
                up = nn.Module()
                # 存储块和注意力模块
                up.block = block
                up.attn = attn
                # 如果不是最低分辨率，添加上采样层
                if i_level != 0:
                    up.upsample = Upsample(block_in, resamp_with_conv)
                    # 更新当前分辨率
                    curr_res = curr_res * 2
                # 将上采样模块插入到列表前面，确保顺序一致
                self.up.insert(0, up)  # prepend to get consistent order
    
            # 创建输出的归一化层
            self.norm_out = Normalize(block_in, zq_ch, add_conv=add_conv)
            # 定义输出的卷积层
            self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
    # 前向传播方法，接受输入 z 和条件 zq
    def forward(self, z, zq):
        # 断言 z 的形状与预期形状相符（已注释掉）
        # assert z.shape[1:] == self.z_shape[1:]
        # 保存输入 z 的形状
        self.last_z_shape = z.shape
    
        # 时间步嵌入初始化
        temb = None
    
        # 将 z 输入到卷积层
        h = self.conv_in(z)
    
        # 中间处理层
        h = self.mid.block_1(h, temb, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, temb, zq)
    
        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, zq)
                # 如果当前层有注意力模块，则应用注意力模块
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            # 如果不是最后一层，则进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)
    
        # 结束处理，条件性返回结果
        if self.give_pre_end:
            return h
    
        # 输出层归一化处理
        h = self.norm_out(h, zq)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 最后卷积层处理
        h = self.conv_out(h)
        return h
    
    # 带特征输出的前向传播方法，接受输入 z 和条件 zq
    def forward_with_features_output(self, z, zq):
        # 断言 z 的形状与预期形状相符（已注释掉）
        # assert z.shape[1:] == self.z_shape[1:]
        # 保存输入 z 的形状
        self.last_z_shape = z.shape
    
        # 时间步嵌入初始化
        temb = None
        output_features = {}
    
        # 将 z 输入到卷积层
        h = self.conv_in(z)
        # 保存卷积层输出特征
        output_features["conv_in"] = h
    
        # 中间处理层
        h = self.mid.block_1(h, temb, zq)
        # 保存中间块 1 的输出特征
        output_features["mid_block_1"] = h
        h = self.mid.attn_1(h, zq)
        # 保存中间注意力 1 的输出特征
        output_features["mid_attn_1"] = h
        h = self.mid.block_2(h, temb, zq)
        # 保存中间块 2 的输出特征
        output_features["mid_block_2"] = h
    
        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, zq)
                # 保存每个上采样块的输出特征
                output_features[f"up_{i_level}_block_{i_block}"] = h
                # 如果当前层有注意力模块，则应用注意力模块并保存特征
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
                    output_features[f"up_{i_level}_attn_{i_block}"] = h
            # 如果不是最后一层，则进行上采样并保存特征
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                output_features[f"up_{i_level}_upsample"] = h
    
        # 结束处理，条件性返回结果
        if self.give_pre_end:
            return h
    
        # 输出层归一化处理
        h = self.norm_out(h, zq)
        # 保存归一化后的特征
        output_features["norm_out"] = h
        # 应用非线性激活函数并保存特征
        h = nonlinearity(h)
        output_features["nonlinearity"] = h
        # 最后卷积层处理并保存特征
        h = self.conv_out(h)
        output_features["conv_out"] = h
    
        return h, output_features
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\quantize.py`

```py
# 导入 PyTorch 和其他必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange

# 定义一个改进版的向量量化器类，继承自 nn.Module
class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # 初始化方法，接受多个参数用于配置向量量化器
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True):
        # 调用父类构造函数
        super().__init__()
        # 设置向量数量
        self.n_e = n_e
        # 设置嵌入维度
        self.e_dim = e_dim
        # 设置 beta 值
        self.beta = beta
        # 设置是否使用旧版兼容性
        self.legacy = legacy

        # 创建一个嵌入层，尺寸为 (n_e, e_dim)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # 将嵌入权重初始化为均匀分布
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # 如果提供了 remap 参数
        self.remap = remap
        if self.remap is not None:
            # 从文件加载已使用的索引并注册为缓冲区
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            # 设置重新嵌入的大小
            self.re_embed = self.used.shape[0]
            # 设置未知索引，默认为 "random"
            self.unknown_index = unknown_index  # "random" 或 "extra" 或整数
            if self.unknown_index == "extra":
                # 如果未知索引是 "extra"，调整索引值
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            # 打印重映射信息
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            # 如果没有重映射，则重新嵌入大小等于 n_e
            self.re_embed = n_e

        # 设置是否使用合理的索引形状
        self.sane_index_shape = sane_index_shape

    # 将索引映射到已使用的索引
    def remap_to_used(self, inds):
        # 记录输入索引的形状
        ishape = inds.shape
        # 确保输入至少有两个维度
        assert len(ishape) > 1
        # 将索引重塑为 (批量大小, -1) 的形状
        inds = inds.reshape(ishape[0], -1)
        # 将已使用的索引移到当前设备
        used = self.used.to(inds)
        # 检查 inds 是否在 used 中匹配
        match = (inds[:, :, None] == used[None, None, ...]).long()
        # 找到匹配的最大值索引
        new = match.argmax(-1)
        # 查找未知索引
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            # 随机生成未知索引
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            # 用指定的未知索引填充未知位置
            new[unknown] = self.unknown_index
        # 返回重塑后的新索引
        return new.reshape(ishape)

    # 将已使用的索引映射回所有索引
    def unmap_to_all(self, inds):
        # 记录输入索引的形状
        ishape = inds.shape
        # 确保输入至少有两个维度
        assert len(ishape) > 1
        # 将索引重塑为 (批量大小, -1) 的形状
        inds = inds.reshape(ishape[0], -1)
        # 将已使用的索引移到当前设备
        used = self.used.to(inds)
        # 如果重新嵌入的大小大于已使用的索引数量，处理额外的标记
        if self.re_embed > self.used.shape[0]:  # extra token
            # 将超出范围的索引设为零
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        # 使用索引反向收集所有标记
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        # 返回重塑后的标记
        return back.reshape(ishape)
    # 前向传播函数，处理输入 z，选择温度和日志缩放参数
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        # 确保温度参数为 None 或 1.0，适用于 Gumbel 接口
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        # 确保日志缩放参数为 False，适用于 Gumbel 接口
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        # 确保返回日志参数为 False，适用于 Gumbel 接口
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # 将 z 变形为 (batch, height, width, channel) 并展平
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        # 将 z 展平为二维张量，形状为 (batch_size, embedding_dim)
        z_flattened = z.view(-1, self.e_dim)
        # 计算 z 到嵌入 e_j 的距离 (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            # 计算 z_flattened 的平方和，保留维度
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            # 加上嵌入权重的平方和
            + torch.sum(self.embedding.weight**2, dim=1)
            # 减去 2 * z_flattened 和嵌入权重的内积
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))
        )
        # 找到距离最近的编码索引
        min_encoding_indices = torch.argmin(d, dim=1)
        # 根据最小编码索引获取量化的 z，并调整为原始形状
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None  # 初始化困惑度为 None
        min_encodings = None  # 初始化最小编码为 None

        # 计算嵌入的损失
        if not self.legacy:
            # 计算损失，考虑 beta 和两个均方差项
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            # 计算损失，考虑 beta 和两个均方差项（顺序不同）
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # 保持梯度
        z_q = z + (z_q - z).detach()

        # 重新调整 z_q 的形状以匹配原始输入形状
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        if self.remap is not None:
            # 如果需要重映射，将最小编码索引展平并添加批次维度
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            # 使用重映射函数
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            # 再次展平
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            # 确保最小编码索引形状合理，重新调整形状
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        # 返回量化的 z、损失及其他信息
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    # 根据索引获取代码簿条目，返回形状为 (batch, height, width, channel)
    def get_codebook_entry(self, indices, shape):
        # shape 指定 (batch, height, width, channel)
        if self.remap is not None:
            # 如果需要重映射，展平索引并添加批次维度
            indices = indices.reshape(shape[0], -1)  # add batch axis
            # 使用反重映射函数
            indices = self.unmap_to_all(indices)
            # 再次展平
            indices = indices.reshape(-1)  # flatten again

        # 获取量化的潜在向量
        z_q = self.embedding(indices)

        if shape is not None:
            # 如果形状不为 None，重新调整 z_q 的形状
            z_q = z_q.view(shape)
            # 重新调整以匹配原始输入形状
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # 返回量化后的潜在向量
        return z_q
# 定义 GumbelQuantize 类，继承自 nn.Module
class GumbelQuantize(nn.Module):
    """
    归功于 @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (谢谢！)
    Gumbel Softmax 技巧量化器
    使用 Gumbel-Softmax 的分类重参数化，Jang 等人 2016
    https://arxiv.org/abs/1611.01144
    """

    # 初始化方法，定义类的属性
    def __init__(
        self,
        num_hiddens,  # 隐藏层的神经元数量
        embedding_dim,  # 嵌入维度
        n_embed,  # 嵌入的数量
        straight_through=True,  # 是否使用直通估计
        kl_weight=5e-4,  # KL 散度的权重
        temp_init=1.0,  # 初始温度
        use_vqinterface=True,  # 是否使用 VQ 接口
        remap=None,  # 重新映射参数
        unknown_index="random",  # 未知索引的处理方式
    ):
        super().__init__()  # 调用父类初始化

        self.embedding_dim = embedding_dim  # 设置嵌入维度
        self.n_embed = n_embed  # 设置嵌入数量

        self.straight_through = straight_through  # 保存直通估计的状态
        self.temperature = temp_init  # 设置温度
        self.kl_weight = kl_weight  # 设置 KL 权重

        # 定义卷积层，将隐藏层映射到嵌入空间
        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        # 定义嵌入层，将索引映射到嵌入向量
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface  # 保存 VQ 接口使用状态

        self.remap = remap  # 保存重新映射参数
        if self.remap is not None:  # 如果存在重新映射
            # 注册用于存储重新映射的张量
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]  # 重新映射的嵌入数量
            self.unknown_index = unknown_index  # 保存未知索引
            if self.unknown_index == "extra":  # 如果未知索引为“extra”
                self.unknown_index = self.re_embed  # 设置未知索引为重新映射数量
                self.re_embed = self.re_embed + 1  # 增加重新映射数量
            # 打印重新映射的信息
            print(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_embed  # 否则，重新映射数量等于嵌入数量

    # 将索引重新映射到已使用的索引
    def remap_to_used(self, inds):
        ishape = inds.shape  # 获取输入的形状
        assert len(ishape) > 1  # 确保输入维度大于1
        inds = inds.reshape(ishape[0], -1)  # 将索引重塑为二维形状
        used = self.used.to(inds)  # 将已使用的张量移动到索引的设备上
        match = (inds[:, :, None] == used[None, None, ...]).long()  # 计算匹配矩阵
        new = match.argmax(-1)  # 获取匹配的最大值索引
        unknown = match.sum(2) < 1  # 检查未知索引
        if self.unknown_index == "random":  # 如果未知索引为随机
            # 随机生成未知索引
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index  # 否则设置为未知索引
        return new.reshape(ishape)  # 返回重塑后的新索引

    # 将索引映射回所有索引
    def unmap_to_all(self, inds):
        ishape = inds.shape  # 获取输入的形状
        assert len(ishape) > 1  # 确保输入维度大于1
        inds = inds.reshape(ishape[0], -1)  # 将索引重塑为二维形状
        used = self.used.to(inds)  # 将已使用的张量移动到索引的设备上
        if self.re_embed > self.used.shape[0]:  # 如果有额外的标记
            inds[inds >= self.used.shape[0]] = 0  # 将超过范围的索引设置为零
        # 根据输入索引从已使用张量中收集值
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)  # 返回重塑后的结果
    # 定义前向传播方法，接受潜在变量 z、温度 temp 和返回 logits 的标志
    def forward(self, z, temp=None, return_logits=False):
        # 在评估模式下强制硬性为 True，因为必须进行量化。实际上，总是设为 True 似乎也可以
        hard = self.straight_through if self.training else True
        # 如果未提供温度，则使用类的温度属性
        temp = self.temperature if temp is None else temp
    
        # 将输入 z 投影到 logits 空间
        logits = self.proj(z)
        if self.remap is not None:
            # 仅继续使用的 logits
            full_zeros = torch.zeros_like(logits)  # 创建与 logits 同形状的零张量
            logits = logits[:, self.used, ...]  # 只保留使用的 logits
    
        # 使用 Gumbel-softmax 生成软 one-hot 编码
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # 返回到所有条目，但未使用的设置为零
            full_zeros[:, self.used, ...] = soft_one_hot  # 将使用的编码放入全零张量中
            soft_one_hot = full_zeros  # 更新为全零张量
    
        # 计算量化后的表示
        z_q = einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)
    
        # 加上对先验损失的 KL 散度
        qy = F.softmax(logits, dim=1)  # 对 logits 进行 softmax 操作
        # 计算 KL 散度
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
    
        # 找到软 one-hot 编码中最大值的索引
        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)  # 进行重映射
        if self.use_vqinterface:
            if return_logits:
                # 如果需要返回 logits，则返回量化后的表示、KL 散度和索引
                return z_q, diff, (None, None, ind), logits
            # 如果不需要返回 logits，则返回量化后的表示、KL 散度和索引
            return z_q, diff, (None, None, ind)
        # 返回量化后的表示、KL 散度和索引
        return z_q, diff, ind
    
    # 定义获取代码本条目的方法，接受索引和形状
    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape  # 解构形状信息
        assert b * h * w == indices.shape[0]  # 确保索引数量与形状一致
        indices = rearrange(indices, "(b h w) -> b h w", b=b, h=h, w=w)  # 重新排列索引形状
        if self.remap is not None:
            indices = self.unmap_to_all(indices)  # 如果需要，进行反映射
        # 创建 one-hot 编码并调整维度顺序
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        # 计算量化后的表示
        z_q = einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        return z_q  # 返回量化后的表示
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\vqvae_blocks.py`

```py
# pytorch_diffusion + derived encoder decoder
import math  # 导入数学库，用于数学计算
import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import numpy as np  # 导入 NumPy 库，用于数组处理

def get_timestep_embedding(timesteps, embedding_dim):
    """
    该函数实现了 Denoising Diffusion Probabilistic Models 中的嵌入构建
    从 Fairseq。
    构建正弦波嵌入。
    该实现与 tensor2tensor 中的实现匹配，但与 "Attention Is All You Need" 的
    第 3.5 节中的描述略有不同。
    """
    # 确保 timesteps 是一维数组
    assert len(timesteps.shape) == 1

    # 计算嵌入维度的一半
    half_dim = embedding_dim // 2
    # 计算用于正弦和余弦的频率
    emb = math.log(10000) / (half_dim - 1)
    # 生成正弦和余弦嵌入所需的频率向量
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将频率向量移动到与 timesteps 相同的设备
    emb = emb.to(device=timesteps.device)
    # 根据时间步生成最终的嵌入
    emb = timesteps.float()[:, None] * emb[None, :]
    # 将正弦和余弦嵌入合并
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度是奇数，进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回生成的嵌入
    return emb

def nonlinearity(x):
    # 定义 Swish 非线性激活函数
    return x * torch.sigmoid(x)  # 返回 Swish 激活值

def Normalize(in_channels):
    # 定义归一化层，使用 GroupNorm
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # 初始化 Upsample 类
        super().__init__()  # 调用父类构造函数
        self.with_conv = with_conv  # 根据参数决定是否使用卷积层
        if self.with_conv:
            # 如果使用卷积，则定义卷积层
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 定义前向传播方法
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")  # 进行上采样
        if self.with_conv:
            x = self.conv(x)  # 如果需要，经过卷积层处理
        return x  # 返回处理后的张量

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # 初始化 Downsample 类
        super().__init__()  # 调用父类构造函数
        self.with_conv = with_conv  # 根据参数决定是否使用卷积层
        if self.with_conv:
            # 定义卷积层，步幅为 2，填充为 0
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # 定义前向传播方法
        if self.with_conv:
            pad = (0, 1, 0, 1)  # 定义填充参数
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)  # 对输入张量进行填充
            x = self.conv(x)  # 经过卷积层处理
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)  # 进行平均池化
        return x  # 返回处理后的张量

class ResnetBlock(nn.Module):
    # 初始化方法，设置输入输出通道、卷积快捷连接、丢弃率和时间嵌入通道数
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        # 调用父类的初始化方法
        super().__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 如果未提供输出通道数，则设置为输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        # 设置输出通道数
        self.out_channels = out_channels
        # 设置是否使用卷积快捷连接的标志
        self.use_conv_shortcut = conv_shortcut

        # 初始化归一化层，处理输入通道
        self.norm1 = Normalize(in_channels)
        # 初始化第一层卷积，输入输出通道数相应，使用3x3卷积核
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果时间嵌入通道数大于0，则初始化时间嵌入投影层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化第二层归一化，处理输出通道
        self.norm2 = Normalize(out_channels)
        # 初始化丢弃层，设置丢弃率
        self.dropout = torch.nn.Dropout(dropout)
        # 初始化第二层卷积，输入输出通道数相应，使用3x3卷积核
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入通道数与输出通道数不相同
        if self.in_channels != self.out_channels:
            # 根据是否使用卷积快捷连接，初始化相应的卷积层
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法，处理输入数据和时间嵌入
    def forward(self, x, temb):
        # 将输入赋值给中间变量
        h = x
        # 对输入进行归一化处理
        h = self.norm1(h)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 经过第一层卷积
        h = self.conv1(h)

        # 如果时间嵌入不为 None，则将其添加到中间变量中
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # 对中间变量进行第二次归一化
        h = self.norm2(h)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 应用丢弃层
        h = self.dropout(h)
        # 经过第二层卷积
        h = self.conv2(h)

        # 如果输入通道数与输出通道数不相同
        if self.in_channels != self.out_channels:
            # 根据是否使用卷积快捷连接处理输入
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        # 返回输入和中间变量的和
        return x + h
# 定义注意力模块类，继承自 nn.Module
class AttnBlock(nn.Module):
    # 初始化方法，接受输入通道数
    def __init__(self, in_channels):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 初始化归一化层
        self.norm = Normalize(in_channels)
        # 初始化用于生成查询（q）的卷积层
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化用于生成键（k）的卷积层
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化用于生成值（v）的卷积层
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化输出投影的卷积层
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法
    def forward(self, x):
        # 将输入赋值给 h_
        h_ = x
        # 对输入进行归一化处理
        h_ = self.norm(h_)
        # 通过查询卷积层得到查询向量 q
        q = self.q(h_)
        # 通过键卷积层得到键向量 k
        k = self.k(h_)
        # 通过值卷积层得到值向量 v
        v = self.v(h_)

        # 计算注意力
        # 获取查询向量的形状参数
        b, c, h, w = q.shape
        # 将 q 进行重塑，改变形状为 (b, c, h*w)
        q = q.reshape(b, c, h * w)
        # 变换 q 的维度顺序，变为 (b, hw, c)
        q = q.permute(0, 2, 1)  # b,hw,c
        # 将 k 进行重塑，改变形状为 (b, c, hw)
        k = k.reshape(b, c, h * w)  # b,c,hw

        # # 原始版本，在 fp16 中会出现 nan
        # w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_ = w_ * (int(c)**(-0.5))
        # # 实现 c**-0.5 在 q 上
        # 将查询向量 q 乘以 c 的倒数平方根
        q = q * (int(c) ** (-0.5))
        # 计算权重 w_，使用批量矩阵相乘
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        # 对权重 w_ 进行 softmax 归一化
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # 根据值向量进行加权
        # 将值向量 v 进行重塑，改变形状为 (b, c, h*w)
        v = v.reshape(b, c, h * w)
        # 变换 w_ 的维度顺序，变为 (b, hw, hw) 
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # 通过批量矩阵相乘计算加权后的结果 h_
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # 将 h_ 进行重塑，改变形状回到 (b, c, h, w)
        h_ = h_.reshape(b, c, h, w)

        # 通过输出投影层生成最终结果
        h_ = self.proj_out(h_)

        # 返回输入和经过注意力模块处理后的结果相加
        return x + h_


# 定义编码器类，继承自 nn.Module
class Encoder(nn.Module):
    # 初始化方法，接收多个参数
    def __init__(
        self,
        *,
        ch,  # 通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍率
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力计算的分辨率
        dropout=0.0,  # dropout 概率
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入通道数
        resolution,  # 输入分辨率
        z_channels,  # z 维度通道数
        double_z=True,  # 是否使用双 z
        **ignore_kwargs,  # 其他忽略的关键字参数
    # 定义类的初始化方法
        ):
            # 调用父类的初始化方法
            super().__init__()
            # 初始化通道数
            self.ch = ch
            # 初始化时间嵌入通道数
            self.temb_ch = 0
            # 获取分辨率数量
            self.num_resolutions = len(ch_mult)
            # 获取残差块数量
            self.num_res_blocks = num_res_blocks
            # 设置分辨率
            self.resolution = resolution
            # 输入通道数
            self.in_channels = in_channels
    
            # downsampling
            # 初始化卷积层，输入通道为in_channels，输出通道为self.ch
            self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
    
            # 当前分辨率
            curr_res = resolution
            # 设置输入通道的倍数
            in_ch_mult = (1,) + tuple(ch_mult)
            # 初始化一个模块列表来存储下采样模块
            self.down = nn.ModuleList()
            # 遍历每个分辨率级别
            for i_level in range(self.num_resolutions):
                # 初始化块和注意力模块的模块列表
                block = nn.ModuleList()
                attn = nn.ModuleList()
                # 计算当前块的输入和输出通道数
                block_in = ch * in_ch_mult[i_level]
                block_out = ch * ch_mult[i_level]
                # 遍历每个残差块
                for i_block in range(self.num_res_blocks):
                    # 添加残差块到块列表
                    block.append(
                        ResnetBlock(
                            in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                        )
                    )
                    # 更新输入通道数为输出通道数
                    block_in = block_out
                    # 如果当前分辨率在注意力分辨率中，添加注意力块
                    if curr_res in attn_resolutions:
                        attn.append(AttnBlock(block_in))
                # 创建下采样模块
                down = nn.Module()
                down.block = block
                down.attn = attn
                # 如果不是最后一个分辨率级别，添加下采样层
                if i_level != self.num_resolutions - 1:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                    # 将当前分辨率减半
                    curr_res = curr_res // 2
                # 将下采样模块添加到列表中
                self.down.append(down)
    
            # middle
            # 创建中间模块
            self.mid = nn.Module()
            # 添加第一个残差块
            self.mid.block_1 = ResnetBlock(
                in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
            )
            # 添加中间注意力块
            self.mid.attn_1 = AttnBlock(block_in)
            # 添加第二个残差块
            self.mid.block_2 = ResnetBlock(
                in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
            )
    
            # end
            # 规范化输出
            self.norm_out = Normalize(block_in)
            # 初始化输出卷积层，输出通道为2 * z_channels或z_channels
            self.conv_out = torch.nn.Conv2d(
                block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
            )
    
        # 定义前向传播方法
        def forward(self, x):
            # 确保输入张量的宽和高等于设定的分辨率，若不匹配则抛出异常
            # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
    
            # timestep embedding
            # 初始化时间嵌入
            temb = None
    
            # downsampling
            # 通过输入数据进行卷积操作
            hs = [self.conv_in(x)]
            # 遍历每个分辨率级别
            for i_level in range(self.num_resolutions):
                # 遍历每个残差块
                for i_block in range(self.num_res_blocks):
                    # 通过残差块处理前一层的输出
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    # 如果存在注意力模块，则进行注意力处理
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    # 将当前层输出添加到列表中
                    hs.append(h)
                # 如果不是最后一个分辨率级别，则进行下采样
                if i_level != self.num_resolutions - 1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
    
            # middle
            # 获取最后一层的输出
            h = hs[-1]
            # 通过中间块进行处理
            h = self.mid.block_1(h, temb)
            # 进行注意力处理
            h = self.mid.attn_1(h)
            # 通过第二个中间块进行处理
            h = self.mid.block_2(h, temb)
    
            # end
            # 规范化处理
            h = self.norm_out(h)
            # 应用非线性激活函数
            h = nonlinearity(h)
            # 通过输出卷积层得到最终输出
            h = self.conv_out(h)
            # 返回处理后的输出
            return h
    # 定义一个前向传播函数，输出特征
    def forward_with_features_output(self, x):
        # 断言输入张量的高和宽等于预设的分辨率
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # 时间步嵌入初始化为空
        temb = None
        # 用于存储各层输出特征的字典
        output_features = {}

        # 下采样阶段，首先通过输入卷积层处理输入 x
        hs = [self.conv_in(x)]
        # 将输入卷积的输出保存到输出特征字典中
        output_features["conv_in"] = hs[-1]
        # 遍历每个分辨率层级
        for i_level in range(self.num_resolutions):
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks):
                # 通过当前层级的当前块进行处理
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # 将当前块的输出保存到输出特征字典中
                output_features["down{}_block{}".format(i_level, i_block)] = h
                # 如果当前层级有注意力机制，应用注意力机制
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                    # 将注意力机制的输出保存到输出特征字典中
                    output_features["down{}_attn{}".format(i_level, i_block)] = h
                # 将当前块的输出加入历史输出列表
                hs.append(h)
            # 如果不是最后一个分辨率层级，进行下采样
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                # 将下采样的输出保存到输出特征字典中
                output_features["down{}_downsample".format(i_level)] = hs[-1]

        # 中间层处理
        h = hs[-1]
        # 通过中间块1进行处理
        h = self.mid.block_1(h, temb)
        # 将中间块1的输出保存到输出特征字典中
        output_features["mid_block_1"] = h
        # 应用中间层的注意力机制
        h = self.mid.attn_1(h)
        # 将中间层注意力机制的输出保存到输出特征字典中
        output_features["mid_attn_1"] = h
        # 通过中间块2进行处理
        h = self.mid.block_2(h, temb)
        # 将中间块2的输出保存到输出特征字典中
        output_features["mid_block_2"] = h

        # 结束处理阶段
        h = self.norm_out(h)  # 进行归一化处理
        output_features["norm_out"] = h  # 保存归一化输出
        h = nonlinearity(h)  # 应用非线性激活函数
        output_features["nonlinearity"] = h  # 保存非线性输出
        h = self.conv_out(h)  # 通过输出卷积层处理
        output_features["conv_out"] = h  # 保存输出卷积的结果

        # 返回最终输出和特征字典
        return h, output_features
# 定义一个名为Decoder的类，继承自nn.Module
class Decoder(nn.Module):
    # 初始化函数，接收一系列参数
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍数
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力机制的分辨率
        dropout=0.0,  # dropout的比例
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入的通道数
        resolution,  # 分辨率
        z_channels,  # z的通道数
        give_pre_end=False,  # 是否给出预处理结果
        **ignorekwargs,  # 忽略的关键字参数
    ):
        super().__init__()
        # 将参数赋值给类的属性
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # 计算最低分辨率下的in_ch_mult、block_in和curr_res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # 打印z的形状
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # 将z转换为block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间部分
        self.mid = nn.Module()
        # 中间部分的第一个残差块
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        # 中间部分的注意力机制
        self.mid.attn_1 = AttnBlock(block_in)
        # 中间部分的第二个残差块
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # 上采样部分
        self.up = nn.ModuleList()
        # 从最高分辨率开始逆序遍历
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            # 根据残差块的数量创建残差块和注意力机制
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                # 如果当前分辨率在attn_resolutions中，则添加注意力机制
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            # 如果不是最低分辨率，则添加上采样层
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            # 将up插入到up列表的开头，以保持一致的顺序
            self.up.insert(0, up)

        # 结束部分
        # 归一化层
        self.norm_out = Normalize(block_in)
        # 输出卷积层
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
    # 定义前向传播函数，接收输入 z
    def forward(self, z):
        # 检查 z 的形状是否与期望的 z_shape 相匹配（被注释掉的断言）
        # assert z.shape[1:] == self.z_shape[1:]
        # 保存当前输入 z 的形状
        self.last_z_shape = z.shape

        # 初始化时间步嵌入为 None
        temb = None

        # 将输入 z 通过第一层卷积层进行处理，得到 block_in
        h = self.conv_in(z)

        # 中间处理阶段
        # 将 h 传入第一块中间模块，结合时间步嵌入 temb
        h = self.mid.block_1(h, temb)
        # 经过第一个注意力层
        h = self.mid.attn_1(h)
        # 将 h 传入第二块中间模块，结合时间步嵌入 temb
        h = self.mid.block_2(h, temb)

        # 上采样阶段
        # 从高到低的分辨率进行迭代
        for i_level in reversed(range(self.num_resolutions)):
            # 在每个分辨率下遍历所有的块
            for i_block in range(self.num_res_blocks + 1):
                # 将 h 传入当前上采样块进行处理，结合时间步嵌入 temb
                h = self.up[i_level].block[i_block](h, temb)
                # 如果当前块有注意力层，进行注意力处理
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            # 如果不是最后一层，进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 结束阶段
        # 如果给定了预结束标志，直接返回 h
        if self.give_pre_end:
            return h

        # 对 h 进行归一化处理
        h = self.norm_out(h)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过输出卷积层得到最终结果
        h = self.conv_out(h)
        # 返回处理后的结果 h
        return h
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\__init__.py`

```py

```

# `.\cogvideo-finetune\sat\sgm\modules\cp_enc_dec.py`

```py
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 导入 PyTorch 分布式计算库
import torch.distributed
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 从上级模块导入实用函数
from ..util import (
    get_context_parallel_group,  # 获取并行计算组
    get_context_parallel_rank,   # 获取当前并行计算的排名
    get_context_parallel_world_size,  # 获取并行计算的世界大小
)

# 设置使用的计算模式为 CP (Context Parallel)
_USE_CP = True

# 将输入转换为元组，确保长度为指定值
def cast_tuple(t, length=1):
    # 如果 t 已经是元组，直接返回；否则返回重复的元组
    return t if isinstance(t, tuple) else ((t,) * length)

# 检查 num 是否可以被 den 整除
def divisible_by(num, den):
    # 返回 num 除以 den 的余数是否为 0
    return (num % den) == 0

# 检查 n 是否为奇数
def is_odd(n):
    # 返回 n 除以 2 的结果是否为偶数的相反值
    return not divisible_by(n, 2)

# 检查值 v 是否存在（不为 None）
def exists(v):
    # 返回 v 是否不为 None
    return v is not None

# 将输入 t 转换为成对的元组
def pair(t):
    # 如果 t 是元组，直接返回；否则返回重复的元组
    return t if isinstance(t, tuple) else (t, t)

# 获取时间步嵌入
def get_timestep_embedding(timesteps, embedding_dim):
    """
    该实现与 Denoising Diffusion Probabilistic Models 中的实现相匹配：
    来自 Fairseq。
    构建正弦嵌入。
    与 tensor2tensor 中的实现相匹配，但与“Attention Is All You Need”第 3.5 节的描述略有不同。
    """
    # 确保 timesteps 是一维数组
    assert len(timesteps.shape) == 1

    # 计算一半维度
    half_dim = embedding_dim // 2
    # 计算正弦嵌入的缩放因子
    emb = math.log(10000) / (half_dim - 1)
    # 生成正弦嵌入
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将嵌入移动到与 timesteps 相同的设备
    emb = emb.to(device=timesteps.device)
    # 根据时间步生成嵌入
    emb = timesteps.float()[:, None] * emb[None, :]
    # 拼接正弦和余弦嵌入
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度为奇数，则填充一个零
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回嵌入结果
    return emb

# 应用非线性函数（swish）
def nonlinearity(x):
    # swish 激活函数
    return x * torch.sigmoid(x)

# 创建 LeakyReLU 激活函数
def leaky_relu(p=0.1):
    # 返回带有给定负斜率的 LeakyReLU 实例
    return nn.LeakyReLU(p)

# 在并行计算中拆分输入
def _split(input_, dim):
    # 获取并行计算的世界大小
    cp_world_size = get_context_parallel_world_size()

    # 如果并行计算的世界大小为 1，直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取当前并行计算的排名
    cp_rank = get_context_parallel_rank()

    # print('in _split, cp_rank:', cp_rank, 'input_size:', input_.shape)

    # 获取输入的第一帧，并保持连续性
    inpu_first_frame_ = input_.transpose(0, dim)[:1].transpose(0, dim).contiguous()
    # 更新输入，去掉第一帧
    input_ = input_.transpose(0, dim)[1:].transpose(0, dim).contiguous()
    # 计算每个并行计算的维度大小
    dim_size = input_.size()[dim] // cp_world_size

    # 按指定维度拆分输入
    input_list = torch.split(input_, dim_size, dim=dim)
    # 获取当前排名对应的输出
    output = input_list[cp_rank]

    # 如果当前排名为 0，拼接第一帧
    if cp_rank == 0:
        output = torch.cat([inpu_first_frame_, output], dim=dim)
    # 确保输出是连续的
    output = output.contiguous()

    # print('out _split, cp_rank:', cp_rank, 'output_size:', output.shape)

    # 返回拆分后的输出
    return output

# 在并行计算中收集输入
def _gather(input_, dim):
    # 获取并行计算的世界大小
    cp_world_size = get_context_parallel_world_size()

    # 如果并行计算的世界大小为 1，直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取并行计算组
    group = get_context_parallel_group()
    # 获取当前并行计算的排名
    cp_rank = get_context_parallel_rank()

    # print('in _gather, cp_rank:', cp_rank, 'input_size:', input_.shape)

    # 获取输入的第一帧，并保持连续性
    input_first_frame_ = input_.transpose(0, dim)[:1].transpose(0, dim).contiguous()
    # 如果当前排名为 0，更新输入，去掉第一帧
    if cp_rank == 0:
        input_ = input_.transpose(0, dim)[1:].transpose(0, dim).contiguous()

    # 创建一个包含空张量的列表，用于收集输入
    tensor_list = [torch.empty_like(torch.cat([input_first_frame_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]

    # 如果当前排名为 0，拼接第一帧到输入中
    if cp_rank == 0:
        input_ = torch.cat([input_first_frame_, input_], dim=dim)
    # 将输入张量存入指定的 tensor_list 中，索引由 cp_rank 确定
    tensor_list[cp_rank] = input_
    # 从所有进程中收集输入张量，并将结果存入 tensor_list 中，使用指定的分组
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # 将 tensor_list 中的所有张量在指定维度上连接成一个新的张量，并确保内存是连续的
    output = torch.cat(tensor_list, dim=dim).contiguous()

    # 调试输出当前进程的 cp_rank 和输出张量的尺寸（此行已被注释）
    # print('out _gather, cp_rank:', cp_rank, 'output_size:', output.shape)

    # 返回连接后的输出张量
    return output
# 定义函数 _conv_split，接收输入张量、维度和卷积核大小
def _conv_split(input_, dim, kernel_size):
    # 获取当前并行上下文的进程数量
    cp_world_size = get_context_parallel_world_size()

    # 如果并行上下文进程数为 1，则直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取当前进程在并行上下文中的排名
    cp_rank = get_context_parallel_rank()

    # 计算每个进程处理的维度大小
    dim_size = (input_.size()[dim] - kernel_size) // cp_world_size

    # 如果当前进程是 0 号进程，处理输入的前一部分
    if cp_rank == 0:
        output = input_.transpose(dim, 0)[: dim_size + kernel_size].transpose(dim, 0)
    else:
        # 其他进程处理输入的对应部分
        output = input_.transpose(dim, 0)[cp_rank * dim_size + 1 : (cp_rank + 1) * dim_size + kernel_size].transpose(
            dim, 0
        )
    # 确保输出张量在内存中的连续性
    output = output.contiguous()

    # 返回处理后的输出
    return output


# 定义函数 _conv_gather，接收输入张量、维度和卷积核大小
def _conv_gather(input_, dim, kernel_size):
    # 获取当前并行上下文的进程数量
    cp_world_size = get_context_parallel_world_size()

    # 如果并行上下文进程数为 1，则直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取当前的并行组
    group = get_context_parallel_group()
    # 获取当前进程在并行上下文中的排名
    cp_rank = get_context_parallel_rank()

    # 处理输入的第一部分卷积核
    input_first_kernel_ = input_.transpose(0, dim)[:kernel_size].transpose(0, dim).contiguous()
    # 如果当前进程是 0 号进程，处理剩余的输入
    if cp_rank == 0:
        input_ = input_.transpose(0, dim)[kernel_size:].transpose(0, dim).contiguous()
    else:
        # 其他进程处理相应的输入部分
        input_ = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim).contiguous()

    # 创建张量列表以存储各进程的输出
    tensor_list = [torch.empty_like(torch.cat([input_first_kernel_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]
    # 如果当前进程是 0 号进程，合并输入
    if cp_rank == 0:
        input_ = torch.cat([input_first_kernel_, input_], dim=dim)

    # 将当前进程的输入保存到张量列表中
    tensor_list[cp_rank] = input_
    # 收集所有进程的输入到张量列表
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # 合并张量列表中的所有输出，确保输出在内存中连续
    output = torch.cat(tensor_list, dim=dim).contiguous()

    # 返回处理后的输出
    return output
```

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\denoiser.py`

```py
# 从 typing 模块导入字典和联合类型的支持
from typing import Dict, Union

# 导入 PyTorch 库
import torch
import torch.nn as nn

# 从上级模块的 util 中导入 append_dims 和 instantiate_from_config 函数
from ...util import append_dims, instantiate_from_config


# 定义去噪器类，继承自 nn.Module
class Denoiser(nn.Module):
    # 初始化方法，接受权重配置和缩放配置
    def __init__(self, weighting_config, scaling_config):
        # 调用父类的初始化方法
        super().__init__()

        # 根据权重配置实例化权重处理对象
        self.weighting = instantiate_from_config(weighting_config)
        # 根据缩放配置实例化缩放处理对象
        self.scaling = instantiate_from_config(scaling_config)

    # 可能对 sigma 进行量化的函数，当前仅返回原值
    def possibly_quantize_sigma(self, sigma):
        return sigma

    # 可能对 c_noise 进行量化的函数，当前仅返回原值
    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    # 计算权重的函数，返回处理后的 sigma
    def w(self, sigma):
        return self.weighting(sigma)

    # 前向传播方法，定义了模型的计算过程
    def forward(
        self,
        network: nn.Module,  # 网络模块
        input: torch.Tensor,  # 输入张量
        sigma: torch.Tensor,  # sigma 张量
        cond: Dict,  # 条件字典
        **additional_model_inputs,  # 其他模型输入参数
    ) -> torch.Tensor:  # 返回一个张量
        # 量化 sigma
        sigma = self.possibly_quantize_sigma(sigma)
        # 获取 sigma 的形状
        sigma_shape = sigma.shape
        # 将 sigma 的维度扩展到输入的维度
        sigma = append_dims(sigma, input.ndim)
        # 通过缩放处理得到多个输出
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        # 量化 c_noise，并恢复原来的形状
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        # 返回通过网络处理的结果，加上输入与 c_skip 的加权和
        return network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out + input * c_skip


# 定义离散去噪器类，继承自 Denoiser
class DiscreteDenoiser(Denoiser):
    # 初始化方法，接受多个参数进行配置
    def __init__(
        self,
        weighting_config,  # 权重配置
        scaling_config,  # 缩放配置
        num_idx,  # 索引数量
        discretization_config,  # 离散化配置
        do_append_zero=False,  # 是否添加零
        quantize_c_noise=True,  # 是否量化 c_noise
        flip=True,  # 是否翻转
    ):
        # 调用父类的初始化方法
        super().__init__(weighting_config, scaling_config)
        # 根据离散化配置实例化 sigma 对象
        sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        # 保存 sigma 对象
        self.sigmas = sigmas
        # self.register_buffer("sigmas", sigmas)  # 可选，注册为缓冲区
        # 保存是否量化 c_noise 的配置
        self.quantize_c_noise = quantize_c_noise

    # 将 sigma 转换为索引的函数
    def sigma_to_idx(self, sigma):
        # 计算 sigma 与每个 sigma 值的距离
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        # 返回距离最小的索引
        return dists.abs().argmin(dim=0).view(sigma.shape)

    # 将索引转换为 sigma 的函数
    def idx_to_sigma(self, idx):
        # 根据索引返回对应的 sigma 值
        return self.sigmas.to(idx.device)[idx]

    # 可能对 sigma 进行量化的函数
    def possibly_quantize_sigma(self, sigma):
        # 通过索引转换函数进行量化
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    # 可能对 c_noise 进行量化的函数
    def possibly_quantize_c_noise(self, c_noise):
        # 如果配置为量化，则返回 c_noise 的索引
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            # 否则返回原值
            return c_noise
```

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\denoiser_scaling.py`

```py
# 从 abc 模块导入 ABC 类和 abstractmethod 装饰器
from abc import ABC, abstractmethod
# 从 typing 模块导入 Any 和 Tuple 类型
from typing import Any, Tuple

# 导入 torch 库
import torch


# 定义一个抽象基类 DenoiserScaling，继承自 ABC
class DenoiserScaling(ABC):
    # 定义一个抽象方法 __call__，接受一个 torch.Tensor 参数并返回一个四元组
    @abstractmethod
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


# 定义 EDMScaling 类
class EDMScaling:
    # 初始化方法，接受一个 sigma_data 参数，默认值为 0.5
    def __init__(self, sigma_data: float = 0.5):
        # 设置实例变量 sigma_data
        self.sigma_data = sigma_data

    # 定义 __call__ 方法，接受一个 torch.Tensor 参数并返回一个四元组
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 计算 c_skip，使用 sigma 和 sigma_data
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        # 计算 c_out，结合 sigma 和 sigma_data
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        # 计算 c_in，涉及 sigma 和 sigma_data
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        # 计算 c_noise，基于 sigma 的对数
        c_noise = 0.25 * sigma.log()
        # 返回计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 EpsScaling 类
class EpsScaling:
    # 定义 __call__ 方法，接受一个 torch.Tensor 参数并返回一个四元组
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 创建与 sigma 形状相同的全 1 张量，设备与 sigma 相同
        c_skip = torch.ones_like(sigma, device=sigma.device)
        # c_out 为 sigma 的负值
        c_out = -sigma
        # 计算 c_in，涉及 sigma 的平方
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        # 复制 sigma 作为 c_noise
        c_noise = sigma.clone()
        # 返回计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 VScaling 类
class VScaling:
    # 定义 __call__ 方法，接受一个 torch.Tensor 参数并返回一个四元组
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 计算 c_skip，涉及 sigma 的平方
        c_skip = 1.0 / (sigma**2 + 1.0)
        # 计算 c_out，结合 sigma 的负值
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        # 计算 c_in，涉及 sigma 的平方
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        # 复制 sigma 作为 c_noise
        c_noise = sigma.clone()
        # 返回计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 VScalingWithEDMcNoise 类，继承自 DenoiserScaling
class VScalingWithEDMcNoise(DenoiserScaling):
    # 定义 __call__ 方法，接受一个 torch.Tensor 参数并返回一个四元组
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 计算 c_skip，涉及 sigma 的平方
        c_skip = 1.0 / (sigma**2 + 1.0)
        # 计算 c_out，结合 sigma 的负值
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        # 计算 c_in，涉及 sigma 的平方
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        # 计算 c_noise，基于 sigma 的对数
        c_noise = 0.25 * sigma.log()
        # 返回计算结果
        return c_skip, c_out, c_in, c_noise


# 定义 VideoScaling 类，类似于 VScaling
class VideoScaling:  # similar to VScaling
    # 定义 __call__ 方法，接受一个 torch.Tensor 和可选的其他模型输入
    def __call__(
        self, alphas_cumprod_sqrt: torch.Tensor, **additional_model_inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 将 alphas_cumprod_sqrt 赋值给 c_skip
        c_skip = alphas_cumprod_sqrt
        # 计算 c_out，涉及 alphas_cumprod_sqrt
        c_out = -((1 - alphas_cumprod_sqrt**2) ** 0.5)
        # 创建与 alphas_cumprod_sqrt 形状相同的全 1 张量，设备相同
        c_in = torch.ones_like(alphas_cumprod_sqrt, device=alphas_cumprod_sqrt.device)
        # 复制 additional_model_inputs 中的 "idx" 作为 c_noise
        c_noise = additional_model_inputs["idx"].clone()
        # 返回计算结果
        return c_skip, c_out, c_in, c_noise
```

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\denoiser_weighting.py`

```py
# 导入 PyTorch 库
import torch


# 定义一个单位权重类
class UnitWeighting:
    # 定义可调用方法，接受 sigma 作为参数
    def __call__(self, sigma):
        # 返回与 sigma 形状相同的全 1 张量，设备与 sigma 相同
        return torch.ones_like(sigma, device=sigma.device)


# 定义 EDM 权重类
class EDMWeighting:
    # 初始化方法，设置 sigma_data 的默认值为 0.5
    def __init__(self, sigma_data=0.5):
        # 将传入的 sigma_data 保存为实例变量
        self.sigma_data = sigma_data

    # 定义可调用方法，接受 sigma 作为参数
    def __call__(self, sigma):
        # 返回计算的权重值
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


# 定义 V 权重类，继承自 EDMWeighting
class VWeighting(EDMWeighting):
    # 初始化方法
    def __init__(self):
        # 调用父类构造方法，设置 sigma_data 为 1.0
        super().__init__(sigma_data=1.0)


# 定义 Eps 权重类
class EpsWeighting:
    # 定义可调用方法，接受 sigma 作为参数
    def __call__(self, sigma):
        # 返回 sigma 的平方的倒数
        return sigma**-2.0
```

# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\discretizer.py`

```py
# 从 abc 模块导入抽象方法，用于定义抽象基类
from abc import abstractmethod
# 从 functools 模块导入 partial，用于创建部分应用的函数
from functools import partial

# 导入 numpy 库，通常用于数值计算
import numpy as np
# 导入 PyTorch 库，通常用于深度学习
import torch

# 从自定义模块中导入 make_beta_schedule 函数，用于生成 beta 调度
from ...modules.diffusionmodules.util import make_beta_schedule
# 从自定义模块中导入 append_zero 函数，用于处理 sigma 数组
from ...util import append_zero


# 定义一个函数，用于生成大致均匀间隔的步数
def generate_roughly_equally_spaced_steps(num_substeps: int, max_step: int) -> np.ndarray:
    # 使用 linspace 生成从 max_step-1 到 0 的均匀间隔的数组，反转并转换为整数
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


# 定义一个抽象基类 Discretization
class Discretization:
    # 定义一个可调用方法，用于处理输入参数
    def __call__(self, n, do_append_zero=True, device="cpu", flip=False, return_idx=False):
        # 根据 return_idx 的值获取 sigma 和索引
        if return_idx:
            sigmas, idx = self.get_sigmas(n, device=device, return_idx=return_idx)
        else:
            # 获取 sigma，不返回索引
            sigmas = self.get_sigmas(n, device=device, return_idx=return_idx)
        # 如果 do_append_zero 为真，则在 sigmas 末尾添加零
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        # 根据 flip 的值决定返回的 sigmas 和索引
        if return_idx:
            return sigmas if not flip else torch.flip(sigmas, (0,)), idx
        else:
            return sigmas if not flip else torch.flip(sigmas, (0,))

    # 定义一个抽象方法 get_sigmas，必须在子类中实现
    @abstractmethod
    def get_sigmas(self, n, device):
        pass


# 定义 EDMDiscretization 类，继承自 Discretization
class EDMDiscretization(Discretization):
    # 初始化方法，设置 sigma_min、sigma_max 和 rho 的默认值
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min  # 设置最小 sigma 值
        self.sigma_max = sigma_max  # 设置最大 sigma 值
        self.rho = rho  # 设置 rho 值

    # 实现 get_sigmas 方法
    def get_sigmas(self, n, device="cpu"):
        # 在指定设备上生成从 0 到 1 的均匀分布数组
        ramp = torch.linspace(0, 1, n, device=device)
        # 计算 sigma_min 和 sigma_max 的倒数
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        # 根据公式计算 sigmas
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas  # 返回计算出的 sigmas


# 定义 LegacyDDPMDiscretization 类，继承自 Discretization
class LegacyDDPMDiscretization(Discretization):
    # 初始化方法，设置线性开始、结束和时间步数的默认值
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
    ):
        super().__init__()  # 调用父类的初始化方法
        self.num_timesteps = num_timesteps  # 设置时间步数
        # 生成 beta 调度并计算对应的 alpha 值
        betas = make_beta_schedule("linear", num_timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1.0 - betas  # 计算 alpha 值
        # 计算 alpha 值的累积乘积
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # 创建一个部分应用的 torch.tensor 函数，指定数据类型
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

    # 实现 get_sigmas 方法
    def get_sigmas(self, n, device="cpu"):
        # 如果 n 小于时间步数，生成均匀间隔的时间步
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]  # 根据时间步获取对应的 alpha 值
        # 如果 n 等于时间步数，直接使用累积 alpha 值
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        # 如果 n 超过时间步数，抛出值错误
        else:
            raise ValueError

        # 创建一个部分应用的 torch.tensor 函数，指定数据类型和设备
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        # 计算 sigmas
        sigmas = to_torch((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return torch.flip(sigmas, (0,))  # 反转返回 sigmas


# 定义 ZeroSNRDDPMDiscretization 类，继承自 Discretization
class ZeroSNRDDPMDiscretization(Discretization):
    # 初始化方法，设置线性开始、结束、时间步数和其他参数的默认值
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.0,  # 噪声调度参数
        keep_start=False,  # 是否保持起始状态
        post_shift=False,  # 是否在后续进行偏移
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 如果保留起始值且没有后移，则对线性起始值进行缩放
        if keep_start and not post_shift:
            linear_start = linear_start / (shift_scale + (1 - shift_scale) * linear_start)
        # 设置时间步数
        self.num_timesteps = num_timesteps
        # 创建线性调度的 beta 值
        betas = make_beta_schedule("linear", num_timesteps, linear_start=linear_start, linear_end=linear_end)
        # 计算 alpha 值
        alphas = 1.0 - betas
        # 计算累积的 alpha 值
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # 将部分功能固定为转换为 torch 张量
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

        # SNR 偏移处理
        if not post_shift:
            # 调整累积的 alpha 值
            self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1 - shift_scale) * self.alphas_cumprod)

        # 存储后移状态
        self.post_shift = post_shift
        # 存储偏移缩放因子
        self.shift_scale = shift_scale

    def get_sigmas(self, n, device="cpu", return_idx=False):
        # 如果 n 小于时间步数，则生成等间隔的时间步
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            # 取出对应的累积 alpha 值
            alphas_cumprod = self.alphas_cumprod[timesteps]
        # 如果 n 等于时间步数，直接使用全部累积 alpha 值
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        # 如果 n 超过时间步数，则抛出错误
        else:
            raise ValueError

        # 将累积 alpha 值转换为 torch 张量
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod = to_torch(alphas_cumprod)
        # 计算累积 alpha 值的平方根
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        # 备份初始和最终的累积 alpha 值的平方根
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

        # 调整平方根的累积 alpha 值
        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

        # 如果开启后移，则进一步调整平方根的累积 alpha 值
        if self.post_shift:
            alphas_cumprod_sqrt = (
                alphas_cumprod_sqrt**2 / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)
            ) ** 0.5

        # 根据是否返回索引来决定返回值
        if return_idx:
            return torch.flip(alphas_cumprod_sqrt, (0,)), timesteps
        else:
            # 返回反转的平方根 alpha 值
            return torch.flip(alphas_cumprod_sqrt, (0,))  # sqrt(alpha_t): 0 -> 0.99
```