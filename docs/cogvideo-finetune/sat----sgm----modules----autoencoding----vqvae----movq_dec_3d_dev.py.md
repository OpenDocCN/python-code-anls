# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\movq_dec_3d_dev.py`

```py
# pytorch_diffusion + derived encoder decoder
import math  # 导入数学库，提供数学函数
import torch  # 导入 PyTorch 库，进行张量计算
import torch.nn as nn  # 导入 nn 模块，构建神经网络
import torch.nn.functional as F  # 导入功能性模块，提供常用操作
import numpy as np  # 导入 NumPy 库，进行数值计算

from beartype import beartype  # 从 beartype 导入 beartype，用于类型检查
from beartype.typing import Union, Tuple, Optional, List  # 导入类型提示
from einops import rearrange  # 从 einops 导入 rearrange，用于重排张量维度

from .movq_enc_3d import CausalConv3d, Upsample3D, DownSample3D  # 从本地模块导入 3D 卷积和上采样、下采样类


def cast_tuple(t, length=1):
    # 如果 t 不是元组，则将其转换为指定长度的元组
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    # 检查 num 是否可以被 den 整除
    return (num % den) == 0


def is_odd(n):
    # 检查 n 是否为奇数
    return not divisible_by(n, 2)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    这个函数构建正弦嵌入，与 Denoising Diffusion Probabilistic Models 的实现匹配：
    来源于 Fairseq。
    构建正弦嵌入。
    与 tensor2tensor 的实现匹配，但与 "Attention Is All You Need" 第 3.5 节中的描述略有不同。
    """
    # 确保 timesteps 是一维的
    assert len(timesteps.shape) == 1

    # 计算嵌入维度的一半
    half_dim = embedding_dim // 2
    # 计算嵌入的基础
    emb = math.log(10000) / (half_dim - 1)
    # 计算正弦嵌入的值
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将嵌入移动到 timesteps 的设备上
    emb = emb.to(device=timesteps.device)
    # 计算每个时间步的嵌入
    emb = timesteps.float()[:, None] * emb[None, :]
    # 将正弦和余弦值连接起来
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度是奇数，进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回最终的嵌入
    return emb


def nonlinearity(x):
    # 使用 Swish 激活函数
    return x * torch.sigmoid(x)


class SpatialNorm3D(nn.Module):
    # 定义一个 3D 空间归一化的类
    def __init__(
        self,
        f_channels,
        zq_channels,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=False,
        pad_mode="constant",
        **norm_layer_params,
    ):
        # 初始化函数，设置参数
        super().__init__()  # 调用父类的初始化函数
        # 创建归一化层
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        # 如果需要冻结归一化层的参数
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:  # 遍历归一化层的参数
                p.requires_grad = False  # 冻结参数不进行更新
        self.add_conv = add_conv  # 是否添加卷积层
        # 如果添加卷积层，创建 causal 卷积层
        if self.add_conv:
            # self.conv = nn.Conv3d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
            self.conv = CausalConv3d(zq_channels, zq_channels, kernel_size=3, pad_mode=pad_mode)
        # 创建一个 1x1 卷积层用于 y 和 b 通道
        # self.conv_y = nn.Conv3d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        # self.conv_b = nn.Conv3d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_y = CausalConv3d(zq_channels, f_channels, kernel_size=1, pad_mode=pad_mode)
        self.conv_b = CausalConv3d(zq_channels, f_channels, kernel_size=1, pad_mode=pad_mode)
    # 定义前向传播方法，接收输入 f 和 zq
    def forward(self, f, zq):
        # 如果 zq 的第三维大于 1，表示有多个通道
        if zq.shape[2] > 1:
            # 分割 f 为第一个通道和其余通道
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            # 获取第一个通道和其余通道的尺寸
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            # 分割 zq 为第一个通道和其余通道
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]
            # 对第一个通道进行最近邻插值调整尺寸
            zq_first = torch.nn.functional.interpolate(zq_first, size=f_first_size, mode="nearest")
            # 对其余通道进行最近邻插值调整尺寸
            zq_rest = torch.nn.functional.interpolate(zq_rest, size=f_rest_size, mode="nearest")
            # 将调整后的通道合并在一起
            zq = torch.cat([zq_first, zq_rest], dim=2)
        # 如果 zq 只有一个通道，直接调整其尺寸
        else:
            zq = torch.nn.functional.interpolate(zq, size=f.shape[-3:], mode="nearest")
        # 如果需要添加卷积层
        if self.add_conv:
            # 对 zq 进行卷积操作
            zq = self.conv(zq)
        # 对 f 应用归一化层
        norm_f = self.norm_layer(f)
        # 计算新的 f，结合卷积后的 zq
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        # 返回新的 f
        return new_f
# 定义一个 3D 归一化函数，接收输入通道、量化通道及是否添加卷积的标志
def Normalize3D(in_channels, zq_ch, add_conv):
    # 调用空间归一化 3D，传入相应参数
    return SpatialNorm3D(
        in_channels,
        zq_ch,
        norm_layer=nn.GroupNorm,  # 使用分组归一化层
        freeze_norm_layer=False,   # 不冻结归一化层
        add_conv=add_conv,         # 是否添加卷积
        num_groups=32,             # 设置组的数量
        eps=1e-6,                  # 设置小常数以防止除零
        affine=True,               # 使用仿射变换
    )


# 定义 3D ResNet 块，继承自 nn.Module
class ResnetBlock3D(nn.Module):
    # 初始化方法，接收多个参数配置
    def __init__(
        self,
        *,
        in_channels,               # 输入通道数
        out_channels=None,         # 输出通道数，可选
        conv_shortcut=False,       # 是否使用卷积短接
        dropout,                   # dropout 比率
        temb_channels=512,         # 时间嵌入通道数
        zq_ch=None,                # 量化通道
        add_conv=False,            # 是否添加卷积
        pad_mode="constant",       # 填充模式
    ):
        super().__init__()  # 调用父类构造函数
        self.in_channels = in_channels  # 设置输入通道数
        out_channels = in_channels if out_channels is None else out_channels  # 设置输出通道数
        self.out_channels = out_channels  # 保存输出通道数
        self.use_conv_shortcut = conv_shortcut  # 保存是否使用卷积短接的标志

        # 创建第一个归一化层
        self.norm1 = Normalize3D(in_channels, zq_ch, add_conv=add_conv)
        # self.conv1 = torch.nn.Conv3d(in_channels,
        #                              out_channels,
        #                              kernel_size=3,
        #                              stride=1,
        #                              padding=1)
        # 使用因果卷积创建第一个卷积层
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        if temb_channels > 0:  # 如果时间嵌入通道数大于零
            # 创建时间嵌入投影层
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 创建第二个归一化层
        self.norm2 = Normalize3D(out_channels, zq_ch, add_conv=add_conv)
        # 创建 dropout 层
        self.dropout = torch.nn.Dropout(dropout)
        # self.conv2 = torch.nn.Conv3d(out_channels,
        #                              out_channels,
        #                              kernel_size=3,
        #                              stride=1,
        #                              padding=1)
        # 使用因果卷积创建第二个卷积层
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        # 如果输入和输出通道数不一致
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:  # 如果使用卷积短接
                # self.conv_shortcut = torch.nn.Conv3d(in_channels,
                #                                      out_channels,
                #                                      kernel_size=3,
                #                                      stride=1,
                #                                      padding=1)
                # 使用因果卷积创建短接卷积层
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
            else:
                # 创建一个 1x1 的卷积层作为短接
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                # self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1, pad_mode=pad_mode)
    # 定义前向传播函数，接收输入张量 x、时间嵌入 temb 和 zq
        def forward(self, x, temb, zq):
            # 将输入赋值给 h
            h = x
            # 对 h 应用第一层归一化，使用 zq 作为参数
            h = self.norm1(h, zq)
            # 对 h 应用非线性激活函数
            h = nonlinearity(h)
            # 对 h 应用第一层卷积
            h = self.conv1(h)
    
            # 如果时间嵌入 temb 不为空
            if temb is not None:
                # 将时间嵌入经过非线性处理并进行投影后加到 h
                h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
    
            # 对 h 应用第二层归一化，使用 zq 作为参数
            h = self.norm2(h, zq)
            # 对 h 应用非线性激活函数
            h = nonlinearity(h)
            # 对 h 应用 dropout 操作
            h = self.dropout(h)
            # 对 h 应用第二层卷积
            h = self.conv2(h)
    
            # 如果输入通道数与输出通道数不相等
            if self.in_channels != self.out_channels:
                # 如果使用卷积短路
                if self.use_conv_shortcut:
                    # 对输入 x 应用卷积短路
                    x = self.conv_shortcut(x)
                else:
                    # 对输入 x 应用 NIN 短路
                    x = self.nin_shortcut(x)
    
            # 返回输入 x 与 h 的和
            return x + h
# 定义一个二维注意力块类，继承自 nn.Module
class AttnBlock2D(nn.Module):
    # 初始化方法，设置输入通道数、zq 通道数以及是否添加卷积
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 初始化 3D 归一化层
        self.norm = Normalize3D(in_channels, zq_ch, add_conv=add_conv)
        # 定义查询（Q）卷积层，kernel_size=1
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 定义键（K）卷积层，kernel_size=1
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 定义值（V）卷积层，kernel_size=1
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 定义输出投影卷积层，kernel_size=1
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法
    def forward(self, x, zq):
        # 保存输入数据
        h_ = x
        # 对输入数据进行归一化
        h_ = self.norm(h_, zq)

        # 获取时间步长
        t = h_.shape[2]
        # 重新排列张量维度，将时间步和批次合并
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")

        # 计算查询（Q）、键（K）和值（V）
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 计算注意力
        b, c, h, w = q.shape  # 获取批次大小、通道数、高度和宽度
        q = q.reshape(b, c, h * w)  # 重新排列查询张量维度
        q = q.permute(0, 2, 1)  # 交换维度顺序，变为 b, hw, c
        k = k.reshape(b, c, h * w)  # 重新排列键张量维度
        # 计算注意力权重矩阵，使用批量矩阵乘法
        w_ = torch.bmm(q, k)  # b, hw, hw，计算 q 和 k 的点积
        w_ = w_ * (int(c) ** (-0.5))  # 对权重进行缩放
        w_ = torch.nn.functional.softmax(w_, dim=2)  # 对权重进行 softmax 归一化

        # 注意力机制应用于值（V）
        v = v.reshape(b, c, h * w)  # 重新排列值张量维度
        w_ = w_.permute(0, 2, 1)  # 交换权重维度顺序
        # 计算加权值，得到注意力输出
        h_ = torch.bmm(v, w_)  # b, c, hw，计算 v 和权重的点积
        h_ = h_.reshape(b, c, h, w)  # 重新排列输出张量维度

        # 通过输出投影层进行变换
        h_ = self.proj_out(h_)

        # 恢复张量到原来的维度结构
        h_ = rearrange(h_, "(b t) c h w -> b c t h w", t=t)

        # 返回输入和输出的和
        return x + h_


# 定义一个三维 MOVQ 解码器类，继承自 nn.Module
class MOVQDecoder3D(nn.Module):
    # 初始化方法，设置多个参数，包括通道数、分辨率等
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        zq_ch=None,
        add_conv=False,
        pad_mode="first",
        temporal_compress_times=4,
        **ignorekwargs,
    # 定义前向传播函数，接受输入 z 和一个可选的 use_cp 参数
        def forward(self, z, use_cp=False):
            # 断言输入 z 的形状与预期的形状一致（此行被注释掉）
            # assert z.shape[1:] == self.z_shape[1:]
            # 保存输入 z 的形状，以备后用
            self.last_z_shape = z.shape
    
            # 初始化时间步嵌入变量
            temb = None
    
            # 获取 z 的时间步长度
            t = z.shape[2]
            # 将 z 赋值给 zq，作为输入块
    
            zq = z
            # 对输入 z 进行卷积操作，得到初步特征 h
            h = self.conv_in(z)
    
            # 中间层处理
            h = self.mid.block_1(h, temb, zq)  # 通过第一个中间块处理特征
            # h = self.mid.attn_1(h, zq)  # 注释掉的注意力机制处理
            h = self.mid.block_2(h, temb, zq)  # 通过第二个中间块处理特征
    
            # 上采样过程
            for i_level in reversed(range(self.num_resolutions)):  # 反向遍历分辨率层级
                for i_block in range(self.num_res_blocks + 1):  # 遍历每个块
                    h = self.up[i_level].block[i_block](h, temb, zq)  # 对当前特征进行块处理
                    if len(self.up[i_level].attn) > 0:  # 如果当前层有注意力机制
                        h = self.up[i_level].attn[i_block](h, zq)  # 进行注意力机制处理
                if i_level != 0:  # 如果不是最后一层
                    h = self.up[i_level].upsample(h)  # 进行上采样
    
            # 结束处理
            if self.give_pre_end:  # 如果需要返回中间结果
                return h
    
            h = self.norm_out(h, zq)  # 对 h 进行规范化处理
            h = nonlinearity(h)  # 应用非线性激活函数
            h = self.conv_out(h)  # 最后卷积操作，输出结果
            return h  # 返回最终结果
    
        # 获取最后一层的权重
        def get_last_layer(self):
            return self.conv_out.conv.weight  # 返回卷积层的权重
# 定义一个新的 3D 解码器类，继承自 nn.Module
class NewDecoder3D(nn.Module):
    # 初始化方法，接收多个参数以配置解码器
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道倍增因子
        num_res_blocks,  # 残差块数量
        attn_resolutions,  # 注意力分辨率
        dropout=0.0,  # dropout 比率
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入通道
        resolution,  # 输入分辨率
        z_channels,  # 噪声通道数
        give_pre_end=False,  # 是否给出预处理结束
        zq_ch=None,  # 可选的量化通道数
        add_conv=False,  # 是否添加额外卷积层
        pad_mode="first",  # 填充模式
        temporal_compress_times=4,  # 时间压缩倍数
        post_quant_conv=False,  # 是否使用后量化卷积
        **ignorekwargs,  # 其他忽略的关键字参数
    ):
        # 初始化父类 nn.Module
        super(NewDecoder3D, self).__init__()

    # 定义前向传播方法，接收输入 z
    def forward(self, z):
        # 断言检查 z 的形状是否与 z_shape 匹配
        # assert z.shape[1:] == self.z_shape[1:]
        # 记录 z 的最后形状
        self.last_z_shape = z.shape

        # 定义时间步嵌入，初始为 None
        temb = None

        # 获取 z 的时间步长
        t = z.shape[2]
        # z 赋值给 zq，作为量化输入

        zq = z
        # 如果定义了后量化卷积，则对 z 进行处理
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        # 对输入 z 进行初始卷积处理
        h = self.conv_in(z)

        # 中间层处理
        h = self.mid.block_1(h, temb, zq)  # 通过第一个中间块
        # h = self.mid.attn_1(h, zq)  # 可选的注意力机制
        h = self.mid.block_2(h, temb, zq)  # 通过第二个中间块

        # 上采样处理
        for i_level in reversed(range(self.num_resolutions)):  # 从高到低分辨率处理
            for i_block in range(self.num_res_blocks + 1):  # 遍历每个残差块
                h = self.up[i_level].block[i_block](h, temb, zq)  # 通过当前上采样块
                # 如果存在注意力模块，则应用
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            # 如果不是最后一层，则进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 结束处理
        if self.give_pre_end:  # 如果需要预处理结束，则返回 h
            return h

        # 对输出进行归一化处理
        h = self.norm_out(h, zq)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过最终卷积层得到输出
        h = self.conv_out(h)
        # 返回最终输出
        return h

    # 定义获取最后一层权重的方法
    def get_last_layer(self):
        # 返回最后卷积层的权重
        return self.conv_out.conv.weight
```