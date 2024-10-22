# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\movq_dec_3d.py`

```py
# pytorch_diffusion + derived encoder decoder
# 导入所需的库
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from .movq_enc_3d import CausalConv3d, Upsample3D, DownSample3D


# 将输入转换为元组，确保其长度为指定值
def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


# 检查一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0


# 判断一个数是否为奇数
def is_odd(n):
    return not divisible_by(n, 2)


# 获取时间步的嵌入向量
def get_timestep_embedding(timesteps, embedding_dim):
    """
    此函数与 Denoising Diffusion Probabilistic Models 中的实现匹配：
    来自 Fairseq。
    构建正弦嵌入向量。
    此实现与 tensor2tensor 中的实现匹配，但与 "Attention Is All You Need" 中第 3.5 节的描述略有不同。
    """
    # 确保时间步的维度为 1
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2  # 计算嵌入维度的一半
    emb = math.log(10000) / (half_dim - 1)  # 计算对数因子
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)  # 生成指数衰减的嵌入
    emb = emb.to(device=timesteps.device)  # 将嵌入移动到时间步的设备上
    emb = timesteps.float()[:, None] * emb[None, :]  # 扩展时间步并计算嵌入
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # 计算正弦和余弦值
    if embedding_dim % 2 == 1:  # 如果嵌入维度为奇数，则进行零填充
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb  # 返回嵌入


# 定义非线性激活函数，使用 Swish 激活函数
def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


# 定义三维空间归一化模块
class SpatialNorm3D(nn.Module):
    def __init__(
        self,
        f_channels,  # 特征通道数
        zq_channels,  # 嵌入通道数
        norm_layer=nn.GroupNorm,  # 归一化层类型
        freeze_norm_layer=False,  # 是否冻结归一化层的参数
        add_conv=False,  # 是否添加卷积层
        pad_mode="constant",  # 填充模式
        **norm_layer_params,  # 归一化层的其他参数
    ):
        super().__init__()  # 调用父类构造函数
        # 初始化归一化层
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if freeze_norm_layer:  # 如果需要冻结归一化层
            for p in self.norm_layer.parameters:  # 遍历所有参数
                p.requires_grad = False  # 不更新参数
        self.add_conv = add_conv  # 保存是否添加卷积层的标志
        if self.add_conv:  # 如果添加卷积层
            # 创建三维因果卷积层
            self.conv = CausalConv3d(zq_channels, zq_channels, kernel_size=3, pad_mode=pad_mode)
        # 创建用于特征和嵌入的卷积层
        self.conv_y = CausalConv3d(zq_channels, f_channels, kernel_size=1, pad_mode=pad_mode)
        self.conv_b = CausalConv3d(zq_channels, f_channels, kernel_size=1, pad_mode=pad_mode)

    # 前向传播函数
    def forward(self, f, zq):
        if zq.shape[2] > 1:  # 如果嵌入的时间步数大于1
            # 将特征拆分为第一时间步和剩余部分
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]  # 获取尺寸
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]  # 拆分嵌入
            # 使用最近邻插值调整 zq_first 的尺寸
            zq_first = torch.nn.functional.interpolate(zq_first, size=f_first_size, mode="nearest")
            # 使用最近邻插值调整 zq_rest 的尺寸
            zq_rest = torch.nn.functional.interpolate(zq_rest, size=f_rest_size, mode="nearest")
            zq = torch.cat([zq_first, zq_rest], dim=2)  # 合并嵌入
        else:  # 如果时间步数为1
            zq = torch.nn.functional.interpolate(zq, size=f.shape[-3:], mode="nearest")  # 调整 zq 尺寸
        if self.add_conv:  # 如果添加卷积层
            zq = self.conv(zq)  # 通过卷积层处理 zq
        norm_f = self.norm_layer(f)  # 对特征进行归一化
        # 计算新的特征值
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f  # 返回新的特征值


# 定义 Normalize3D 类的起始部分（未完成）
def Normalize3D(in_channels, zq_ch, add_conv):
    # 返回一个三维空间归一化层的实例
        return SpatialNorm3D(
            # 输入通道数
            in_channels,
            # 量化通道数
            zq_ch,
            # 归一化层使用的类型，这里使用的是分组归一化
            norm_layer=nn.GroupNorm,
            # 是否冻结归一化层的参数，这里设置为不冻结
            freeze_norm_layer=False,
            # 是否添加卷积层，使用传入的参数
            add_conv=add_conv,
            # 归一化的组数
            num_groups=32,
            # 防止除零的极小值
            eps=1e-6,
            # 是否使用仿射变换，这里设置为使用
            affine=True,
        )
# 定义一个三维残差块类，继承自 nn.Module
class ResnetBlock3D(nn.Module):
    # 初始化方法，接收多种参数以配置该块
    def __init__(
        self,
        *,
        in_channels,  # 输入通道数
        out_channels=None,  # 输出通道数（可选）
        conv_shortcut=False,  # 是否使用卷积快捷连接
        dropout,  # dropout 比例
        temb_channels=512,  # 时间嵌入通道数
        zq_ch=None,  # zq 相关通道数（可选）
        add_conv=False,  # 是否添加卷积层
        pad_mode="constant",  # 填充模式
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels
        # 确定输出通道数，若未指定则等于输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        # 保存输出通道数
        self.out_channels = out_channels
        # 保存是否使用卷积快捷连接的标志
        self.use_conv_shortcut = conv_shortcut

        # 初始化第一个归一化层
        self.norm1 = Normalize3D(in_channels, zq_ch, add_conv=add_conv)
        # 初始化第一个因果卷积层
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        # 若时间嵌入通道数大于0，初始化时间嵌入投影层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化第二个归一化层
        self.norm2 = Normalize3D(out_channels, zq_ch, add_conv=add_conv)
        # 初始化 dropout 层
        self.dropout = torch.nn.Dropout(dropout)
        # 初始化第二个因果卷积层
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        # 如果输入和输出通道数不相等
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷连接，则初始化对应的卷积层
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
            # 否则，初始化 1x1 卷积层作为快捷连接
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法
    def forward(self, x, temb, zq):
        # 将输入赋值给 h
        h = x
        # 对 h 进行归一化
        h = self.norm1(h, zq)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过第一个卷积层
        h = self.conv1(h)

        # 如果时间嵌入不为 None，则将其投影到 h 上
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        # 对 h 进行第二次归一化
        h = self.norm2(h, zq)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 应用 dropout
        h = self.dropout(h)
        # 通过第二个卷积层
        h = self.conv2(h)

        # 如果输入和输出通道数不相等
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷连接
            if self.use_conv_shortcut:
                # 将输入 x 通过卷积快捷连接
                x = self.conv_shortcut(x)
            # 否则使用 1x1 卷积
            else:
                x = self.nin_shortcut(x)

        # 返回输入和 h 的和
        return x + h


# 定义一个二维注意力块类，继承自 nn.Module
class AttnBlock2D(nn.Module):
    # 初始化方法，接收输入通道数、zq 通道数和是否添加卷积的标志
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 初始化归一化层
        self.norm = Normalize3D(in_channels, zq_ch, add_conv=add_conv)
        # 初始化查询、键、值卷积层，均为 1x1 卷积
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化输出卷积层
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    # 前向传播函数，接受输入 x 和查询 zq
    def forward(self, x, zq):
        # 将输入 x 赋值给 h_
        h_ = x
        # 对 h_ 进行归一化处理，使用查询 zq
        h_ = self.norm(h_, zq)
    
        # 获取 h_ 的时间步长 t
        t = h_.shape[2]
        # 重排 h_ 的维度，将时间步和批次维度合并
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")
    
        # 计算查询、键和值
        q = self.q(h_)  # 计算查询
        k = self.k(h_)  # 计算键
        v = self.v(h_)  # 计算值
    
        # 计算注意力
        b, c, h, w = q.shape  # 解包 q 的形状信息
        q = q.reshape(b, c, h * w)  # 将 q 重塑为 (b, c, hw)
        q = q.permute(0, 2, 1)  # 变换维度顺序为 (b, hw, c)
        k = k.reshape(b, c, h * w)  # 将 k 重塑为 (b, c, hw)
        # 计算 q 和 k 的批量矩阵乘法，得到注意力权重 w_
        w_ = torch.bmm(q, k)  # 计算注意力权重
        w_ = w_ * (int(c) ** (-0.5))  # 对权重进行缩放
        w_ = torch.nn.functional.softmax(w_, dim=2)  # 对最后一维进行 softmax
    
        # 根据注意力权重对值进行加权
        v = v.reshape(b, c, h * w)  # 将 v 重塑为 (b, c, hw)
        w_ = w_.permute(0, 2, 1)  # 变换维度顺序为 (b, hw, hw)
        # 计算加权和，得到输出特征 h_
        h_ = torch.bmm(v, w_)  # 计算输出
        h_ = h_.reshape(b, c, h, w)  # 将 h_ 重塑回 (b, c, h, w)
    
        # 对 h_ 进行投影
        h_ = self.proj_out(h_)
    
        # 将 h_ 的维度重排回原来的形状
        h_ = rearrange(h_, "(b t) c h w -> b c t h w", t=t)
    
        # 返回输入 x 和输出 h_ 的和
        return x + h_
# 定义一个名为 MOVQDecoder3D 的神经网络模块类，继承自 nn.Module
class MOVQDecoder3D(nn.Module):
    # 初始化方法，接受多个参数来配置解码器
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍增因子
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力机制的分辨率
        dropout=0.0,  # dropout 概率，用于防止过拟合
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入的通道数
        resolution,  # 输入图像的分辨率
        z_channels,  # 噪声通道数
        give_pre_end=False,  # 是否给出预处理结束标志
        zq_ch=None,  # 量化通道数
        add_conv=False,  # 是否添加卷积层
        pad_mode="first",  # 填充模式，默认为 'first'
        temporal_compress_times=4,  # 时间压缩的倍数
        **ignorekwargs,  # 其他未指定的关键字参数
    ):
        # 调用父类构造函数
        super().__init__()
        # 初始化通道数
        self.ch = ch
        # 初始化时间嵌入通道数为0
        self.temb_ch = 0
        # 获取分辨率数量
        self.num_resolutions = len(ch_mult)
        # 获取残差块数量
        self.num_res_blocks = num_res_blocks
        # 设置分辨率
        self.resolution = resolution
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置是否在前向传播后给出结束标志
        self.give_pre_end = give_pre_end

        # 计算 temporal_compress_times 的 log2 值
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        # 如果 zq_ch 为 None，则使用 z_channels
        if zq_ch is None:
            zq_ch = z_channels

        # 计算当前块的输入通道数
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # 计算当前分辨率
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # 设置 z 的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # 创建输入卷积层
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, pad_mode=pad_mode)

        # 创建中间模块
        self.mid = nn.Module()
        # 添加第一个残差块
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            pad_mode=pad_mode,
        )

        # 添加第二个残差块
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            pad_mode=pad_mode,
        )

        # 创建上采样模块
        self.up = nn.ModuleList()
        # 从最高分辨率开始遍历
        for i_level in reversed(range(self.num_resolutions)):
            # 创建块和注意力模块的容器
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # 计算当前块的输出通道数
            block_out = ch * ch_mult[i_level]
            # 添加指定数量的残差块
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        pad_mode=pad_mode,
                    )
                )
                # 更新输入通道数
                block_in = block_out
                # 如果当前分辨率在注意力分辨率列表中，添加注意力块
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock2D(block_in, zq_ch, add_conv=add_conv))
            # 创建上采样模块
            up = nn.Module()
            up.block = block
            up.attn = attn
            # 如果不是最底层，进行上采样配置
            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = Upsample3D(block_in, resamp_with_conv, compress_time=False)
                else:
                    up.upsample = Upsample3D(block_in, resamp_with_conv, compress_time=True)
                # 更新当前分辨率为原来的两倍
                curr_res = curr_res * 2
            # 将上采样模块插入到列表开头以保持顺序
            self.up.insert(0, up)  # prepend to get consistent order

        # 创建输出归一化层
        self.norm_out = Normalize3D(block_in, zq_ch, add_conv=add_conv)
        # 创建输出卷积层
        self.conv_out = CausalConv3d(block_in, out_ch, kernel_size=3, pad_mode=pad_mode)
    # 定义前向传播方法，接受输入 z 和可选参数 use_cp
    def forward(self, z, use_cp=False):
        # 保存输入 z 的形状以便后续使用
        self.last_z_shape = z.shape
    
        # 定义时间步嵌入变量，初始为 None
        temb = None
    
        # 获取 z 的时间步数（即第三维的大小）
        t = z.shape[2]
        # 将 z 赋值给 zq，用于后续计算
    
        zq = z
        # 通过输入卷积层处理 z
        h = self.conv_in(z)
    
        # 中间处理阶段
        # 使用第一个中间块处理 h，传入 temb 和 zq
        h = self.mid.block_1(h, temb, zq)
        # h = self.mid.attn_1(h, zq)  # 注释掉的注意力层
        # 使用第二个中间块处理 h，传入 temb 和 zq
        h = self.mid.block_2(h, temb, zq)
    
        # 上采样阶段
        # 反向遍历每个分辨率级别
        for i_level in reversed(range(self.num_resolutions)):
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks + 1):
                # 在当前上采样级别中处理 h，传入 temb 和 zq
                h = self.up[i_level].block[i_block](h, temb, zq)
                # 如果当前级别有注意力层，则对 h 进行注意力处理
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            # 如果当前不是最后一个级别，则进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)
    
        # 结束阶段
        # 如果给定了预结束标志，则直接返回 h
        if self.give_pre_end:
            return h
    
        # 对 h 进行归一化处理，传入 zq
        h = self.norm_out(h, zq)
        # 对 h 应用非线性激活函数
        h = nonlinearity(h)
        # 通过输出卷积层处理 h
        h = self.conv_out(h)
        # 返回最终的 h
        return h
    
    # 获取最后一层的卷积权重
    def get_last_layer(self):
        return self.conv_out.conv.weight
# 定义一个新的3D解码器类，继承自 nn.Module
class NewDecoder3D(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍增因子
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力分辨率
        dropout=0.0,  # dropout比率
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入通道数
        resolution,  # 输入的分辨率
        z_channels,  # 噪声通道数
        give_pre_end=False,  # 是否返回预结束的输出
        zq_ch=None,  # 可选的量化通道数
        add_conv=False,  # 是否添加额外的卷积层
        pad_mode="first",  # 填充模式
        temporal_compress_times=4,  # 时间压缩次数
        post_quant_conv=False,  # 是否使用量化后的卷积
        **ignorekwargs,  # 其他忽略的参数
    ):
    def forward(self, z):
        # 断言输入的形状与预期的 z_shape 一致（已注释）
        # self.last_z_shape = z.shape  # 保存最后的输入形状
        self.last_z_shape = z.shape

        # 时间步嵌入初始化为 None
        temb = None

        # 获取输入 z 的时间步长
        t = z.shape[2]
        # 将 z 赋值给 zq 以备后续使用
        zq = z
        # 如果存在后量化卷积，则对 z 进行处理
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        # 对 z 进行初始卷积
        h = self.conv_in(z)

        # 中间处理阶段
        h = self.mid.block_1(h, temb, zq)  # 通过第一个中间块处理 h
        # h = self.mid.attn_1(h, zq)  # (注释掉)可能的注意力机制处理
        h = self.mid.block_2(h, temb, zq)  # 通过第二个中间块处理 h

        # 上采样阶段
        for i_level in reversed(range(self.num_resolutions)):  # 反向遍历每个分辨率级别
            for i_block in range(self.num_res_blocks + 1):  # 遍历每个残差块
                h = self.up[i_level].block[i_block](h, temb, zq)  # 处理 h
                # 如果有注意力模块，则应用注意力
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            # 如果当前级别不是0，则进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 结束处理阶段
        if self.give_pre_end:  # 如果需要预结束输出，则返回 h
            return h

        # 通过归一化处理输出
        h = self.norm_out(h, zq)
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.conv_out(h)  # 通过最终卷积层处理 h
        return h  # 返回最终的输出

    # 获取最后一层的权重
    def get_last_layer(self):
        return self.conv_out.conv.weight  # 返回卷积层的权重
```