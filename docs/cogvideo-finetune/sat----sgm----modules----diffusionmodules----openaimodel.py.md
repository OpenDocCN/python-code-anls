# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\openaimodel.py`

```py
# 导入操作系统相关的模块
import os
# 导入数学相关的模块
import math
# 从 abc 模块导入抽象方法装饰器
from abc import abstractmethod
# 从 functools 导入偏函数工具
from functools import partial
# 导入类型提示相关的类型
from typing import Iterable, List, Optional, Tuple, Union

# 导入 numpy 库并简写为 np
import numpy as np
# 导入 PyTorch 库并简写为 th
import torch as th
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的功能模块
import torch.nn.functional as F
# 从 einops 导入重排工具
from einops import rearrange

# 从本地模块中导入 SpatialTransformer
from ...modules.attention import SpatialTransformer
# 从本地模块中导入多个实用函数
from ...modules.diffusionmodules.util import (
    avg_pool_nd,  # 导入平均池化函数
    checkpoint,  # 导入检查点函数
    conv_nd,  # 导入多维卷积函数
    linear,  # 导入线性层函数
    normalization,  # 导入归一化函数
    timestep_embedding,  # 导入时间步嵌入函数
    zero_module,  # 导入零模块函数
)
# 从本地模块中导入 LoRA 相关功能
from ...modules.diffusionmodules.lora import inject_trainable_lora_extended, update_lora_scale
# 从本地模块中导入空间视频变换器
from ...modules.video_attention import SpatialVideoTransformer
# 从本地工具模块导入默认值和存在性检查
from ...util import default, exists

# 虚函数替代
def convert_module_to_f16(x):
    pass

# 虚函数替代
def convert_module_to_f32(x):
    pass

# 定义 AttentionPool2d 类，继承自 nn.Module
class AttentionPool2d(nn.Module):
    """
    来源于 CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    # 初始化方法，定义各个参数
    def __init__(
        self,
        spacial_dim: int,  # 空间维度
        embed_dim: int,  # 嵌入维度
        num_heads_channels: int,  # 通道数对应的头数量
        output_dim: int = None,  # 输出维度，默认为 None
    ):
        # 调用父类初始化
        super().__init__()
        # 定义位置嵌入参数
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        # 定义 QKV 投影卷积层
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        # 定义输出卷积层
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        # 计算头的数量
        self.num_heads = embed_dim // num_heads_channels
        # 初始化注意力机制
        self.attention = QKVAttention(self.num_heads)

    # 前向传播方法
    def forward(self, x):
        # 获取输入的批量大小和通道数
        b, c, *_spatial = x.shape
        # 将输入张量重塑为 (b, c, -1) 的形状
        x = x.reshape(b, c, -1)  # NC(HW)
        # 计算输入的均值并连接
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        # 添加位置嵌入
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        # 通过 QKV 投影层处理输入
        x = self.qkv_proj(x)
        # 通过注意力机制处理输入
        x = self.attention(x)
        # 通过输出卷积层处理输入
        x = self.c_proj(x)
        # 返回处理后的第一个通道
        return x[:, :, 0]

# 定义时间步块类，继承自 nn.Module
class TimestepBlock(nn.Module):
    """
    任何模块，其中 forward() 方法将时间步嵌入作为第二个参数。
    """

    # 抽象前向传播方法
    @abstractmethod
    def forward(self, x, emb):
        """
        根据给定的时间步嵌入对 `x` 应用模块。
        """

# 定义时间步嵌入顺序模块类，继承自 nn.Sequential 和 TimestepBlock
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    一个顺序模块，将时间步嵌入作为额外输入传递给支持的子模块。
    """

    # 前向传播方法，支持多个输入参数
    def forward(
        self,
        x: th.Tensor,  # 输入张量
        emb: th.Tensor,  # 时间步嵌入张量
        context: Optional[th.Tensor] = None,  # 可选上下文张量
        image_only_indicator: Optional[th.Tensor] = None,  # 可选图像指示器
        time_context: Optional[int] = None,  # 可选时间上下文
        num_video_frames: Optional[int] = None,  # 可选视频帧数
    # 处理模型中的层，按不同类型调用相应的前向传播方法
        ):
            # 从指定路径导入 VideoResBlock 类
            from ...modules.diffusionmodules.video_model import VideoResBlock
    
            # 遍历当前模型的每一层
            for layer in self:
                # 将当前层赋值给模块变量
                module = layer
    
                # 检查模块是否为 TimestepBlock 且不是 VideoResBlock
                if isinstance(module, TimestepBlock) and not isinstance(module, VideoResBlock):
                    # 调用当前层的前向传播方法，传入 x 和 emb
                    x = layer(x, emb)
                # 检查模块是否为 VideoResBlock
                elif isinstance(module, VideoResBlock):
                    # 调用当前层的前向传播方法，传入 x、emb、num_video_frames 和 image_only_indicator
                    x = layer(x, emb, num_video_frames, image_only_indicator)
                # 检查模块是否为 SpatialVideoTransformer
                elif isinstance(module, SpatialVideoTransformer):
                    # 调用当前层的前向传播方法，传入多个上下文参数
                    x = layer(
                        x,
                        context,
                        time_context,
                        num_video_frames,
                        image_only_indicator,
                    )
                # 检查模块是否为 SpatialTransformer
                elif isinstance(module, SpatialTransformer):
                    # 调用当前层的前向传播方法，传入 x 和 context
                    x = layer(x, context)
                # 处理其他类型的模块
                else:
                    # 调用当前层的前向传播方法，传入 x
                    x = layer(x)
            # 返回最终的输出 x
            return x
# 定义一个上采样层，具有可选的卷积功能
class Upsample(nn.Module):
    """
    一个上采样层，带有可选的卷积。
    :param channels: 输入和输出的通道数。
    :param use_conv: 布尔值，决定是否应用卷积。
    :param dims: 决定信号是 1D、2D 还是 3D。如果是 3D，则在内部两个维度进行上采样。
    """

    # 初始化方法，设置上采样层的参数
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_up=False):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入和输出的通道数
        self.channels = channels
        # 如果没有指定输出通道，则默认为输入通道
        self.out_channels = out_channels or channels
        # 保存是否使用卷积的标志
        self.use_conv = use_conv
        # 保存信号的维度
        self.dims = dims
        # 保存是否进行三次上采样的标志
        self.third_up = third_up
        # 如果使用卷积，则创建相应的卷积层
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    # 前向传播方法
    def forward(self, x):
        # 断言输入的通道数与初始化时的通道数相同
        assert x.shape[1] == self.channels
        # 如果信号是 3D，则进行三维上采样
        if self.dims == 3:
            # 确定时间因子，如果不进行三次上采样，则因子为 1
            t_factor = 1 if not self.third_up else 2
            # 使用最近邻插值进行上采样
            x = F.interpolate(
                x,
                (t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2),  # 新的形状
                mode="nearest",  # 使用最近邻插值
            )
        else:
            # 对于其他维度，按比例进行上采样
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # 按比例进行上采样
        # 如果使用卷积，则应用卷积层
        if self.use_conv:
            x = self.conv(x)
        # 返回上采样后的结果
        return x


# 定义一个转置上采样层，执行 2x 上采样而不添加填充
class TransposedUpsample(nn.Module):
    "学习的 2x 上采样，无填充"

    # 初始化方法，设置转置上采样层的参数
    def __init__(self, channels, out_channels=None, ks=5):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入和输出的通道数
        self.channels = channels
        self.out_channels = out_channels or channels
        # 创建转置卷积层，进行 2x 上采样
        self.up = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=ks, stride=2)

    # 前向传播方法
    def forward(self, x):
        # 返回经过转置卷积层的结果
        return self.up(x)


# 定义一个下采样层，具有可选的卷积功能
class Downsample(nn.Module):
    """
    一个下采样层，带有可选的卷积。
    :param channels: 输入和输出的通道数。
    :param use_conv: 布尔值，决定是否应用卷积。
    :param dims: 决定信号是 1D、2D 还是 3D。如果是 3D，则在内部两个维度进行下采样。
    """
    # 初始化函数，设置下采样层的参数
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_down=False):
        # 调用父类的初始化函数
        super().__init__()
        # 设置下采样层的输入通道数
        self.channels = channels
        # 如果未指定输出通道数，则输出通道数与输入通道数相同
        self.out_channels = out_channels or channels
        # 记录是否使用卷积操作
        self.use_conv = use_conv
        # 设置下采样层的维度，默认为2
        self.dims = dims
        # 根据维度设置步长
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        # 如果使用卷积操作
        if use_conv:
            # 打印下采样层的维度信息
            print(f"Building a Downsample layer with {dims} dims.")
            print(
                f"  --> settings are: \n in-chn: {self.channels}, out-chn: {self.out_channels}, "
                f"kernel-size: 3, stride: {stride}, padding: {padding}"
            )
            # 如果维度为3，打印第三个轴（时间轴）的下采样信息
            if dims == 3:
                print(f"  --> Downsampling third axis (time): {third_down}")
            # 根据维度和参数设置卷积操作
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        # 如果不使用卷积操作
        else:
            # 断言输入通道数和输出通道数相同
            assert self.channels == self.out_channels
            # 设置操作为n维平均池化
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    # 前向传播函数
    def forward(self, x):
        # 断言输入数据的通道数与下采样层的输入通道数相同
        assert x.shape[1] == self.channels
        # 返回下采样操作后的结果
        return self.op(x)
# 定义一个残差块，可以选择是否改变通道数
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    # 初始化函数，接受多个参数
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化各个属性
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        # 如果 kernel_size 是可迭代对象，则计算 padding
        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        # 构建输入层
        self.in_layers = nn.Sequential(
            normalization(channels),  # 归一化
            nn.SiLU(),  # 激活函数
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),  # 卷积层
        )

        # 设置上采样或下采样
        self.updown = up or down

        # 如果是上采样，则初始化上采样层
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        # 如果是下采样，则初始化下采样层
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # 设置是否跳过时间步嵌入
        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        # 如果跳过时间步嵌入，则设置相关属性
        if self.skip_t_emb:
            print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        # 如果不跳过时间步嵌入，则初始化时间步嵌入层
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),  # 激活函数
                linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )

        # 构建输出层
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),  # 归一化
            nn.SiLU(),  # 激活函数
            nn.Dropout(p=dropout),  # 随机失活
            zero_module(  # 零填充
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        # 设置跳跃连接
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    # 前向传播函数
    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # 使用检查点函数进行前向传播
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
    # 定义前向传播函数，接收输入 x 和嵌入 emb
        def _forward(self, x, emb):
            # 检查是否需要进行上下采样
            if self.updown:
                # 分离输入层中的最后一层和其他层
                in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
                # 通过其他层处理输入 x
                h = in_rest(x)
                # 更新隐藏状态 h
                h = self.h_upd(h)
                # 更新输入 x
                x = self.x_upd(x)
                # 最后一层对更新后的隐藏状态 h 进行处理
                h = in_conv(h)
            else:
                # 如果不需要上下采样，直接通过所有输入层处理 x
                h = self.in_layers(x)
    
            # 检查是否需要跳过时间嵌入
            if self.skip_t_emb:
                # 创建与 h 相同形状的全零张量
                emb_out = th.zeros_like(h)
            else:
                # 通过嵌入层处理 emb，转换为与 h 相同的数据类型
                emb_out = self.emb_layers(emb).type(h.dtype)
            # 将 emb_out 的形状扩展到与 h 的形状相同
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            # 检查是否使用缩放和偏移规范化
            if self.use_scale_shift_norm:
                # 获取输出层中的规范化层和剩余层
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                # 将 emb_out 分割为缩放和偏移
                scale, shift = th.chunk(emb_out, 2, dim=1)
                # 进行规范化，应用缩放和偏移
                h = out_norm(h) * (1 + scale) + shift
                # 处理剩余层
                h = out_rest(h)
            else:
                # 检查是否需要交换时间嵌入的维度
                if self.exchange_temb_dims:
                    # 重新排列 emb_out 的维度
                    emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                # 将嵌入结果添加到隐藏状态 h 中
                h = h + emb_out
                # 通过输出层处理 h
                h = self.out_layers(h)
            # 返回跳过连接的结果与 h 的和
            return self.skip_connection(x) + h
# 定义一个注意力模块，允许空间位置之间相互关注
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    # 初始化方法，设置注意力模块的参数
    def __init__(
        self,
        channels,  # 输入通道数
        num_heads=1,  # 注意力头的数量，默认为1
        num_head_channels=-1,  # 每个注意力头的通道数，默认为-1表示自动计算
        use_checkpoint=False,  # 是否使用检查点，默认为False
        use_new_attention_order=False,  # 是否使用新注意力顺序，默认为False
    ):
        super().__init__()  # 调用父类的初始化方法
        self.channels = channels  # 保存输入通道数
        if num_head_channels == -1:  # 如果没有指定每个头的通道数
            self.num_heads = num_heads  # 使用指定的头数量
        else:
            # 检查输入通道数是否可以被每个头的通道数整除
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            # 计算头的数量
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint  # 保存检查点使用标志
        self.norm = normalization(channels)  # 创建归一化层
        self.qkv = conv_nd(1, channels, channels * 3, 1)  # 创建用于计算Q、K、V的卷积层
        if use_new_attention_order:  # 如果使用新注意力顺序
            # 在拆分头之前拆分QKV
            self.attention = QKVAttention(self.num_heads)  # 创建新的QKV注意力实例
        else:
            # 在拆分QKV之前拆分头
            self.attention = QKVAttentionLegacy(self.num_heads)  # 创建旧的QKV注意力实例

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))  # 创建输出投影层

    # 前向传播方法
    def forward(self, x, **kwargs):
        # TODO add crossframe attention and use mixed checkpoint
        # 使用检查点进行前向传播
        return checkpoint(
            self._forward, (x,), self.parameters(), True  # 将输入和模型参数传递给检查点
        )  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    # 实际的前向传播逻辑
    def _forward(self, x):
        b, c, *spatial = x.shape  # 解包输入形状为批次大小、通道数和空间维度
        x = x.reshape(b, c, -1)  # 将输入重新形状化为二维，方便计算
        qkv = self.qkv(self.norm(x))  # 通过归一化和卷积计算Q、K、V
        h = self.attention(qkv)  # 应用注意力机制
        h = self.proj_out(h)  # 通过投影层生成输出
        return (x + h).reshape(b, c, *spatial)  # 将结果形状恢复并返回

# 计算注意力操作中的浮点运算次数
def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape  # 解包输入形状为批次大小、通道数和空间维度
    num_spatial = int(np.prod(spatial))  # 计算空间维度的总数
    # 我们执行两个矩阵乘法，它们的运算次数相同
    # 第一个计算权重矩阵，第二个计算值向量的组合
    matmul_ops = 2 * b * (num_spatial**2) * c  # 计算矩阵乘法的操作数
    model.total_ops += th.DoubleTensor([matmul_ops])  # 将操作数累加到模型的总操作数中

# 定义旧版QKV注意力模块
class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    # 初始化方法，设置注意力头数量
    def __init__(self, n_heads):
        super().__init__()  # 调用父类的初始化方法
        self.n_heads = n_heads  # 保存注意力头数量
    # 定义前向传播方法，接受 QKV 张量作为输入
    def forward(self, qkv):
        # 文档字符串，描述函数用途及参数
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        # 获取输入张量的批次大小、宽度和长度
        bs, width, length = qkv.shape
        # 确保宽度可以被头数的三倍整除
        assert width % (3 * self.n_heads) == 0
        # 计算每个头的通道数
        ch = width // (3 * self.n_heads)
        # 将 QKV 张量重塑并分割为 Q、K 和 V
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        # 计算缩放因子
        scale = 1 / math.sqrt(math.sqrt(ch))
        # 计算注意力权重，使用爱因斯坦求和约定
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        # 对权重进行 softmax 归一化
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 根据权重和 V 计算加权和
        a = th.einsum("bts,bcs->bct", weight, v)
        # 将输出重塑为原始批次大小和通道数
        return a.reshape(bs, -1, length)
    
    # 定义静态方法以计算模型的浮点运算次数
    @staticmethod
    def count_flops(model, _x, y):
        # 调用辅助函数以计算注意力的浮点运算次数
        return count_flops_attn(model, _x, y)
# 定义一个 QKVAttention 类，继承自 nn.Module
class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    # 初始化方法，接收头数 n_heads 作为参数
    def __init__(self, n_heads):
        # 调用父类的初始化方法
        super().__init__()
        # 存储头数
        self.n_heads = n_heads

    # 前向传播方法，接受一个 qkv 张量
    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        # 解包 qkv 张量的形状
        bs, width, length = qkv.shape
        # 确保宽度可以被 3 * n_heads 整除
        assert width % (3 * self.n_heads) == 0
        # 计算每个头的通道数
        ch = width // (3 * self.n_heads)
        # 将 qkv 张量分成 q、k、v 三个部分
        q, k, v = qkv.chunk(3, dim=1)
        # 计算缩放因子
        scale = 1 / math.sqrt(math.sqrt(ch))
        # 计算权重，通过爱因斯坦求和表示法
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # 用 f16 进行计算更稳定，避免后续除法
        # 对权重进行 softmax 操作，归一化权重
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 计算注意力结果 a
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        # 返回结果，重新调整形状
        return a.reshape(bs, -1, length)

    # 静态方法，用于计算模型的浮点运算量
    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# 定义 Timestep 类，继承自 nn.Module
class Timestep(nn.Module):
    # 初始化方法，接收维度 dim 作为参数
    def __init__(self, dim):
        # 调用父类的初始化方法
        super().__init__()
        # 存储维度
        self.dim = dim

    # 前向传播方法，接受时间 t
    def forward(self, t):
        # 计算时间嵌入并返回
        return timestep_embedding(t, self.dim)


# 定义一个字典，将字符串映射到对应的数据类型
str_to_dtype = {"fp32": th.float32, "fp16": th.float16, "bf16": th.bfloat16}


# 定义 UNetModel 类，继承自 nn.Module
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    """
    # 参数 resblock_updown: 是否使用残差块进行上采样或下采样
    # 参数 use_new_attention_order: 是否使用不同的注意力模式以提高效率
    """

    # 初始化方法
    def __init__(
        self,
        # 输入通道数
        in_channels,
        # 模型通道数
        model_channels,
        # 输出通道数
        out_channels,
        # 残差块数量
        num_res_blocks,
        # 注意力分辨率
        attention_resolutions,
        # dropout 比例，默认为 0
        dropout=0,
        # 通道倍增参数，默认为 (1, 2, 4, 8)
        channel_mult=(1, 2, 4, 8),
        # 是否使用卷积重采样，默认为 True
        conv_resample=True,
        # 维度，默认为 2
        dims=2,
        # 类别数量，默认为 None
        num_classes=None,
        # 是否使用检查点，默认为 False
        use_checkpoint=False,
        # 是否使用半精度浮点数，默认为 False
        use_fp16=False,
        # 头部数量，默认为 -1
        num_heads=-1,
        # 每个头的通道数，默认为 -1
        num_head_channels=-1,
        # 上采样时的头部数量，默认为 -1
        num_heads_upsample=-1,
        # 是否使用缩放和位移归一化，默认为 False
        use_scale_shift_norm=False,
        # 是否在上采样和下采样中使用残差块，默认为 False
        resblock_updown=False,
        # 是否使用新的注意力顺序，默认为 False
        use_new_attention_order=False,
        # 是否使用空间变换器，支持自定义变换器
        use_spatial_transformer=False,  
        # 变换器深度，默认为 1
        transformer_depth=1,  
        # 上下文维度，默认为 None
        context_dim=None,  
        # 用于将离散 ID 预测到第一个阶段 VQ 模型的字典的自定义支持
        n_embed=None,  
        # 是否使用传统方式，默认为 True
        legacy=True,
        # 禁用自注意力，默认为 None
        disable_self_attentions=None,
        # 注意力块数量，默认为 None
        num_attention_blocks=None,
        # 禁用中间自注意力，默认为 False
        disable_middle_self_attn=False,
        # 在变换器中使用线性输入，默认为 False
        use_linear_in_transformer=False,
        # 空间变换器注意力类型，默认为 "softmax"
        spatial_transformer_attn_type="softmax",
        # ADM 输入通道数，默认为 None
        adm_in_channels=None,
        # 是否使用 Fairscale 检查点，默认为 False
        use_fairscale_checkpoint=False,
        # 是否将模型卸载到 CPU，默认为 False
        offload_to_cpu=False,
        # 中间变换器深度，默认为 None
        transformer_depth_middle=None,
        # 数据类型，默认为 "fp32"
        dtype="fp32",
        # 是否初始化 LoRA，默认为 False
        lora_init=False,
        # LoRA 等级，默认为 4
        lora_rank=4,
        # LoRA 缩放因子，默认为 1.0
        lora_scale=1.0,
        # LoRA 权重路径，默认为 None
        lora_weight_path=None,
    # 初始化 LoRA 方法
    def _init_lora(self, rank, scale, ckpt_dir=None):
        # 注入可训练的 LoRA 扩展
        inject_trainable_lora_extended(self, target_replace_module=None, rank=rank, scale=scale)

        # 如果提供了检查点目录
        if ckpt_dir is not None:
            # 打开最新文件，读取最新的检查点
            with open(os.path.join(ckpt_dir, "latest")) as latest_file:
                latest = latest_file.read().strip()
            # 构建检查点路径
            ckpt_path = os.path.join(ckpt_dir, latest, "mp_rank_00_model_states.pt")
            # 打印加载的 LoRA 路径
            print(f"loading lora from {ckpt_path}")
            # 从检查点加载模型状态字典
            sd = th.load(ckpt_path)["module"]
            # 处理模型状态字典，提取相关键
            sd = {
                key[len("model.diffusion_model") :]: sd[key] for key in sd if key.startswith("model.diffusion_model")
            }
            # 加载模型状态字典，严格模式设置为 False
            self.load_state_dict(sd, strict=False)

    # 更新 LoRA 缩放因子的函数
    def _update_scale(self, scale):
        # 调用更新缩放的方法
        update_lora_scale(self, scale)

    # 将模型的主干转换为浮点16的函数
    def convert_to_fp16(self):
        """
        将模型的主干转换为 float16。
        """
        # 应用转换函数到输入块
        self.input_blocks.apply(convert_module_to_f16)
        # 应用转换函数到中间块
        self.middle_block.apply(convert_module_to_f16)
        # 应用转换函数到输出块
        self.output_blocks.apply(convert_module_to_f16)

    # 将模型的主干转换为浮点32的函数
    def convert_to_fp32(self):
        """
        将模型的主干转换为 float32。
        """
        # 应用转换函数到输入块
        self.input_blocks.apply(convert_module_to_f32)
        # 应用转换函数到中间块
        self.middle_block.apply(convert_module_to_f32)
        # 应用转换函数到输出块
        self.output_blocks.apply(convert_module_to_f32)
    # 定义模型的前向传播方法，接受输入批次和可选参数
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # 方法说明：对输入批次应用模型，返回输出
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # 确保只有在类条件模型时才提供标签 y
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        # 初始化隐藏状态列表
        hs = []
        # 获取时间步嵌入，用于模型的输入
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        # 通过时间嵌入生成时间特征
        emb = self.time_embed(t_emb)
    
        # 如果模型是类条件的，添加标签嵌入
        if self.num_classes is not None:
            # 确保输入和标签批次大小一致
            assert y.shape[0] == x.shape[0]
            # 将标签嵌入添加到时间特征中
            emb = emb + self.label_emb(y)
    
        # h = x.type(self.dtype)  # 将输入转换为模型的数据类型（已注释）
        h = x  # 使用输入 x 作为初始隐藏状态
        # 遍历输入块，依次处理输入数据
        for module in self.input_blocks:
            h = module(h, emb, context)  # 通过模块处理隐藏状态
            hs.append(h)  # 将当前隐藏状态添加到列表中
        # 处理中间块，更新隐藏状态
        h = self.middle_block(h, emb, context)
        # 遍历输出块，生成最终输出
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)  # 将当前隐藏状态与之前的状态合并
            h = module(h, emb, context)  # 通过模块处理合并后的隐藏状态
        # 将最终隐藏状态转换为输入的原始数据类型
        h = h.type(x.dtype)
        # 检查是否支持预测代码本ID
        if self.predict_codebook_ids:
            # 如果不支持，抛出异常
            assert False, "not supported anymore. what the f*** are you doing?"
        else:
            # 返回最终输出
            return self.out(h)
# 定义一个名为 NoTimeUNetModel 的类，继承自 UNetModel
class NoTimeUNetModel(UNetModel):
    # 定义前向传播方法，接受输入 x、时间步 timesteps、上下文 context 和 y 以及其他参数
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # 将时间步初始化为与 timesteps 形状相同的零张量
        timesteps = th.zeros_like(timesteps)
        # 调用父类的前向传播方法并返回结果
        return super().forward(x, timesteps, context, y, **kwargs)


# 定义一个名为 EncoderUNetModel 的类，继承自 nn.Module
class EncoderUNetModel(nn.Module):
    """
    半个 UNet 模型，具有注意力和时间步嵌入功能。
    用法见 UNet。
    """

    # 定义初始化方法，接受多个参数设置模型的各项属性
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        *args,
        **kwargs,
    ):
        # 调用父类初始化方法
        super().__init__()

    # 定义将模型转换为 float16 方法
    def convert_to_fp16(self):
        """
        将模型的主体转换为 float16。
        """
        # 对输入块和中间块应用转换函数
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    # 定义将模型转换为 float32 方法
    def convert_to_fp32(self):
        """
        将模型的主体转换为 float32。
        """
        # 对输入块和中间块应用转换函数
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    # 定义前向传播方法，接受输入 x 和时间步 timesteps
    def forward(self, x, timesteps):
        """
        将模型应用于输入批次。
        :param x: 输入的 [N x C x ...] 张量。
        :param timesteps: 一维时间步批次。
        :return: [N x K] 输出张量。
        """
        # 将时间步嵌入生成的向量传递给模型
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []  # 初始化结果列表
        # h = x.type(self.dtype)  # 可选：将输入张量转换为指定的数据类型
        h = x  # 将输入赋值给 h
        for module in self.input_blocks:
            # 逐个模块处理输入，并应用嵌入
            h = module(h, emb)
            # 如果池化方式是空间池化，添加平均值到结果列表
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        # 处理中间块
        h = self.middle_block(h, emb)
        # 如果池化方式是空间池化，添加平均值到结果列表
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            # 将结果在最后一个维度拼接
            h = th.cat(results, axis=-1)
            # 返回输出
            return self.out(h)
        else:
            # 如果不是空间池化，将 h 转换为输入的数据类型
            h = h.type(x.dtype)
            # 返回输出
            return self.out(h)


# 主程序入口
if __name__ == "__main__":

    # 定义一个名为 Dummy 的类，继承自 nn.Module
    class Dummy(nn.Module):
        # 初始化方法，接受输入通道和模型通道的参数
        def __init__(self, in_channels=3, model_channels=64):
            # 调用父类初始化方法
            super().__init__()
            # 创建一个输入块的模块列表，包含时间步嵌入的卷积层
            self.input_blocks = nn.ModuleList(
                [TimestepEmbedSequential(conv_nd(2, in_channels, model_channels, 3, padding=1))]
            )

    # 创建 UNetModel 实例，并将其移至 GPU
    model = UNetModel(
        use_checkpoint=True,  # 使用检查点
        image_size=64,  # 图像大小
        in_channels=4,  # 输入通道数
        out_channels=4,  # 输出通道数
        model_channels=128,  # 模型通道数
        attention_resolutions=[4, 2],  # 注意力分辨率
        num_res_blocks=2,  # 残差块数量
        channel_mult=[1, 2, 4],  # 通道倍增系数
        num_head_channels=64,  # 头通道数
        use_spatial_transformer=False,  # 不使用空间变换器
        use_linear_in_transformer=True,  # 在变换器中使用线性输入
        transformer_depth=1,  # 变换器深度
        legacy=False,  # 不是旧版
    ).cuda()  # 移至 GPU
    # 创建一个形状为 (11, 4, 64, 64) 的随机张量，并将其移到 GPU 上
        x = th.randn(11, 4, 64, 64).cuda()
        # 生成一个包含 11 个随机整数的张量，范围在 0 到 10 之间，设备为 GPU
        t = th.randint(low=0, high=10, size=(11,), device="cuda")
        # 使用模型处理输入张量 x 和标签 t，返回结果
        o = model(x, t)
        # 打印完成信息
        print("done.")
```