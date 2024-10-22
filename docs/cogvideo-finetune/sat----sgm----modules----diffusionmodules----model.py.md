# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\model.py`

```py
# pytorch_diffusion + derived encoder decoder
# 导入所需的数学库和类型提示
import math
from typing import Any, Callable, Optional

# 导入 numpy 和 pytorch 相关的库
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version

# 尝试导入 xformers 模块及其操作，如果失败则设置标志为 False
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    # 如果未找到 xformers 模块，则打印提示信息
    print("no module 'xformers'. Processing without...")

# 从自定义模块导入线性注意力和内存高效交叉注意力
from ...modules.attention import LinearAttention, MemoryEfficientCrossAttention


def get_timestep_embedding(timesteps, embedding_dim):
    """
    该函数实现了 Denoising Diffusion Probabilistic Models 中的时间步嵌入
    来自 Fairseq。
    构建正弦嵌入。
    与 tensor2tensor 中的实现相匹配，但与 "Attention Is All You Need" 中第 3.5 节的描述略有不同。
    """
    # 确保 timesteps 具有一维形状
    assert len(timesteps.shape) == 1

    # 计算半维度并获取嵌入公式中的指数项
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将嵌入项移动到与 timesteps 相同的设备上
    emb = emb.to(device=timesteps.device)
    # 计算时间步长与嵌入的乘积
    emb = timesteps.float()[:, None] * emb[None, :]
    # 将正弦和余弦值拼接在一起
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度为奇数，则进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回最终的嵌入
    return emb


def nonlinearity(x):
    # 实现 swish 激活函数
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    # 返回分组归一化层
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # 初始化 Upsample 类，设置输入通道数和是否使用卷积
        super().__init__()
        self.with_conv = with_conv
        # 如果使用卷积，初始化卷积层
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 对输入张量进行上采样
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 如果使用卷积，则应用卷积层
        if self.with_conv:
            x = self.conv(x)
        # 返回处理后的张量
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # 初始化 Downsample 类，设置输入通道数和是否使用卷积
        super().__init__()
        self.with_conv = with_conv
        # 如果使用卷积，初始化卷积层
        if self.with_conv:
            # PyTorch 卷积层不支持不对称填充，需手动处理
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # 如果使用卷积，进行填充并应用卷积层
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # 否则使用平均池化进行下采样
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        # 返回处理后的张量
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels
        # 如果没有指定输出通道数，则使用输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        # 保存输出通道数
        self.out_channels = out_channels
        # 保存是否使用卷积快捷方式的标志
        self.use_conv_shortcut = conv_shortcut

        # 初始化归一化层，输入通道数作为参数
        self.norm1 = Normalize(in_channels)
        # 初始化第一个卷积层，输入通道数、输出通道数，卷积核大小为3，步幅为1，填充为1
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果temb_channels大于0，初始化temb的线性变换层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化第二个归一化层，输出通道数作为参数
        self.norm2 = Normalize(out_channels)
        # 初始化丢弃层，使用给定的丢弃率
        self.dropout = torch.nn.Dropout(dropout)
        # 初始化第二个卷积层，输入和输出通道数一致，卷积核大小为3，步幅为1，填充为1
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入和输出通道数不一致
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷方式，则初始化卷积快捷方式层
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            # 否则初始化1x1的线性变换快捷方式层
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        # 将输入赋值给临时变量h
        h = x
        # 对h进行归一化处理
        h = self.norm1(h)
        # 对h应用非线性激活函数
        h = nonlinearity(h)
        # 对h进行第一个卷积操作
        h = self.conv1(h)

        # 如果temb不为None，则对h进行temb的投影
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # 对h进行第二个归一化处理
        h = self.norm2(h)
        # 对h应用非线性激活函数
        h = nonlinearity(h)
        # 对h进行丢弃操作
        h = self.dropout(h)
        # 对h进行第二个卷积操作
        h = self.conv2(h)

        # 如果输入和输出通道数不一致
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷方式，则对输入x进行卷积处理
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            # 否则对输入x进行1x1的线性变换
            else:
                x = self.nin_shortcut(x)

        # 返回输入x与h的相加结果
        return x + h
class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


# 创建一个名为LinAttnBlock的类，继承自LinearAttention类
class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    # 构造函数，接受输入通道数作为参数
    def __init__(self, in_channels):
        # 调用父类的构造函数，传入输入通道数、头数为1、头通道数为输入通道数
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)



class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


# 创建一个名为AttnBlock的类，继承自nn.Module类
class AttnBlock(nn.Module):
    # 构造函数，接受输入通道数作为参数
    def __init__(self, in_channels):
        # 调用父类的构造函数
        super().__init__()
        # 将输入通道数保存到实例变量中
        self.in_channels = in_channels

        # 创建一个Normalize对象，输入通道数为in_channels
        self.norm = Normalize(in_channels)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)



    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        # compute attention

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)


    # 定义attention方法，接受一个torch.Tensor类型的参数h_，返回一个torch.Tensor类型的结果
    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        # 对输入张量进行归一化处理
        h_ = self.norm(h_)
        # 使用卷积层q、k、v对输入张量进行卷积操作
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 获取q的形状信息
        b, c, h, w = q.shape
        # 将q、k、v的形状进行变换
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        # 使用缩放点积注意力机制计算注意力
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        # 返回计算得到的注意力张量，并将其形状进行变换
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)



    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


    # 定义forward方法，接受输入张量x和其他关键字参数，返回一个torch.Tensor类型的结果
    def forward(self, x, **kwargs):
        # 将输入张量赋值给局部变量h_
        h_ = x
        # 使用attention方法对h_进行处理
        h_ = self.attention(h_)
        # 使用卷积层proj_out对h_进行卷积操作
        h_ = self.proj_out(h_)
        # 返回输入张量x与处理后的张量h_的和
        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attention_op: Optional[Any] = None


# 创建一个名为MemoryEfficientAttnBlock的类，继承自nn.Module类
class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    # 构造函数，接受输入通道数作为参数
    def __init__(self, in_channels):
        # 调用父类的构造函数
        super().__init__()
        # 将输入通道数保存到实例变量中
        self.in_channels = in_channels

        # 创建一个Normalize对象，输入通道数为in_channels
        self.norm = Normalize(in_channels)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个卷积层，输入通道数为in_channels，输出通道数为in_channels，卷积核大小为1x1，步长为1，填充为0
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建一个Optional类型的实例变量attention_op，初始值为None
        self.attention_op: Optional[Any] = None
    # 定义注意力机制方法，输入为张量 h_，输出为处理后的张量
    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        # 对输入张量进行规范化处理
        h_ = self.norm(h_)
        # 通过线性变换生成查询向量 q
        q = self.q(h_)
        # 通过线性变换生成键向量 k
        k = self.k(h_)
        # 通过线性变换生成值向量 v
        v = self.v(h_)

        # 计算注意力机制
        # 获取查询向量的形状，B: 批次大小, C: 通道数, H: 高度, W: 宽度
        B, C, H, W = q.shape
        # 将 q, k, v 的形状从 (B, C, H, W) 转换为 (B, H*W, C)
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        # 扩展 q, k, v 的维度并重塑形状，以便进行注意力计算
        q, k, v = map(
            lambda t: t.unsqueeze(3)  # 在第 3 维增加一个维度
            .reshape(B, t.shape[1], 1, C)  # 重塑为 (B, 经过处理的长度, 1, C)
            .permute(0, 2, 1, 3)  # 调整维度顺序为 (B, 1, 经过处理的长度, C)
            .reshape(B * 1, t.shape[1], C)  # 重塑为 (B, 经过处理的长度, C)
            .contiguous(),  # 确保内存连续性
            (q, k, v),
        )
        # 使用高效的注意力计算方法，返回注意力输出
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # 对输出进行重塑，调整形状以匹配最终的输出要求
        out = out.unsqueeze(0).reshape(B, 1, out.shape[1], C).permute(0, 2, 1, 3).reshape(B, out.shape[1], C)
        # 将输出形状调整回 (B, C, H, W)
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)

    # 定义前向传播方法，输入为张量 x，接受额外参数 kwargs
    def forward(self, x, **kwargs):
        # 将输入赋值给 h_
        h_ = x
        # 通过注意力机制处理 h_
        h_ = self.attention(h_)
        # 将处理结果通过输出线性层
        h_ = self.proj_out(h_)
        # 返回输入 x 与处理结果 h_ 的和
        return x + h_
# 定义一个内存高效的交叉注意力包装类，继承自 MemoryEfficientCrossAttention
class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    # 前向传播方法，接收输入和上下文
    def forward(self, x, context=None, mask=None, **unused_kwargs):
        # 获取输入的批量大小、通道数、高度和宽度
        b, c, h, w = x.shape
        # 重排输入张量，合并高度和宽度维度
        x = rearrange(x, "b c h w -> b (h w) c")
        # 调用父类的前向方法进行注意力计算
        out = super().forward(x, context=context, mask=mask)
        # 重排输出张量，恢复为原始形状
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        # 返回输入与输出的和
        return x + out


# 创建注意力模块的函数，根据给定的类型和参数
def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    # 确保提供的注意力类型是有效的
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
        "memory-efficient-cross-attn",
        "linear",
        "none",
    ], f"attn_type {attn_type} unknown"
    # 检查 PyTorch 版本以决定注意力实现
    if version.parse(torch.__version__) < version.parse("2.0.0") and attn_type != "none":
        # 确保 xformers 可用以支持较早的版本
        assert XFORMERS_IS_AVAILABLE, (
            f"We do not support vanilla attention in {torch.__version__} anymore, "
            f"as it is too expensive. Please install xformers via e.g. 'pip install xformers==0.0.16'"
        )
        # 更新注意力类型为 vanilla-xformers
        attn_type = "vanilla-xformers"
    # 打印当前创建的注意力类型及输入通道数
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    # 根据类型创建对应的注意力模块
    if attn_type == "vanilla":
        # 确保没有额外的参数
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        # 打印构建内存高效注意力块的信息
        print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "memory-efficient-cross-attn":
        # 设置查询维度为输入通道数
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        # 返回身份映射
        return nn.Identity(in_channels)
    else:
        # 返回线性注意力块
        return LinAttnBlock(in_channels)


# 定义模型类，继承自 nn.Module
class Model(nn.Module):
    # 初始化模型参数
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
        use_timestep=True,
        use_linear_attn=False,
        attn_type="vanilla",
    # 定义前向传播函数，接受输入数据 x、时间 t 和上下文 context
        def forward(self, x, t=None, context=None):
            # 检查输入张量的空间维度是否与预设分辨率一致（注释掉）
            # assert x.shape[2] == x.shape[3] == self.resolution
            # 如果上下文不为 None，沿通道轴拼接输入和上下文
            if context is not None:
                x = torch.cat((x, context), dim=1)
            # 如果使用时间步，进行时间步嵌入
            if self.use_timestep:
                # 确保时间 t 不为 None
                assert t is not None
                # 获取时间步嵌入
                temb = get_timestep_embedding(t, self.ch)
                # 通过第一层全连接进行变换
                temb = self.temb.dense[0](temb)
                # 应用非线性激活函数
                temb = nonlinearity(temb)
                # 通过第二层全连接进行变换
                temb = self.temb.dense[1](temb)
            else:
                # 如果不使用时间步，则将 temb 设为 None
                temb = None
    
            # 进行下采样
            hs = [self.conv_in(x)]  # 初始化列表，包含输入经过卷积的结果
            for i_level in range(self.num_resolutions):  # 遍历分辨率级别
                for i_block in range(self.num_res_blocks):  # 遍历每个分辨率的残差块
                    # 将上一层的输出和时间嵌入输入到当前块
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    # 如果当前层有注意力机制，应用注意力机制
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    # 将当前层的输出添加到列表中
                    hs.append(h)
                # 如果不是最后一个分辨率，进行下采样
                if i_level != self.num_resolutions - 1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
    
            # 处理中间层
            h = hs[-1]  # 获取最后一层的输出
            h = self.mid.block_1(h, temb)  # 经过中间第一块处理
            h = self.mid.attn_1(h)  # 应用中间第一块的注意力机制
            h = self.mid.block_2(h, temb)  # 经过中间第二块处理
    
            # 进行上采样
            for i_level in reversed(range(self.num_resolutions)):  # 反向遍历分辨率级别
                for i_block in range(self.num_res_blocks + 1):  # 遍历每个分辨率的残差块加一
                    # 将当前层输出与上一层的输出拼接，并传入时间嵌入
                    h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                    # 如果当前层有注意力机制，应用注意力机制
                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
                # 如果不是第一个分辨率，进行上采样
                if i_level != 0:
                    h = self.up[i_level].upsample(h)
    
            # 结束处理
            h = self.norm_out(h)  # 归一化输出
            h = nonlinearity(h)  # 应用非线性激活函数
            h = self.conv_out(h)  # 最终卷积层处理
            return h  # 返回最终结果
    
        # 定义获取最后一层权重的函数
        def get_last_layer(self):
            return self.conv_out.weight  # 返回最后一层卷积的权重
# 定义一个编码器类，继承自 nn.Module
class Encoder(nn.Module):
    # 初始化函数，接收多个参数用于配置编码器
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数倍增因子
        num_res_blocks,  # 残差块数量
        attn_resolutions,  # 注意力机制应用的分辨率
        dropout=0.0,  # dropout 概率
        resamp_with_conv=True,  # 是否使用卷积进行下采样
        in_channels,  # 输入图像的通道数
        resolution,  # 输入图像的分辨率
        z_channels,  # 潜在空间的通道数
        double_z=True,  # 是否双倍潜在通道
        use_linear_attn=False,  # 是否使用线性注意力机制
        attn_type="vanilla",  # 注意力机制的类型
        **ignore_kwargs,  # 其他未使用的参数
    ):
        # 调用父类构造函数
        super().__init__()
        # 如果使用线性注意力，设置注意力类型为线性
        if use_linear_attn:
            attn_type = "linear"
        # 设置类的属性
        self.ch = ch
        self.temb_ch = 0  # 时间嵌入通道数
        self.num_resolutions = len(ch_mult)  # 分辨率数量
        self.num_res_blocks = num_res_blocks  # 残差块数量
        self.resolution = resolution  # 输入分辨率
        self.in_channels = in_channels  # 输入通道数

        # 下采样层
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution  # 当前分辨率
        in_ch_mult = (1,) + tuple(ch_mult)  # 输入通道倍增因子
        self.in_ch_mult = in_ch_mult  # 保存输入通道倍增因子
        self.down = nn.ModuleList()  # 下采样模块列表
        # 遍历每个分辨率级别
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()  # 残差块列表
            attn = nn.ModuleList()  # 注意力模块列表
            block_in = ch * in_ch_mult[i_level]  # 当前块的输入通道数
            block_out = ch * ch_mult[i_level]  # 当前块的输出通道数
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks):
                # 添加残差块
                block.append(
                    ResnetBlock(
                        in_channels=block_in,  # 输入通道数
                        out_channels=block_out,  # 输出通道数
                        temb_channels=self.temb_ch,  # 时间嵌入通道数
                        dropout=dropout,  # dropout 概率
                    )
                )
                block_in = block_out  # 更新输入通道数
                # 如果当前分辨率在注意力分辨率中，添加注意力模块
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()  # 创建一个下采样模块
            down.block = block  # 设置残差块
            down.attn = attn  # 设置注意力模块
            # 如果不是最后一个分辨率级别，添加下采样层
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2  # 更新当前分辨率
            self.down.append(down)  # 将下采样模块添加到列表中

        # 中间层
        self.mid = nn.Module()  # 创建中间层模块
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,  # 输入通道数
            out_channels=block_in,  # 输出通道数
            temb_channels=self.temb_ch,  # 时间嵌入通道数
            dropout=dropout,  # dropout 概率
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)  # 添加第一个注意力模块
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,  # 输入通道数
            out_channels=block_in,  # 输出通道数
            temb_channels=self.temb_ch,  # 时间嵌入通道数
            dropout=dropout,  # dropout 概率
        )

        # 结束层
        self.norm_out = Normalize(block_in)  # 归一化层
        self.conv_out = torch.nn.Conv2d(
            block_in,  # 输入通道数
            2 * z_channels if double_z else z_channels,  # 输出通道数，依据是否双倍潜在通道选择
            kernel_size=3,  # 卷积核大小
            stride=1,  # 步幅
            padding=1,  # 填充
        )
    # 定义前向传播方法，接受输入 x
        def forward(self, x):
            # 初始化时间步嵌入变量为 None
            temb = None
    
            # 初始化下采样，创建输入的卷积特征
            hs = [self.conv_in(x)]
            # 遍历每个分辨率级别
            for i_level in range(self.num_resolutions):
                # 遍历每个残差块
                for i_block in range(self.num_res_blocks):
                    # 使用下采样块处理最后一层特征和时间步嵌入
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    # 如果当前层有注意力机制，则应用它
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    # 将处理后的特征添加到特征列表中
                    hs.append(h)
                # 如果不是最后一个分辨率级别，则进行下采样
                if i_level != self.num_resolutions - 1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
    
            # 处理中间层特征
            h = hs[-1]
            # 应用中间块1处理
            h = self.mid.block_1(h, temb)
            # 应用中间层的注意力机制1
            h = self.mid.attn_1(h)
            # 应用中间块2处理
            h = self.mid.block_2(h, temb)
    
            # 结束层处理
            h = self.norm_out(h)  # 应用输出归一化
            h = nonlinearity(h)   # 应用非线性激活函数
            h = self.conv_out(h)  # 应用输出卷积
            return h  # 返回最终输出
# 定义一个解码器类，继承自 nn.Module
class Decoder(nn.Module):
    # 初始化方法，接收多个参数用于解码器的配置
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍增因子
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力机制应用的分辨率
        dropout=0.0,  # 丢弃率，默认为0
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入图像的通道数
        resolution,  # 输入图像的分辨率
        z_channels,  # 潜在空间的通道数
        give_pre_end=False,  # 是否提供前置结束标志
        tanh_out=False,  # 是否使用 Tanh 激活函数作为输出
        use_linear_attn=False,  # 是否使用线性注意力机制
        attn_type="vanilla",  # 注意力机制的类型，默认为普通注意力
        **ignorekwargs,  # 其他未明确列出的参数
    ):
        # 初始化父类
        super().__init__()
        # 如果使用线性注意力，设置注意力类型为"linear"
        if use_linear_attn:
            attn_type = "linear"
        # 设置通道数
        self.ch = ch
        # 设置时间嵌入通道数为0
        self.temb_ch = 0
        # 计算分辨率的数量
        self.num_resolutions = len(ch_mult)
        # 设置残差块的数量
        self.num_res_blocks = num_res_blocks
        # 设置输入分辨率
        self.resolution = resolution
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置是否在最后给出前置层
        self.give_pre_end = give_pre_end
        # 设置是否使用tanh作为输出激活函数
        self.tanh_out = tanh_out

        # 计算输入通道数乘数，块输入和当前分辨率
        in_ch_mult = (1,) + tuple(ch_mult)
        # 计算当前层的输入通道数
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # 计算当前分辨率
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # 设置z的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # 打印z的形状和维度
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # 创建注意力类
        make_attn_cls = self._make_attn()
        # 创建残差块类
        make_resblock_cls = self._make_resblock()
        # 创建卷积类
        make_conv_cls = self._make_conv()
        # z到块输入的卷积层
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间模块
        self.mid = nn.Module()
        # 创建第一个残差块
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        # 创建第一个注意力层
        self.mid.attn_1 = make_attn_cls(block_in, attn_type=attn_type)
        # 创建第二个残差块
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # 上采样模块
        self.up = nn.ModuleList()
        # 反向遍历分辨率
        for i_level in reversed(range(self.num_resolutions)):
            # 创建块和注意力列表
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # 计算块输出通道数
            block_out = ch * ch_mult[i_level]
            # 创建残差块
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    make_resblock_cls(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                # 更新块输入通道数
                block_in = block_out
                # 如果当前分辨率需要注意力层，添加注意力层
                if curr_res in attn_resolutions:
                    attn.append(make_attn_cls(block_in, attn_type=attn_type))
            # 创建上采样模块
            up = nn.Module()
            up.block = block
            up.attn = attn
            # 如果不是最后一层，添加上采样层
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                # 更新当前分辨率
                curr_res = curr_res * 2
            # 将上采样模块添加到列表的开头
            self.up.insert(0, up)  # prepend to get consistent order

        # 结束模块
        # 创建归一化层
        self.norm_out = Normalize(block_in)
        # 创建输出卷积层
        self.conv_out = make_conv_cls(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    # 创建注意力函数
    def _make_attn(self) -> Callable:
        return make_attn

    # 创建残差块函数
    def _make_resblock(self) -> Callable:
        return ResnetBlock

    # 创建卷积函数
    def _make_conv(self) -> Callable:
        return torch.nn.Conv2d
    # 获取模型最后一层的权重
        def get_last_layer(self, **kwargs):
            # 返回卷积输出层的权重
            return self.conv_out.weight
    
        # 前向传播函数
        def forward(self, z, **kwargs):
            # 检查输入 z 的形状是否与预期相同（注释掉的断言）
            # assert z.shape[1:] == self.z_shape[1:]
            # 保存输入 z 的形状
            self.last_z_shape = z.shape
    
            # 初始化时间步嵌入
            temb = None
    
            # 将输入 z 传入卷积输入层
            h = self.conv_in(z)
    
            # 中间层处理
            h = self.mid.block_1(h, temb, **kwargs)  # 通过第一个中间块处理
            h = self.mid.attn_1(h, **kwargs)        # 通过第一个注意力层处理
            h = self.mid.block_2(h, temb, **kwargs)  # 通过第二个中间块处理
    
            # 向上采样过程
            for i_level in reversed(range(self.num_resolutions)):  # 反向遍历分辨率层级
                for i_block in range(self.num_res_blocks + 1):  # 遍历每个块
                    h = self.up[i_level].block[i_block](h, temb, **kwargs)  # 通过上采样块处理
                    if len(self.up[i_level].attn) > 0:  # 如果有注意力层
                        h = self.up[i_level].attn[i_block](h, **kwargs)  # 通过注意力层处理
                if i_level != 0:  # 如果不是最后一个分辨率层
                    h = self.up[i_level].upsample(h)  # 执行上采样操作
    
            # 结束处理
            if self.give_pre_end:  # 如果需要返回预处理结果
                return h
    
            h = self.norm_out(h)  # 通过输出归一化层处理
            h = nonlinearity(h)   # 应用非线性激活函数
            h = self.conv_out(h, **kwargs)  # 通过卷积输出层处理
            if self.tanh_out:  # 如果需要使用双曲正切激活函数
                h = torch.tanh(h)  # 应用双曲正切激活函数
            return h  # 返回最终输出
```