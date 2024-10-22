# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\model.py`

```py
# pytorch_diffusion + derived encoder decoder
# 导入数学库
import math
# 导入类型注解相关
from typing import Any, Callable, Optional

# 导入 numpy 库
import numpy as np
# 导入 pytorch 库
import torch
# 导入 pytorch 神经网络模块
import torch.nn as nn
# 导入 rearrange 函数以处理张量重排列
from einops import rearrange
# 导入版本管理库
from packaging import version

# 尝试导入 xformers 模块
try:
    import xformers
    import xformers.ops

    # 如果成功导入，设置标志为 True
    XFORMERS_IS_AVAILABLE = True
except:
    # 如果导入失败，设置标志为 False，并打印提示信息
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

# 从其他模块导入 LinearAttention 和 MemoryEfficientCrossAttention
from ...modules.attention import LinearAttention, MemoryEfficientCrossAttention


def get_timestep_embedding(timesteps, embedding_dim):
    """
    此函数与 Denoising Diffusion Probabilistic Models 中的实现相匹配：
    来自 Fairseq。
    构建正弦嵌入。
    此实现与 tensor2tensor 中的实现相匹配，但与 "Attention Is All You Need" 第 3.5 节中的描述略有不同。
    """
    # 确保时间步长是一维的
    assert len(timesteps.shape) == 1

    # 计算嵌入维度的一半
    half_dim = embedding_dim // 2
    # 计算嵌入因子的对数
    emb = math.log(10000) / (half_dim - 1)
    # 计算并生成指数衰减的嵌入
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将嵌入移动到与时间步相同的设备上
    emb = emb.to(device=timesteps.device)
    # 扩展时间步并与嵌入相乘
    emb = timesteps.float()[:, None] * emb[None, :]
    # 将正弦和余弦嵌入拼接在一起
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度是奇数，则进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回最终的嵌入
    return emb


def nonlinearity(x):
    # 使用 swish 激活函数
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    # 返回一个 GroupNorm 归一化层
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # 初始化 Upsample 类
        super().__init__()
        # 记录是否使用卷积
        self.with_conv = with_conv
        # 如果使用卷积，则定义卷积层
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        # 使用最近邻插值将输入张量上采样
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 如果使用卷积，则应用卷积层
        if self.with_conv:
            x = self.conv(x)
        # 返回处理后的张量
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        # 初始化 Downsample 类
        super().__init__()
        # 记录是否使用卷积
        self.with_conv = with_conv
        # 如果使用卷积，则定义卷积层
        if self.with_conv:
            # 因为 pytorch 卷积不支持不对称填充，需手动处理
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        # 如果使用卷积，先进行填充再应用卷积层
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        # 否则使用平均池化进行下采样
        else:
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
        # 如果未指定输出通道数，则设置为输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        # 保存输出通道数
        self.out_channels = out_channels
        # 保存是否使用卷积捷径的标志
        self.use_conv_shortcut = conv_shortcut

        # 初始化输入通道数的归一化层
        self.norm1 = Normalize(in_channels)
        # 定义第一层卷积，输入输出通道及卷积核参数
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # 如果有时间嵌入通道，则定义时间嵌入投影层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化输出通道数的归一化层
        self.norm2 = Normalize(out_channels)
        # 定义 dropout 层
        self.dropout = torch.nn.Dropout(dropout)
        # 定义第二层卷积，输入输出通道及卷积核参数
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # 如果输入和输出通道数不相同
        if self.in_channels != self.out_channels:
            # 如果使用卷积捷径，则定义卷积捷径层
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            # 否则定义 1x1 卷积捷径层
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    # 前向传播函数
    def forward(self, x, temb):
        # 将输入赋值给 h 变量
        h = x
        # 对 h 进行归一化
        h = self.norm1(h)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过第一层卷积处理 h
        h = self.conv1(h)

        # 如果时间嵌入不为 None
        if temb is not None:
            # 将时间嵌入通过非线性激活函数处理后投影到输出通道，并与 h 相加
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # 对 h 进行第二次归一化
        h = self.norm2(h)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过 dropout 层处理 h
        h = self.dropout(h)
        # 通过第二层卷积处理 h
        h = self.conv2(h)

        # 如果输入和输出通道数不相同
        if self.in_channels != self.out_channels:
            # 如果使用卷积捷径，则通过卷积捷径层处理 x
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            # 否则通过 1x1 卷积捷径层处理 x
            else:
                x = self.nin_shortcut(x)

        # 返回 x 和 h 的相加结果
        return x + h
# 定义 LinAttnBlock 类，继承自 LinearAttention
class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""  # 文档字符串，说明该类用于匹配 AttnBlock 的使用方式

    # 初始化方法，接受输入通道数
    def __init__(self, in_channels):
        # 调用父类的初始化方法，设置维度和头数
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


# 定义 AttnBlock 类，继承自 nn.Module
class AttnBlock(nn.Module):
    # 初始化方法，接受输入通道数
    def __init__(self, in_channels):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 初始化归一化层
        self.norm = Normalize(in_channels)
        # 初始化查询卷积层
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # 初始化键卷积层
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # 初始化值卷积层
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # 初始化输出投影卷积层
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    # 定义注意力计算方法
    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        # 对输入进行归一化
        h_ = self.norm(h_)
        # 计算查询、键和值
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 获取查询的形状参数
        b, c, h, w = q.shape
        # 重新排列查询、键和值的形状
        q, k, v = map(
            lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v)
        )
        # 计算缩放的点积注意力
        h_ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # scale is dim ** -0.5 per default
        # 计算注意力

        # 返回重新排列后的注意力结果
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    # 定义前向传播方法
    def forward(self, x, **kwargs):
        # 将输入赋值给 h_
        h_ = x
        # 计算注意力
        h_ = self.attention(h_)
        # 应用输出投影
        h_ = self.proj_out(h_)
        # 返回输入与注意力结果的和
        return x + h_


# 定义 MemoryEfficientAttnBlock 类，继承自 nn.Module
class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """  # 文档字符串，说明该类使用 xformers 高效实现的单头自注意力

    # 初始化方法，接受输入通道数
    def __init__(self, in_channels):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 初始化归一化层
        self.norm = Normalize(in_channels)
        # 初始化查询卷积层
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # 初始化键卷积层
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # 初始化值卷积层
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # 初始化输出投影卷积层
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # 初始化注意力操作，类型为可选的任意类型
        self.attention_op: Optional[Any] = None
    # 定义注意力机制的函数，输入为一个张量，输出也是一个张量
        def attention(self, h_: torch.Tensor) -> torch.Tensor:
            # 先对输入进行归一化处理
            h_ = self.norm(h_)
            # 通过线性变换生成查询张量
            q = self.q(h_)
            # 通过线性变换生成键张量
            k = self.k(h_)
            # 通过线性变换生成值张量
            v = self.v(h_)
    
            # 计算注意力
            # 获取查询张量的形状信息
            B, C, H, W = q.shape
            # 调整张量形状，将其从四维转为二维
            q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))
    
            # 对查询、键、值进行维度调整以便计算注意力
            q, k, v = map(
                lambda t: t.unsqueeze(3)  # 在最后增加一个维度
                .reshape(B, t.shape[1], 1, C)  # 调整形状
                .permute(0, 2, 1, 3)  # 变换维度顺序
                .reshape(B * 1, t.shape[1], C)  # 重新调整形状
                .contiguous(),  # 保证内存连续性
                (q, k, v),
            )
            # 使用内存高效的注意力操作
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )
    
            # 调整输出张量的形状
            out = (
                out.unsqueeze(0)  # 增加一个维度
                .reshape(B, 1, out.shape[1], C)  # 调整形状
                .permute(0, 2, 1, 3)  # 变换维度顺序
                .reshape(B, out.shape[1], C)  # 重新调整形状
            )
            # 将输出张量的形状恢复为原来的格式
            return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
    
        # 定义前向传播函数
        def forward(self, x, **kwargs):
            # 输入数据赋值给 h_
            h_ = x
            # 通过注意力机制处理 h_
            h_ = self.attention(h_)
            # 通过输出投影处理 h_
            h_ = self.proj_out(h_)
            # 返回输入和处理后的 h_ 的和
            return x + h_
# 定义一个内存高效的交叉注意力包装类，继承自 MemoryEfficientCrossAttention
class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    # 前向传播方法，接受输入张量和可选的上下文、掩码
    def forward(self, x, context=None, mask=None, **unused_kwargs):
        # 解包输入张量的维度：批量大小、通道数、高度和宽度
        b, c, h, w = x.shape
        # 重新排列输入张量的维度，将 (b, c, h, w) 转换为 (b, h*w, c)
        x = rearrange(x, "b c h w -> b (h w) c")
        # 调用父类的 forward 方法，处理重新排列后的输入
        out = super().forward(x, context=context, mask=mask)
        # 将输出张量的维度重新排列回 (b, c, h, w)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        # 返回输入与输出的和，进行残差连接
        return x + out


# 定义一个生成注意力模块的函数
def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    # 检查传入的注意力类型是否在支持的类型列表中
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
        "memory-efficient-cross-attn",
        "linear",
        "none",
    ], f"attn_type {attn_type} unknown"
    # 检查 PyTorch 版本，并且如果类型不是 "none"，则验证是否可用 xformers
    if (
        version.parse(torch.__version__) < version.parse("2.0.0")
        and attn_type != "none"
    ):
        assert XFORMERS_IS_AVAILABLE, (
            f"We do not support vanilla attention in {torch.__version__} anymore, "
            f"as it is too expensive. Please install xformers via e.g. 'pip install xformers==0.0.16'"
        )
        # 将注意力类型设置为 "vanilla-xformers"
        attn_type = "vanilla-xformers"
    # 根据注意力类型生成相应的注意力块
    if attn_type == "vanilla":
        # 验证注意力参数不为 None
        assert attn_kwargs is None
        # 返回标准的注意力块
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        # 返回内存高效的注意力块
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "memory-efficient-cross-attn":
        # 设置查询维度为输入通道数
        attn_kwargs["query_dim"] = in_channels
        # 返回内存高效的交叉注意力包装类
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        # 返回一个身份映射层，不改变输入
        return nn.Identity(in_channels)
    else:
        # 返回线性注意力块
        return LinAttnBlock(in_channels)


# 定义一个模型类，继承自 nn.Module
class Model(nn.Module):
    # 初始化方法，接受多个参数进行模型构建
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
    # 定义前向传播方法，接受输入 x、时间步 t 和上下文 context
        def forward(self, x, t=None, context=None):
            # 确保输入 x 的高度和宽度与设定的分辨率相等（被注释掉）
            # assert x.shape[2] == x.shape[3] == self.resolution
            # 如果上下文不为 None，沿通道维度连接输入 x 和上下文
            if context is not None:
                # 假设上下文对齐，沿通道轴拼接
                x = torch.cat((x, context), dim=1)
            # 如果使用时间步，进行时间步嵌入
            if self.use_timestep:
                # 确保时间步 t 不为 None
                assert t is not None
                # 获取时间步嵌入
                temb = get_timestep_embedding(t, self.ch)
                # 通过第一层密集层处理时间步嵌入
                temb = self.temb.dense[0](temb)
                # 应用非线性变换
                temb = nonlinearity(temb)
                # 通过第二层密集层处理
                temb = self.temb.dense[1](temb)
            else:
                # 如果不使用时间步，设置时间步嵌入为 None
                temb = None
    
            # 下采样
            hs = [self.conv_in(x)]  # 初始卷积层的输出
            for i_level in range(self.num_resolutions):
                for i_block in range(self.num_res_blocks):
                    # 通过当前下采样层和时间步嵌入处理前一层输出
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    # 如果存在注意力层，则对输出进行注意力处理
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    # 将处理后的输出添加到列表
                    hs.append(h)
                # 如果不是最后一层分辨率，进行下采样
                if i_level != self.num_resolutions - 1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
    
            # 中间处理
            h = hs[-1]  # 获取最后一层的输出
            h = self.mid.block_1(h, temb)  # 通过中间块处理
            h = self.mid.attn_1(h)  # 通过中间注意力层处理
            h = self.mid.block_2(h, temb)  # 再次通过中间块处理
    
            # 上采样
            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    # 拼接上层输出和当前层的输出，然后通过上采样块处理
                    h = self.up[i_level].block[i_block](
                        torch.cat([h, hs.pop()], dim=1), temb
                    )
                    # 如果存在注意力层，则对输出进行注意力处理
                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
                # 如果不是第一层分辨率，进行上采样
                if i_level != 0:
                    h = self.up[i_level].upsample(h)
    
            # 结束处理
            h = self.norm_out(h)  # 最后的归一化处理
            h = nonlinearity(h)  # 应用非线性变换
            h = self.conv_out(h)  # 通过输出卷积层处理
            return h  # 返回最终输出
    
        # 获取最后一层的卷积权重
        def get_last_layer(self):
            return self.conv_out.weight  # 返回输出卷积层的权重
# 定义一个编码器类，继承自 nn.Module
class Encoder(nn.Module):
    # 初始化方法，接收多个参数用于配置编码器
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
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        mid_attn=True,
        **ignore_kwargs,
    ):
        # 调用父类构造方法
        super().__init__()
        # 如果使用线性注意力，设置注意力类型为线性
        if use_linear_attn:
            attn_type = "linear"
        # 保存输入参数以供后续使用
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.attn_resolutions = attn_resolutions
        self.mid_attn = mid_attn

        # 下采样
        # 定义输入卷积层
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        # 当前分辨率初始化
        curr_res = resolution
        # 定义输入通道的倍率
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        # 初始化下采样模块列表
        self.down = nn.ModuleList()
        # 遍历每个分辨率层级
        for i_level in range(self.num_resolutions):
            # 初始化块和注意力模块列表
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # 输入和输出通道数计算
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks):
                # 添加残差块到块列表中
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                # 更新输入通道数为当前块的输出通道数
                block_in = block_out
                # 如果当前分辨率在注意力分辨率列表中，添加注意力模块
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            # 创建下采样模块
            down = nn.Module()
            down.block = block
            down.attn = attn
            # 如果不是最后一个分辨率，添加下采样层
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                # 更新当前分辨率为一半
                curr_res = curr_res // 2
            # 将下采样模块添加到列表中
            self.down.append(down)

        # 中间层
        self.mid = nn.Module()
        # 添加第一个残差块
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        # 如果使用中间注意力，添加注意力模块
        if mid_attn:
            self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        # 添加第二个残差块
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # 结束层
        # 定义归一化层
        self.norm_out = Normalize(block_in)
        # 定义输出卷积层，根据是否双 z 通道设置输出通道数
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    # 定义前向传播方法，接受输入数据 x
    def forward(self, x):
        # 时间步嵌入初始化为 None
        temb = None

        # 下采样过程
        # 对输入 x 进行卷积操作，生成初始特征图 hs
        hs = [self.conv_in(x)]
        # 遍历每个分辨率层
        for i_level in range(self.num_resolutions):
            # 遍历当前分辨率层中的每个残差块
            for i_block in range(self.num_res_blocks):
                # 使用当前层的残差块处理上一个层的输出和时间步嵌入
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # 如果当前层有注意力机制，则应用注意力
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # 将当前层的输出添加到特征图列表中
                hs.append(h)
            # 如果当前层不是最后一个分辨率层，则进行下采样
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # 中间处理阶段
        h = hs[-1]  # 获取最后一层的输出
        # 通过中间块1处理输入
        h = self.mid.block_1(h, temb)
        # 如果中间层有注意力机制，则应用注意力
        if self.mid_attn:
            h = self.mid.attn_1(h)
        # 通过中间块2处理输出
        h = self.mid.block_2(h, temb)

        # 最终处理阶段
        h = self.norm_out(h)  # 应用输出归一化
        h = nonlinearity(h)   # 应用非线性激活函数
        h = self.conv_out(h)  # 通过输出卷积生成最终结果
        return h  # 返回最终输出
# 定义一个解码器类，继承自 PyTorch 的 nn.Module
class Decoder(nn.Module):
    # 初始化方法，定义解码器的参数
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍增因子
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力机制应用的分辨率
        dropout=0.0,  # dropout 比例，默认值为 0
        resamp_with_conv=True,  # 是否使用卷积进行上采样
        in_channels,  # 输入的通道数
        resolution,  # 输入的分辨率
        z_channels,  # 潜在变量的通道数
        give_pre_end=False,  # 是否在前面给予额外的结束标志
        tanh_out=False,  # 输出是否经过 tanh 激活
        use_linear_attn=False,  # 是否使用线性注意力机制
        attn_type="vanilla",  # 注意力类型，默认为“vanilla”
        mid_attn=True,  # 是否在中间层使用注意力
        **ignorekwargs,  # 其他忽略的参数，采用关键字参数形式
    ):
        # 初始化父类
        super().__init__()
        # 如果使用线性注意力机制，设置注意力类型为线性
        if use_linear_attn:
            attn_type = "linear"
        # 设置通道数
        self.ch = ch
        # 初始化时间嵌入通道数为0
        self.temb_ch = 0
        # 计算分辨率数量
        self.num_resolutions = len(ch_mult)
        # 设置残差块数量
        self.num_res_blocks = num_res_blocks
        # 设置输入分辨率
        self.resolution = resolution
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置是否给出前置结束标志
        self.give_pre_end = give_pre_end
        # 设置激活函数输出
        self.tanh_out = tanh_out
        # 设置注意力分辨率
        self.attn_resolutions = attn_resolutions
        # 设置中间注意力
        self.mid_attn = mid_attn

        # 计算输入通道倍数、块输入通道和当前最低分辨率
        in_ch_mult = (1,) + tuple(ch_mult)
        # 计算当前块的输入通道数
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # 计算当前分辨率
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # 设置潜在变量的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print(
        #     "Working with z of shape {} = {} dimensions.".format(
        #         self.z_shape, np.prod(self.z_shape)
        #     )
        # )

        # 创建注意力和残差块类
        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()
        # 将潜在变量映射到块输入通道
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # 中间层
        self.mid = nn.Module()
        # 创建第一个残差块
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        # 如果启用中间注意力，创建注意力层
        if mid_attn:
            self.mid.attn_1 = make_attn_cls(block_in, attn_type=attn_type)
        # 创建第二个残差块
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # 上采样层
        self.up = nn.ModuleList()
        # 从高到低遍历每个分辨率级别
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()  # 残差块列表
            attn = nn.ModuleList()   # 注意力层列表
            # 计算当前块的输出通道数
            block_out = ch * ch_mult[i_level]
            # 创建每个残差块
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    make_resblock_cls(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                # 更新块输入通道
                block_in = block_out
                # 如果当前分辨率在注意力分辨率中，添加注意力层
                if curr_res in attn_resolutions:
                    attn.append(make_attn_cls(block_in, attn_type=attn_type))
            up = nn.Module()  # 上采样模块
            up.block = block  # 添加残差块
            up.attn = attn   # 添加注意力层
            # 如果不是最低分辨率，添加上采样层
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                # 更新当前分辨率
                curr_res = curr_res * 2
            # 将上采样模块插入列表的开头
            self.up.insert(0, up)  # prepend to get consistent order

        # 结束层
        # 创建归一化层
        self.norm_out = Normalize(block_in)
        # 创建输出卷积层
        self.conv_out = make_conv_cls(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )
    # 定义一个私有方法，用于返回注意力机制的构造函数
    def _make_attn(self) -> Callable:
        return make_attn

    # 定义一个私有方法，用于返回残差块的构造函数
    def _make_resblock(self) -> Callable:
        return ResnetBlock

    # 定义一个私有方法，用于返回二维卷积层的构造函数
    def _make_conv(self) -> Callable:
        return torch.nn.Conv2d

    # 获取最后一层的权重
    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    # 前向传播方法，接收输入 z 和可选参数
    def forward(self, z, **kwargs):
        # 确保输入 z 的形状与预期相同（被注释掉的检查）
        # assert z.shape[1:] == self.z_shape[1:]
        # 记录输入 z 的形状
        self.last_z_shape = z.shape

        # 初始化时间步嵌入
        temb = None

        # 将输入 z 传入卷积层
        h = self.conv_in(z)

        # 中间处理
        h = self.mid.block_1(h, temb, **kwargs)  # 通过第一块中间块处理
        if self.mid_attn:  # 如果启用了中间注意力
            h = self.mid.attn_1(h, **kwargs)  # 应用中间注意力层
        h = self.mid.block_2(h, temb, **kwargs)  # 通过第二块中间块处理

        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):  # 从最高分辨率到最低分辨率
            for i_block in range(self.num_res_blocks + 1):  # 遍历每个残差块
                h = self.up[i_level].block[i_block](h, temb, **kwargs)  # 通过上采样块处理
                if len(self.up[i_level].attn) > 0:  # 如果存在注意力层
                    h = self.up[i_level].attn[i_block](h, **kwargs)  # 应用注意力层
            if i_level != 0:  # 如果不是最低分辨率
                h = self.up[i_level].upsample(h)  # 执行上采样

        # 结束处理
        if self.give_pre_end:  # 如果启用了预处理结束返回
            return h

        h = self.norm_out(h)  # 对输出进行归一化
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.conv_out(h, **kwargs)  # 通过最终卷积层处理
        if self.tanh_out:  # 如果启用了 Tanh 输出
            h = torch.tanh(h)  # 应用 Tanh 激活函数
        return h  # 返回最终输出
```