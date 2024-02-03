# `stable-diffusion-webui\extensions-builtin\ScuNET\scunet_model_arch.py`

```py
# -*- coding: utf-8 -*-
# 导入所需的库
import numpy as np
import torch
import torch.nn as nn
# 从 einops 库中导入 rearrange 和 Rearrange 函数
from einops import rearrange
from einops.layers.torch import Rearrange
# 从 timm.models.layers 中导入 trunc_normal_ 和 DropPath 函数

# 定义 WMSA 类，表示 Swin Transformer 中的自注意力模块
class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    # 初始化函数，定义模块的输入维度、输出维度、头维度、窗口大小和类型
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        # 初始化各参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        # 定义线性层，用于将输入维度映射到 3 倍的输入维度
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        # 定义相对位置参数，形状为 (2 * window_size - 1) * (2 * window_size - 1) * n_heads
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))

        # 定义线性层，将输入维度映射到输出维度
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        # 对相对位置参数进行截断正态分布初始化
        trunc_normal_(self.relative_position_params, std=.02)
        # 重新调整相对位置参数的形状，并转置维度
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1, 2).transpose(0, 1))
    # 生成 SW-MSA 的掩码
    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # 创建一个全零的张量作为注意力掩码，形状为 (h, w, p, p, p, p)，数据类型为布尔型，在指定设备上创建
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        # 如果类型为 'W'，直接返回掩码
        if self.type == 'W':
            return attn_mask

        # 计算偏移量
        s = p - shift
        # 设置掩码的值为 True，实现特定的掩码模式
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        # 重新排列掩码的维度，将其形状变为 (1, 1, w, p, p)
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        # 返回生成的注意力掩码
        return attn_mask
    def forward(self, x):
        """ 
        Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        # 如果不是窗口类型，则对输入张量进行滚动操作
        if self.type != 'W':
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))

        # 重新排列输入张量的维度
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # 对窗口的高度和宽度进行验证
        # assert h_windows == w_windows

        # 重新排列输入张量的维度
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # 添加可学习的相对嵌入
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # 使用注意力掩码来区分不同的子窗口
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        # 如果不是窗口类型，则对输出张量进行滚动操作
        if self.type != 'W':
            output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))

        return output
    # 定义一个方法用于计算相对位置的嵌入
    def relative_embedding(self):
        # 创建一个二维张量，包含了所有可能的坐标组合
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        # 计算坐标之间的相对位置关系
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # 从相对位置参数中获取相对位置关系对应的嵌入
        # 允许使用负数索引
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]
class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        # 定义一个 SwinTransformer 的 Block 类
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'
        # 如果输入分辨率小于等于窗口大小，则类型为 'W'

        self.ln1 = nn.LayerNorm(input_dim)
        # 初始化 LayerNorm 层
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        # 初始化 WMSA 模块
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 初始化 DropPath 模块，如果 drop_path 大于 0 则使用 DropPath，否则使用 nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        # 初始化 LayerNorm 层
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )
        # 初始化 MLP 模块，包含线性层和 GELU 激活函数

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        # 输入 x 经过 LayerNorm、WMSA 和 DropPath 模块后与原始输入相加
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        # 输入 x 经过 LayerNorm、MLP 和 DropPath 模块后与原始输入相加
        return x
        # 返回处理后的结果


class ConvTransBlock(nn.Module):
    # 初始化函数，定义了 SwinTransformer 和 Conv Block
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        # 调用父类的初始化函数
        super(ConvTransBlock, self).__init__()
        # 初始化各个参数
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        # 确保 type 参数的取值范围
        assert self.type in ['W', 'SW']
        # 如果输入分辨率小于等于窗口大小，则设置 type 为 'W'
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        # 初始化转换块
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path,
                                 self.type, self.input_resolution)
        # 初始化第一个卷积层
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        # 初始化第二个卷积层
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)

        # 初始化卷积块
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        )

    # 前向传播函数
    def forward(self, x):
        # 将输入 x 拆分为卷积部分和转换部分
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        # 对卷积部分进行卷积块操作
        conv_x = self.conv_block(conv_x) + conv_x
        # 将转换部分进行维度重排
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        # 对转换部分进行转换块操作
        trans_x = self.trans_block(trans_x)
        # 再次对转换部分进行维度重排
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        # 将卷积部分和转换部分拼接并通过第二个卷积层
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        # 将输入 x 与结果相加得到最终输出
        x = x + res

        return x
class SCUNet(nn.Module):
    # 定义 SCUNet 类，继承自 nn.Module

    def forward(self, x0):
        # 前向传播函数，接收输入 x0

        h, w = x0.size()[-2:]
        # 获取输入 x0 的高度和宽度
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        # 计算需要填充的底部像素数
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        # 计算需要填充的右侧像素数
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)
        # 使用 ReplicationPad2d 对输入 x0 进行填充

        x1 = self.m_head(x0)
        # 将填充后的输入 x0 传入 m_head 模块
        x2 = self.m_down1(x1)
        # 将 x1 传入 m_down1 模块
        x3 = self.m_down2(x2)
        # 将 x2 传入 m_down2 模块
        x4 = self.m_down3(x3)
        # 将 x3 传入 m_down3 模块
        x = self.m_body(x4)
        # 将 x4 传入 m_body 模块
        x = self.m_up3(x + x4)
        # 将 x 和 x4 相加后传入 m_up3 模块
        x = self.m_up2(x + x3)
        # 将 x 和 x3 相加后传入 m_up2 模块
        x = self.m_up1(x + x2)
        # 将 x 和 x2 相加后传入 m_up1 模块
        x = self.m_tail(x + x1)
        # 将 x 和 x1 相加后传入 m_tail 模块

        x = x[..., :h, :w]
        # 裁剪输出 x 的高度和宽度

        return x
        # 返回输出 x

    def _init_weights(self, m):
        # 初始化权重函数，接收模块 m

        if isinstance(m, nn.Linear):
            # 如果 m 是线性层
            trunc_normal_(m.weight, std=.02)
            # 对权重进行截断正态分布初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                # 如果存在偏置项，初始化为常数 0
        elif isinstance(m, nn.LayerNorm):
            # 如果 m 是 LayerNorm 层
            nn.init.constant_(m.bias, 0)
            # 初始化偏置项为常数 0
            nn.init.constant_(m.weight, 1.0)
            # 初始化权重为常数 1.0
```