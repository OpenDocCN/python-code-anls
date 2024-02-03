# `stable-diffusion-webui\extensions-builtin\SwinIR\swinir_model_arch.py`

```py
# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

# 导入所需的库
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 定义一个多层感知机（MLP）模块
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 第一个全连接层
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # 第二个全连接层
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 前向传播过程
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 将输入图像划分为窗口
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # 将输入图像按窗口大小划分
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

# 将窗口恢复为原始图像
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # 对输入张量 x 进行维度置换，将原来的维度顺序从 (0, 1, 2, 3, 4, 5) 变为 (0, 1, 3, 2, 4, 5)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # 返回处理后的张量 x
    return x
# 定义一个 WindowAttention 类，用于实现基于窗口的多头自注意力机制（W-MSA），支持相对位置偏置。
class WindowAttention(nn.Module):
    # 初始化函数，用于设置参数和说明
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels. 输入通道数
        window_size (tuple[int]): The height and width of the window. 窗口的高度和宽度
        num_heads (int): Number of attention heads. 注意力头的数量
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True 是否为查询、键、值添加可学习的偏置
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set 覆盖默认的 qk 缩放比例
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0 注意力权重的 dropout 比例
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0 输出的 dropout 比例
    """
    # 初始化函数，定义了注意力机制的参数
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        # 调用父类的初始化函数
        super().__init__()
        # 设置维度、窗口大小和头数
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        # 计算每个头的维度
        head_dim = dim // num_heads
        # 设置缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        # 定义相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # 获取每个窗口内每个令牌的配对相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 从0开始偏移
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # 定义线性变换层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 定义注意力机制的dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        # 定义投影层
        self.proj = nn.Linear(dim, dim)

        # 定义投影层的dropout层
        self.proj_drop = nn.Dropout(proj_drop)

        # 对相对位置偏置参数表进行截断正态分布初始化
        trunc_normal_(self.relative_position_bias_table, std=.02)
        # 定义softmax层
        self.softmax = nn.Softmax(dim=-1)
    # 前向传播函数，接受输入特征 x 和掩码 mask
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # 获取输入特征 x 的形状信息
        B_, N, C = x.shape
        # 使用 self.qkv 对象对输入 x 进行处理，并重塑形状
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 将处理后的结果分别赋值给 q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 对 q 进行缩放
        q = q * self.scale
        # 计算注意力矩阵
        attn = (q @ k.transpose(-2, -1))

        # 获取相对位置偏置表，并根据索引进行重塑
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # 调整相对位置偏置表的维度
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # 将相对位置偏置添加到注意力矩阵中
        attn = attn + relative_position_bias.unsqueeze(0)

        # 如果存在掩码，则将掩码应用到注意力矩阵中
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # 对注意力矩阵进行 dropout
        attn = self.attn_drop(attn)

        # 计算最终输出结果
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # 返回额外的表示信息
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    # 计算浮点运算量
    def flops(self, N):
        # 计算一个窗口中长度为 N 的令牌的浮点运算量
        flops = 0
        # 计算 qkv 的浮点运算量
        flops += N * self.dim * 3 * self.dim
        # 计算注意力矩阵的浮点运算量
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # 计算 x 的浮点运算量
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # 计算最终输出结果 x 的浮点运算量
        flops += N * self.dim * self.dim
        return flops
# 定义 Swin Transformer 模块
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): 输入通道数。
        input_resolution (tuple[int]): 输入分辨率。
        num_heads (int): 注意力头的数量。
        window_size (int): 窗口大小。
        shift_size (int): SW-MSA 的移动大小。
        mlp_ratio (float): MLP 隐藏层维度与嵌入维度的比率。
        qkv_bias (bool, optional): 如果为 True，则为查询、键、值添加可学习的偏置。默认值为 True。
        qk_scale (float | None, optional): 如果设置，则覆盖默认的 qk 缩放值 head_dim ** -0.5。
        drop (float, optional): 丢弃率。默认值为 0.0。
        attn_drop (float, optional): 注意力丢弃率。默认值为 0.0。
        drop_path (float, optional): 随机深度率。默认值为 0.0。
        act_layer (nn.Module, optional): 激活层。默认值为 nn.GELU。
        norm_layer (nn.Module, optional): 归一化层。默认值为 nn.LayerNorm。
    """
    # 初始化函数，设置模型参数和层
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的维度、输入分辨率、注意力头数、窗口大小、移动大小、MLP比例等参数
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # 如果输入分辨率中的最小值小于等于窗口大小，则不对窗口进行分区
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # 断言确保移动大小在0到窗口大小之间
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # 初始化模型的第一个层，LayerNorm
        self.norm1 = norm_layer(dim)
        # 初始化模型的注意力层，WindowAttention
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # 初始化模型的DropPath层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 初始化模型的第二个层，LayerNorm
        self.norm2 = norm_layer(dim)
        # 计算MLP的隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 初始化模型的MLP层
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 如果移动大小大于0，则计算注意力掩码
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        # 将注意力掩码注册为模型的缓冲区
        self.register_buffer("attn_mask", attn_mask)
    # 为 SW-MSA 计算注意力掩码
    def calculate_mask(self, x_size):
        # 获取输入张量的高度和宽度
        H, W = x_size
        # 创建一个全零张量作为图像掩码，形状为 (1, H, W, 1)
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        # 定义高度和宽度的切片范围
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        # 初始化计数器
        cnt = 0
        # 遍历高度和宽度的切片范围，为图像掩码赋值
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # 将图像掩码划分为窗口，形状为 (nW, window_size, window_size, 1)
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        # 将窗口重塑为二维张量
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # 计算注意力掩码
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # 将不为零的元素替换为 -100.0，将为零的元素替换为 0.0
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # 返回计算得到的注意力掩码
        return attn_mask
    # 前向传播函数，接受输入张量 x 和输入大小 x_size
    def forward(self, x, x_size):
        # 获取输入大小的高度和宽度
        H, W = x_size
        # 获取输入张量的批量大小、序列长度和通道数
        B, L, C = x.shape
        # assert 用于检查输入特征的大小是否正确，这里注释掉了

        # 保存输入张量的快捷连接
        shortcut = x
        # 对输入张量进行归一化处理
        x = self.norm1(x)
        # 将输入张量重塑为指定形状
        x = x.view(B, H, W, C)

        # 如果位移大小大于0，则进行循环位移操作
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 将位移后的张量分割为窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # 如果输入分辨率与 x_size 相同，则直接使用注意力机制处理窗口
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # 合并处理后的窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 如果位移大小大于0，则进行逆循环位移操作
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # 使用残差连接和随机丢弃路径处理输入张量
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 返回处理后的张量
        return x

    # 返回模型的额外信息，包括维度、输入分辨率、头数、窗口大小、位移大小和 MLP 比率
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
    # 计算模型的浮点运算次数
    def flops(self):
        # 初始化浮点运算次数
        flops = 0
        # 获取输入分辨率的高度和宽度
        H, W = self.input_resolution
        # 计算 norm1 操作的浮点运算次数
        flops += self.dim * H * W
        # 计算 W-MSA/SW-MSA 操作的浮点运算次数
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # 计算 mlp 操作的浮点运算次数
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # 计算 norm2 操作的浮点运算次数
        flops += self.dim * H * W
        # 返回总的浮点运算次数
        return flops
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        # 初始化 PatchMerging 类
        super().__init__()
        # 保存输入特征的分辨率和通道数
        self.input_resolution = input_resolution
        self.dim = dim
        # 创建线性层用于特征降维
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # 创建规范化层
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # 获取输入特征的高度和宽度
        H, W = self.input_resolution
        # 获取输入特征的维度
        B, L, C = x.shape
        # 检查输入特征的大小是否正确
        assert L == H * W, "input feature has wrong size"
        # 检查输入特征的高度和宽度是否为偶数
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # 将输入特征重塑为 B, H, W, C 的形状
        x = x.view(B, H, W, C)

        # 按照一定规则对输入特征进行切片
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # 将切片后的特征拼接在一起
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # 将特征重塑为 B, H/2*W/2, 4*C 的形状
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # 对特征进行规范化
        x = self.norm(x)
        # 对特征进行降维
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        # 返回额外的表示信息
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        # 计算模型的 FLOPs
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    # 定义一个函数，用于创建一个Transformer模型的配置
    Args:
        dim (int): 输入通道的数量
        input_resolution (tuple[int]): 输入分辨率
        depth (int): 块的数量
        num_heads (int): 注意力头的数量
        window_size (int): 局部窗口大小
        mlp_ratio (float): mlp隐藏维度与嵌入维度的比率
        qkv_bias (bool, optional): 如果为True，则为查询、键、值添加可学习的偏置。默认值为True
        qk_scale (float | None, optional): 如果设置，则覆盖默认的qk缩放为head_dim ** -0.5
        drop (float, optional): 丢弃率。默认值为0.0
        attn_drop (float, optional): 注意力丢弃率。默认值为0.0
        drop_path (float | tuple[float], optional): 随机深度率。默认值为0.0
        norm_layer (nn.Module, optional): 标准化层。默认值为nn.LayerNorm
        downsample (nn.Module | None, optional): 层末端的下采样层。默认值为None
        use_checkpoint (bool): 是否使用检查点来节省内存。默认值为False
    # 初始化函数，设置模型参数和结构
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
    
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的维度、输入分辨率、深度和是否使用检查点
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
    
        # 构建模型的块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
    
        # 设置补丁合并层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    # 前向传播函数
    def forward(self, x, x_size):
        # 遍历模型的块
        for blk in self.blocks:
            # 如果使用检查点，则调用检查点函数
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        # 如果存在补丁合并层，则对输出进行补丁合并
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
    # 返回模型的额外表示
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
    
    # 计算模型的浮点运算量
    def flops(self):
        flops = 0
        # 遍历模型的块，累加每个块的浮点运算量
        for blk in self.blocks:
            flops += blk.flops()
        # 如果存在补丁合并层，则加上补丁合并层的浮点运算量
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def forward(self, x, x_size):
        # Forward pass of the RSTB block
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        # Calculate the FLOPs (Floating Point Operations) of the RSTB block
        flops = 0
        # Add FLOPs from the residual group
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        # Add FLOPs from the convolution operation
        flops += H * W * self.dim * self.dim * 9
        # Add FLOPs from the patch embedding operation
        flops += self.patch_embed.flops()
        # Add FLOPs from the patch unembedding operation
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    # 定义一个类，用于实现图像分块处理
    class PatchEmbed(nn.Module):
        """
        Args:
            img_size (int): 图像大小。默认值为224。
            patch_size (int): 分块令牌大小。默认值为4。
            in_chans (int): 输入图像通道数。默认值为3。
            embed_dim (int): 线性投影输出通道数。默认值为96。
            norm_layer (nn.Module, optional): 标准化层。默认值为None。
        """
    
        def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
            super().__init__()
            img_size = to_2tuple(img_size)  # 将图像大小转换为二元组
            patch_size = to_2tuple(patch_size)  # 将分块令牌大小转换为二元组
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # 计算分块分辨率
            self.img_size = img_size  # 保存图像大小
            self.patch_size = patch_size  # 保存分块令牌大小
            self.patches_resolution = patches_resolution  # 保存分块分辨率
            self.num_patches = patches_resolution[0] * patches_resolution[1]  # 计算总分块数
    
            self.in_chans = in_chans  # 保存输入通道数
            self.embed_dim = embed_dim  # 保存线性投影输出通道数
    
            if norm_layer is not None:
                self.norm = norm_layer(embed_dim)  # 如果有标准化层，则初始化标准化层
            else:
                self.norm = None
    
        def forward(self, x):
            x = x.flatten(2).transpose(1, 2)  # 将输入张量展平并转置，得到 B Ph*Pw C
            if self.norm is not None:
                x = self.norm(x)  # 如果有标准化层，则对输入张量进行标准化
            return x
    
        def flops(self):
            flops = 0
            H, W = self.img_size
            if self.norm is not None:
                flops += H * W * self.embed_dim  # 计算浮点运算次数
            return flops
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        # 初始化函数，设置参数和属性
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        # 前向传播函数，处理输入数据
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        # 计算浮点运算次数
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        # 初始化函数，根据缩放因子和特征通道数创建上采样模块
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
# 定义 UpsampleOneStep 类，继承自 nn.Sequential
class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """
    
    # 初始化函数，接受 scale（缩放因子）、num_feat（中间特征通道数）、num_out_ch（输出通道数）、input_resolution（输入分辨率）参数
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        # 保存 num_feat 和 input_resolution 参数
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        # 创建空列表 m 用于存储模块
        m = []
        # 添加一个卷积层，输入通道为 num_feat，输出通道为 (scale ** 2) * num_out_ch，卷积核大小为 3，步长为 1，填充为 1
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        # 添加一个像素洗牌层，缩放因子为 scale
        m.append(nn.PixelShuffle(scale))
        # 调用父类 nn.Sequential 的初始化函数，传入模块列表 m
        super(UpsampleOneStep, self).__init__(*m)

    # 计算模型的浮点运算量
    def flops(self):
        # 获取输入分辨率的高度和宽度
        H, W = self.input_resolution
        # 计算浮点运算量，公式为 H * W * num_feat * 3 * 9
        flops = H * W * self.num_feat * 3 * 9
        # 返回计算结果
        return flops


# 定义 SwinIR 类，继承自 nn.Module
class SwinIR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.
    Args:
        img_size (int | tuple(int)): 输入图像大小。默认为64
        patch_size (int | tuple(int)): 补丁大小。默认为1
        in_chans (int): 输入图像通道数。默认为3
        embed_dim (int): 补丁嵌入维度。默认为96
        depths (tuple(int)): 每个Swin Transformer层的深度
        num_heads (tuple(int)): 不同层中的注意力头数
        window_size (int): 窗口大小。默认为7
        mlp_ratio (float): mlp隐藏维度与嵌入维度的比率。默认为4
        qkv_bias (bool): 如果为True，则为查询、键、值添加可学习偏置。默认为True
        qk_scale (float): 如果设置，则覆盖默认的qk缩放，即head_dim ** -0.5。默认为None
        drop_rate (float): 丢弃率。默认为0
        attn_drop_rate (float): 注意力丢弃率。默认为0
        drop_path_rate (float): 随机深度率。默认为0.1
        norm_layer (nn.Module): 标准化层。默认为nn.LayerNorm。
        ape (bool): 如果为True，则在补丁嵌入中添加绝对位置嵌入。默认为False
        patch_norm (bool): 如果为True，则在补丁嵌入后添加标准化。默认为True
        use_checkpoint (bool): 是否使用检查点来节省内存。默认为False
        upscale: 放大因子。2/3/4/8用于图像SR，1用于去噪和压缩伪影减少
        img_range: 图像范围。1.或255。
        upsampler: 重建模块。'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: 残差连接之前的卷积块。'1conv'/'3conv'
    """

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 对线性层的权重进行截断正态初始化
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 如果是线性层并且有偏置，则将偏置初始化为0
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 对LayerNorm层的偏置初始化为0
            nn.init.constant_(m.bias, 0)
            # 对LayerNorm层的权重初始化为1.0
            nn.init.constant_(m.weight, 1.0)
    # 标记该方法不会被 Torch JIT 编译，不会参与权重衰减
    @torch.jit.ignore
    def no_weight_decay(self):
        # 返回不参与权重衰减的参数名称集合
        return {'absolute_pos_embed'}

    # 标记该方法不会被 Torch JIT 编译，不会参与权重衰减
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        # 返回不参与权重衰减的参数名称集合
        return {'relative_position_bias_table'}

    # 检查输入图像的尺寸是否符合要求，如果不符合则进行填充
    def check_image_size(self, x):
        # 获取输入图像的高度和宽度
        _, _, h, w = x.size()
        # 计算需要填充的高度和宽度
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # 对输入图像进行反射填充
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    # 前向传播特征提取部分
    def forward_features(self, x):
        # 记录输入图像的尺寸
        x_size = (x.shape[2], x.shape[3])
        # 对输入图像进行 patch embedding
        x = self.patch_embed(x)
        # 如果使用绝对位置编码，则将其加到输入中
        if self.ape:
            x = x + self.absolute_pos_embed
        # 对位置编码后的输入进行 dropout
        x = self.pos_drop(x)

        # 逐层进行特征提取
        for layer in self.layers:
            x = layer(x, x_size)

        # 对提取后的特征进行归一化
        x = self.norm(x)  # B L C
        # 对特征进行 patch unembedding
        x = self.patch_unembed(x, x_size)

        return x
    # 定义前向传播函数，接受输入张量 x
    def forward(self, x):
        # 获取输入张量 x 的高度和宽度
        H, W = x.shape[2:]
        # 调用 check_image_size 函数，对输入张量 x 进行处理
        x = self.check_image_size(x)

        # 将均值转换为与输入张量 x 相同的数据类型
        self.mean = self.mean.type_as(x)
        # 对输入张量 x 进行归一化处理
        x = (x - self.mean) * self.img_range

        # 根据不同的上采样器类型进行处理
        if self.upsampler == 'pixelshuffle':
            # 对输入张量 x 进行卷积操作
            x = self.conv_first(x)
            # 对卷积后的结果进行处理
            x = self.conv_after_body(self.forward_features(x)) + x
            # 对上采样前的结果进行处理
            x = self.conv_before_upsample(x)
            # 对上采样后的结果进行处理
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # 对输入张量 x 进行卷积操作
            x = self.conv_first(x)
            # 对卷积后的结果进行处理
            x = self.conv_after_body(self.forward_features(x)) + x
            # 对上采样后的结果进行处理
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # 对输入张量 x 进行卷积操作
            x = self.conv_first(x)
            # 对卷积后的结果进行处理
            x = self.conv_after_body(self.forward_features(x)) + x
            # 对上采样前的结果进行处理
            x = self.conv_before_upsample(x)
            # 对上采样后的结果进行处理
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            # 如果上采样比例为 4，则再进行一次上采样处理
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            # 对上采样后的结果进行处理
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # 对输入张量 x 进行卷积操作
            x_first = self.conv_first(x)
            # 对卷积后的结果进行处理
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            # 将处理后的结果与输入张量 x 相加
            x = x + self.conv_last(res)

        # 对处理后的结果进行反归一化处理，并加上均值
        x = x / self.img_range + self.mean

        # 返回处理后的结果，保留指定的高度和宽度
        return x[:, :, :H*self.upscale, :W*self.upscale]

    # 计算模型的浮点运算量
    def flops(self):
        # 初始化浮点运算量为 0
        flops = 0
        # 获取分块分辨率
        H, W = self.patches_resolution
        # 计算嵌入维度相关的浮点运算量
        flops += H * W * 3 * self.embed_dim * 9
        # 计算 patch_embed 模块的浮点运算量
        flops += self.patch_embed.flops()
        # 遍历每个层并计算浮点运算量
        for layer in self.layers:
            flops += layer.flops()
        # 计算嵌入维度相关的浮点运算量
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        # 计算上采样模块的浮点运算量
        flops += self.upsample.flops()
        # 返回总的浮点运算量
        return flops
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 设置放大倍数
    upscale = 4
    # 设置窗口大小
    window_size = 8
    # 计算调整后的图像高度
    height = (1024 // upscale // window_size + 1) * window_size
    # 计算调整后的图像宽度
    width = (720 // upscale // window_size + 1) * window_size
    # 创建 SwinIR 模型对象，设置参数
    model = SwinIR(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    # 打印模型信息
    print(model)
    # 打印图像高度、宽度和模型的浮点运算次数
    print(height, width, model.flops() / 1e9)

    # 生成随机输入张量
    x = torch.randn((1, 3, height, width))
    # 使用模型处理输入张量
    x = model(x)
    # 打印处理后张量的形状
    print(x.shape)
```