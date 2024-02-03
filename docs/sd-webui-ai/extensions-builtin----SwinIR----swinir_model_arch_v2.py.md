# `stable-diffusion-webui\extensions-builtin\SwinIR\swinir_model_arch_v2.py`

```
# -----------------------------------------------------------------------------------
# Swin2SR: Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration, https://arxiv.org/abs/
# Written by Conde and Choi et al.
# -----------------------------------------------------------------------------------

# 导入所需的库
import math
import numpy as np
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

# 将窗口反转为图像
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
    # 将输入张量按照窗口大小进行切割，得到一个新的张量
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # 对切割后的张量进行维度变换，使得后两个维度交换位置
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # 返回变换后的张量
    return x
# 定义一个窗口注意力机制的类，支持相对位置偏差，同时支持移位和非移位窗口
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels. 输入通道数
        window_size (tuple[int]): The height and width of the window. 窗口的高度和宽度
        num_heads (int): Number of attention heads. 注意力头的数量
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True 是否对查询、键、值添加可学习的偏置
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0 注意力权重的丢弃率
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0 输出的丢弃率
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training. 预训练中窗口的高度和宽度
    """
    # 前向传播函数，用于计算自注意力机制的输出
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)  # 输入特征，形状为(num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None  # 掩码，形状为(num_windows, Wh*Ww, Wh*Ww)，可以为0或-inf，或者为None
        """
        B_, N, C = x.shape  # 获取输入特征的形状
        qkv_bias = None  # 初始化qkv_bias变量为None
        if self.q_bias is not None:  # 如果存在q_bias
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))  # 将q_bias和v_bias拼接成qkv_bias
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)  # 线性变换得到qkv
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 重塑形状并进行维度置换
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别获取q、k、v

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # 计算注意力得分
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()  # 对logit_scale进行限制
        attn = attn * logit_scale  # 缩放注意力得分

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)  # 计算相对位置偏置表
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # 获取相对位置偏置
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # 维度置换
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)  # 对相对位置偏置进行缩放
        attn = attn + relative_position_bias.unsqueeze(0)  # 添加相对位置偏置到注意力得分中

        if mask is not None:  # 如果存在掩码
            nW = mask.shape[0]  # 获取掩码的形状
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)  # 将掩码应用到注意力得分中
            attn = attn.view(-1, self.num_heads, N, N)  # 重塑形状
            attn = self.softmax(attn)  # 对注意力得分进行softmax
        else:
            attn = self.softmax(attn)  # 对注意力得分进行softmax

        attn = self.attn_drop(attn)  # 对注意力得分进行dropout

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # 计算加权和并重塑形状
        x = self.proj(x)  # 线性变换
        x = self.proj_drop(x)  # 对输出进行dropout
        return x  # 返回输出
    # 返回模型的额外信息，包括维度、窗口大小、预训练窗口大小和注意力头数
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    # 计算给定长度为 N 的 token 的 1 个窗口的 FLOPs
    def flops(self, N):
        # 初始化 FLOPs 计数
        flops = 0
        # 计算 QKV 的 FLOPs
        flops += N * self.dim * 3 * self.dim
        # 计算注意力机制的 FLOPs
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # 计算乘以注意力权重后的 FLOPs
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # 计算投影层的 FLOPs
        flops += N * self.dim * self.dim
        # 返回总的 FLOPs
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
        drop (float, optional): 丢弃率。默认值为 0.0。
        attn_drop (float, optional): 注意力丢弃率。默认值为 0.0。
        drop_path (float, optional): 随机深度率。默认值为 0.0。
        act_layer (nn.Module, optional): 激活层。默认值为 nn.GELU。
        norm_layer (nn.Module, optional): 归一化层。默认值为 nn.LayerNorm。
        pretrained_window_size (int): 预训练中的窗口大小。
    """
    # 初始化函数，设置模型参数和结构
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
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
        # 断言移动大小在0到窗口大小之间
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # 初始化模型的第一个层，LayerNorm
        self.norm1 = norm_layer(dim)
        # 初始化模型的注意力层，WindowAttention
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

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
        img_mask = torch.zeros((1, H, W, 1))  
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
        mask_windows = window_partition(img_mask, self.window_size)  
        # 将窗口重塑为二维张量
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # 计算注意力掩码
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # 将不为0的元素替换为-100.0，将为0的元素替换为0.0
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
    # 前向传播函数，接受输入张量 x 和输入大小 x_size
    def forward(self, x, x_size):
        # 获取输入大小的高度和宽度
        H, W = x_size
        # 获取输入张量的批量大小、序列长度和通道数
        B, L, C = x.shape
        # 断言序列长度等于高度乘以宽度，用于检查输入特征的大小是否正确

        # 保存输入张量的快捷方式
        shortcut = x
        # 将输入张量重塑为 B, H, W, C 的形状
        x = x.view(B, H, W, C)

        # 循环移位
        if self.shift_size > 0:
            # 沿着指定维度对输入张量进行循环移位
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA（为了在形状为窗口大小的倍数的图像上进行测试而兼容）
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    # 返回模型的额外表示信息
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
    # 计算模型的浮点运算量
    def flops(self):
        # 初始化浮点运算量为0
        flops = 0
        # 获取输入分辨率
        H, W = self.input_resolution
        # 计算 norm1 的浮点运算量
        flops += self.dim * H * W
        # 计算 W-MSA/SW-MSA 的浮点运算量
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # 计算 mlp 的浮点运算量
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # 计算 norm2 的浮点运算量
        flops += self.dim * H * W
        # 返回总的浮点运算量
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
        # 保存输入分辨率和通道数
        self.input_resolution = input_resolution
        self.dim = dim
        # 创建线性层用于降维
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # 创建规范化层
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # 获取输入特征的高度和宽度
        H, W = self.input_resolution
        # 获取输入特征的形状
        B, L, C = x.shape
        # 检查输入特征的形状是否正确
        assert L == H * W, "input feature has wrong size"
        # 检查输入特征的高度和宽度是否为偶数
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # 将输入特征重塑为四维张量
        x = x.view(B, H, W, C)

        # 分割输入特征为四个部分
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # 将四个部分拼接在一起
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # 重塑张量形状
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # 进行降维操作
        x = self.reduction(x)
        # 进行规范化操作
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        # 返回额外的表示信息
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        # 计算 FLOPs
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    # 定义一个类的初始化方法，用于初始化模型的参数
    Args:
        dim (int): 输入通道的数量
        input_resolution (tuple[int]): 输入分辨率
        depth (int): 块的数量
        num_heads (int): 注意力头的数量
        window_size (int): 局部窗口大小
        mlp_ratio (float): mlp隐藏维度与嵌入维度的比率
        qkv_bias (bool, optional): 如果为True，则为查询、键、值添加可学习的偏置。默认值为True
        drop (float, optional): 丢弃率。默认值为0.0
        attn_drop (float, optional): 注意力丢弃率。默认值为0.0
        drop_path (float | tuple[float], optional): 随机深度率。默认值为0.0
        norm_layer (nn.Module, optional): 标准化层。默认值为nn.LayerNorm
        downsample (nn.Module | None, optional): 层末端的下采样层。默认值为None
        use_checkpoint (bool): 是否使用检查点来节省内存。默认值为False
        pretrained_window_size (int): 预训练中的局部窗口大小
    # 初始化函数，设置模型参数和配置
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
    
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的维度、输入分辨率、深度和是否使用检查点
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
    
        # 构建模型的块
        self.blocks = nn.ModuleList([
            # 创建 SwinTransformerBlock 块
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
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
            # 如果使用检查点，则使用检查点函数
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        # 如果存在补丁合并层，则应用补丁合并
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
    # 返回模型的额外表示信息
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
    # 计算整个网络的浮点运算次数
    def flops(self):
        # 初始化浮点运算次数为0
        flops = 0
        # 遍历网络中的每个块，计算每个块的浮点运算次数并累加到总数中
        for blk in self.blocks:
            flops += blk.flops()
        # 如果存在下采样块，则计算下采样块的浮点运算次数并加到总数中
        if self.downsample is not None:
            flops += self.downsample.flops()
        # 返回总的浮点运算次数
        return flops

    # 初始化残差块中的归一化层参数
    def _init_respostnorm(self):
        # 遍历网络中的每个块
        for blk in self.blocks:
            # 初始化第一个归一化层的偏置为0
            nn.init.constant_(blk.norm1.bias, 0)
            # 初始化第一个归一化层的权重为0
            nn.init.constant_(blk.norm1.weight, 0)
            # 初始化第二个归一化层的偏置为0
            nn.init.constant_(blk.norm2.bias, 0)
            # 初始化第二个归一化层的权重为0
            nn.init.constant_(blk.norm2.weight, 0)
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 将图片大小转换为元组形式
        patch_size = to_2tuple(patch_size)  # 将patch大小转换为元组形式
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # 计算patch的分辨率
        self.img_size = img_size  # 保存图片大小
        self.patch_size = patch_size  # 保存patch大小
        self.patches_resolution = patches_resolution  # 保存patch的分辨率
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # 计算patch的数量

        self.in_chans = in_chans  # 保存输入通道数
        self.embed_dim = embed_dim  # 保存线性投影输出通道数

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # 创建卷积层用于投影
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)  # 如果有规范化层，则创建规范化层
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape  # 获取输入张量的形状
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1],
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}."
        x = self.proj(x).flatten(2).transpose(1, 2)  # 投影输入张量并展平，然后转置
        if self.norm is not None:
            x = self.norm(x)  # 如果有规范化层，则对投影后的张量进行规范化
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution  # 获取patch的分辨率
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])  # 计算FLOPs
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim  # 如果有规范化层，则增加FLOPs
        return flops

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.  # 输入通道数
        input_resolution (tuple[int]): Input resolution.  # 输入分辨率
        depth (int): Number of blocks.  # 残差块的数量
        num_heads (int): Number of attention heads.  # 注意力头的数量
        window_size (int): Local window size.  # 局部窗口大小
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.  # MLP隐藏维度与嵌入维度的比率
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True  # 是否为查询、键、值添加可学习的偏置
        drop (float, optional): Dropout rate. Default: 0.0  # 丢弃率
        attn_drop (float, optional): Attention dropout rate. Default: 0.0  # 注意力丢弃率
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0  # 随机深度率
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm  # 标准化层
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None  # 层末的下采样层
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.  # 是否使用检查点来节省内存
        img_size: Input image size.  # 输入图像大小
        patch_size: Patch size.  # 补丁大小
        resi_connection: The convolutional block before residual connection.  # 残差连接前的卷积块
    """

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
        # 前向传播函数，通过一系列操作后返回结果

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()  # 累加残差块的浮点运算量
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9  # 计算卷积运算的浮点运算量
        flops += self.patch_embed.flops()  # 累加补丁嵌入的浮点运算量
        flops += self.patch_unembed.flops()  # 累加补丁解嵌入的浮点运算量

        return flops  # 返回总的浮点运算量
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
        # 初始化函数，设置默认参数和属性
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
        # 前向传播函数，将输入张量转换为指定形状
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        # 计算模型的浮点运算量
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        # 初始化函数，根据不同的缩放因子构建上采样模块
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
class Upsample_hf(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # 检查 scale 是否为 2^n
            for _ in range(int(math.log(scale, 2))):  # 根据 scale 计算需要的卷积层和像素重排层
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:  # 如果 scale 为 3
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')  # 抛出错误，不支持的 scale
        super(Upsample_hf, self).__init__(*m)  # 调用父类构造函数


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))  # 添加卷积层
        m.append(nn.PixelShuffle(scale))  # 添加像素重排层
        super(UpsampleOneStep, self).__init__(*m)  # 调用父类构造函数

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9  # 计算浮点运算量
        return flops



class Swin2SR(nn.Module):
    r""" Swin2SR
        A PyTorch impl of : `Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration`.
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
        drop_rate (float): 丢弃率。默认为0
        attn_drop_rate (float): 注意力丢弃率。默认为0
        drop_path_rate (float): 随机深度率。默认为0.1
        norm_layer (nn.Module): 标准化层。默认为nn.LayerNorm。
        ape (bool): 如果为True，则将绝对位置嵌入添加到补丁嵌入中。默认为False
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
            # 对LayerNorm层的偏置初始化为0，权重初始化为1.0
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # 返回不需要权重衰减的参数，这里返回'absolute_pos_embed'
        return {'absolute_pos_embed'}
    @torch.jit.ignore
    # 忽略 torch.jit 的装饰器，用于指示 JIT 编译器忽略此函数
    def no_weight_decay_keywords(self):
        # 返回一个包含 'relative_position_bias_table' 的字典，用于指定不需要进行权重衰减的关键字
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        # 获取输入张量 x 的高度和宽度
        _, _, h, w = x.size()
        # 计算需要填充的高度和宽度
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # 在输入张量 x 上进行反射填充
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        # 获取输入张量 x 的高度和宽度
        x_size = (x.shape[2], x.shape[3])
        # 对输入张量 x 进行 patch embedding
        x = self.patch_embed(x)
        # 如果使用绝对位置编码，则将其加到输入张量 x 上
        if self.ape:
            x = x + self.absolute_pos_embed
        # 对输入张量 x 进行位置丢弃
        x = self.pos_drop(x)

        # 遍历所有层并对输入张量 x 进行处理
        for layer in self.layers:
            x = layer(x, x_size)

        # 对处理后的张量 x 进行归一化
        x = self.norm(x)  # B L C
        # 对归一化后的张量 x 进行 patch unembedding
        x = self.patch_unembed(x, x_size)

        return x

    def forward_features_hf(self, x):
        # 获取输入张量 x 的高度和宽度
        x_size = (x.shape[2], x.shape[3])
        # 对输入张量 x 进行 patch embedding
        x = self.patch_embed(x)
        # 如果使用绝对位置编码，则将其加到输入张量 x 上
        if self.ape:
            x = x + self.absolute_pos_embed
        # 对输入张量 x 进行位置丢弃
        x = self.pos_drop(x)

        # 遍历所有高频层并对输入张量 x 进行处理
        for layer in self.layers_hf:
            x = layer(x, x_size)

        # 对处理后的张量 x 进行归一化
        x = self.norm(x)  # B L C
        # 对归一化后的张量 x 进行 patch unembedding
        x = self.patch_unembed(x, x_size)

        return x

    def flops(self):
        # 初始化浮点运算数
        flops = 0
        # 获取 patch 分辨率的高度和宽度
        H, W = self.patches_resolution
        # 计算 patch embedding 的浮点运算数
        flops += H * W * 3 * self.embed_dim * 9
        # 计算 patch embedding 的浮点运算数
        flops += self.patch_embed.flops()
        # 遍历所有层并计算每一层的浮点运算数
        for layer in self.layers:
            flops += layer.flops()
        # 计算 patch unembedding 的浮点运算数
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        # 计算上采样的浮点运算数
        flops += self.upsample.flops()
        return flops
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 设置放大倍数
    upscale = 4
    # 设置窗口大小
    window_size = 8
    # 计算调整后的高度
    height = (1024 // upscale // window_size + 1) * window_size
    # 计算调整后的宽度
    width = (720 // upscale // window_size + 1) * window_size
    # 创建 Swin2SR 模型对象，设置参数
    model = Swin2SR(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    # 打印模型信息
    print(model)
    # 打印调整后的高度、宽度以及模型的浮点运算量
    print(height, width, model.flops() / 1e9)

    # 生成随机输入张量
    x = torch.randn((1, 3, height, width))
    # 使用模型进行推理
    x = model(x)
    # 打印输出张量的形状
    print(x.shape)
```