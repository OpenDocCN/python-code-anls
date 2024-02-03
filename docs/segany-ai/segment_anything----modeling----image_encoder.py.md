# `.\segment-anything\segment_anything\modeling\image_encoder.py`

```
# 导入 torch 库中的相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入类型提示相关模块
from typing import Optional, Tuple, Type

# 从当前目录下的 common 模块中导入 LayerNorm2d 和 MLPBlock 类
from .common import LayerNorm2d, MLPBlock

# 定义一个名为 ImageEncoderViT 的类，继承自 nn.Module
class ImageEncoderViT(nn.Module):
    # 初始化函数，设置默认参数值
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    # 前向传播函数，接受输入张量 x，返回处理后的张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入张量进行 patch embedding
        x = self.patch_embed(x)
        # 如果存在位置编码，则将其加到输入张量上
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # 遍历所有 Transformer 块并对输入张量进行处理
        for blk in self.blocks:
            x = blk(x)

        # 对处理后的张量进行维度变换
        x = self.neck(x.permute(0, 3, 1, 2))

        # 返回处理后的张量
        return x

# 定义一个名为 Block 的类，继承自 nn.Module
class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    # 初始化函数，设置默认参数值
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        # 初始化函数，继承父类的初始化方法
        super().__init__()
        # 初始化第一个归一化层
        self.norm1 = norm_layer(dim)
        # 初始化注意力机制
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        # 初始化第二个归一化层
        self.norm2 = norm_layer(dim)
        # 初始化多层感知机块
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        # 设置窗口大小
        self.window_size = window_size

    # 前向传播函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存输入的快捷连接
        shortcut = x
        # 对输入进行第一个归一化
        x = self.norm1(x)
        # 如果窗口大小大于0，则进行窗口分区
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        # 进行注意力计算
        x = self.attn(x)
        # 如果窗口大小大于0，则进行反向窗口分区
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # 添加快捷连接并进行残差连接
        x = shortcut + x
        # 对残差连接进行第二个归一化和多层感知机块处理
        x = x + self.mlp(self.norm2(x))

        return x
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        # 初始化函数，设置注意力模块的参数
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # 创建线性层，用于计算查询、键、值的权重
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 创建线性层，用于将多头注意力的结果进行投影
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # 初始化相对位置编码的参数
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
    # 定义前向传播函数，输入为张量 x，输出为张量 x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入张量 x 的形状信息，分别为批大小 B，高度 H，宽度 W，通道数 _
        B, H, W, _ = x.shape
        # 使用 self.qkv 对象对输入张量 x 进行 qkv 计算，并重塑形状为 (3, B, nHead, H * W, C)，再进行维度置换
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # 将 qkv 重塑为 q, k, v，形状为 (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        
        # 计算注意力矩阵 attn，包括 q 乘以缩放因子 self.scale 与 k 转置后的矩阵相乘
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # 如果使用相对位置编码，调用 add_decomposed_rel_pos 函数添加相对位置信息
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        # 对注意力矩阵 attn 进行 softmax 操作
        attn = attn.softmax(dim=-1)
        # 计算加权后的值，并重塑形状为 (B, nHead, H, W, -1)，再进行维度置换
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        # 对加权后的值进行线性变换
        x = self.proj(x)

        # 返回处理后的张量 x
        return x
# 将输入张量分割为不重叠的窗口，如果需要则进行填充
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    # 计算需要填充的高度和宽度
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    # 如果需要填充，则在高度和宽度上进行填充
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    # 将输入张量重新形状为窗口大小的形状
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    # 调整维度顺序以得到期望的窗口形状
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


# 将窗口重新组合成原始序列并去除填充
def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    # 将窗口重新组合成原始序列
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    # 如果填充的高度或宽度大于原始高度或宽度，则进行裁剪
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


# 获取相对位置编码
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    # 根据查询和键的大小获取相对位置嵌入
    # 参数:
    #   q_size (int): 查询 q 的大小
    #   k_size (int): 键 k 的大小
    #   rel_pos (Tensor): 相对位置嵌入 (L, C)
    # 返回:
    #   根据相对位置提取的位置嵌入
    """
    # 计算最大相对距离
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    
    # 如果 rel_pos 的形状不等于最大相对距离，则插值 rel_pos
    if rel_pos.shape[0] != max_rel_dist:
        # 插值 rel_pos
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # 如果 q 和 k 的形状不同，则使用较短长度的坐标进行缩放
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]
# 定义一个函数，用于计算添加了分解的相对位置嵌入的注意力图
def add_decomposed_rel_pos(
    attn: torch.Tensor,  # 注意力图
    q: torch.Tensor,  # 查询 q 在注意力层中的形状为 (B, q_h * q_w, C)
    rel_pos_h: torch.Tensor,  # 高度轴的相对位置嵌入 (Lh, C)
    rel_pos_w: torch.Tensor,  # 宽度轴的相对位置嵌入 (Lw, C)
    q_size: Tuple[int, int],  # 查询 q 的空间序列大小 (q_h, q_w)
    k_size: Tuple[int, int],  # 键 k 的空间序列大小 (k_h, k_w)
) -> torch.Tensor:  # 返回值为注意力图
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    # 获取高度轴的相对位置嵌入
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    # 获取宽度轴的相对位置嵌入
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    # 计算高度轴的相对位置嵌入
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    # 计算宽度轴的相对位置嵌入
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    # 将相对位置嵌入添加到注意力图中
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


# 定义一个类，用于将图像转换为补丁嵌入
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),  # 卷积核大小
        stride: Tuple[int, int] = (16, 16),  # 步幅
        padding: Tuple[int, int] = (0, 0),  # 填充
        in_chans: int = 3,  # 输入通道数
        embed_dim: int = 768,  # 嵌入维度
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        # 初始化函数，继承父类的初始化方法
        super().__init__()

        # 创建卷积层，将输入通道数转换为嵌入维度
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入数据通过卷积层处理
        x = self.proj(x)
        # 调整数据维度顺序，从 B C H W 调整为 B H W C
        x = x.permute(0, 2, 3, 1)
        return x
```