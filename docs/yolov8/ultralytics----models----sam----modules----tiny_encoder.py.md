# `.\yolov8\ultralytics\models\sam\modules\tiny_encoder.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools  # 导入 itertools 库，用于迭代操作
from typing import Tuple  # 导入 Tuple 类型提示，用于指定元组类型

import torch  # 导入 PyTorch 深度学习库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 神经网络函数模块
import torch.utils.checkpoint as checkpoint  # 导入 PyTorch 检查点模块，用于内存优化

from ultralytics.utils.instance import to_2tuple  # 从 ultralytics.utils.instance 模块中导入 to_2tuple 函数


class Conv2d_BN(torch.nn.Sequential):
    """A sequential container that performs 2D convolution followed by batch normalization."""

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        """Initializes the MBConv model with given input channels, output channels, expansion ratio, activation, and
        drop path.
        """
        super().__init__()
        # 添加 2D 卷积层，不使用偏置参数
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        # 添加批归一化层，并初始化权重为 bn_weight_init，偏置为 0
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


class PatchEmbed(nn.Module):
    """Embeds images into patches and projects them into a specified embedding dimension."""

    def __init__(self, in_chans, embed_dim, resolution, activation):
        """Initialize the PatchMerging class with specified input, output dimensions, resolution and activation
        function.
        """
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        # 构建序列模型，包含两个 Conv2d_BN 层和激活函数
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),  # 第一个卷积 + 批归一化层
            activation(),  # 激活函数
            Conv2d_BN(n // 2, n, 3, 2, 1),  # 第二个卷积 + 批归一化层
        )

    def forward(self, x):
        """Runs input tensor 'x' through the PatchMerging model's sequence of operations."""
        return self.seq(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv) layer, part of the EfficientNet architecture."""
    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        """
        Initializes a convolutional layer with specified dimensions, input resolution, depth, and activation
        function.
        """
        super().__init__()
        
        # 设置输入通道数
        self.in_chans = in_chans
        # 计算隐藏层通道数，根据扩展比例
        self.hidden_chans = int(in_chans * expand_ratio)
        # 设置输出通道数
        self.out_chans = out_chans

        # 第一个卷积层，包括卷积和批归一化
        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        # 第一个激活函数，根据给定的激活函数类实例化
        self.act1 = activation()

        # 第二个卷积层，包括卷积、批归一化和分组卷积（根据隐藏通道数）
        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
        # 第二个激活函数，同样根据给定的激活函数类实例化
        self.act2 = activation()

        # 第三个卷积层，包括卷积、批归一化
        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        # 第三个激活函数，使用给定的激活函数类实例化
        self.act3 = activation()

        # 在训练时，根据是否需要进行 DropPath 操作来决定是否使用 DropPath 层
        # NOTE: `DropPath` is needed only for training.
        self.drop_path = nn.Identity()  # 如果 drop_path <= 0，使用恒等映射作为 drop_path
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Implements the forward pass for the model architecture.
        """
        # 将输入作为快捷连接（shortcut）
        shortcut = x
        # 第一层卷积操作
        x = self.conv1(x)
        # 第一层激活函数
        x = self.act1(x)
        # 第二层卷积操作
        x = self.conv2(x)
        # 第二层激活函数
        x = self.act2(x)
        # 第三层卷积操作
        x = self.conv3(x)
        # DropPath 操作（在训练时可能会对 x 进行操作）
        x = self.drop_path(x)
        # 加上快捷连接
        x += shortcut
        # 最后一层激活函数
        return self.act3(x)
class PatchMerging(nn.Module):
    """Merges neighboring patches in the feature map and projects to a new dimension."""

    def __init__(self, input_resolution, dim, out_dim, activation):
        """Initializes the PatchMerging module with specified parameters.

        Args:
            input_resolution (tuple): Resolution of the input feature map (H, W).
            dim (int): Input dimensionality of the feature map.
            out_dim (int): Output dimensionality after merging and projection.
            activation (torch.nn.Module): Activation function instance.
        """
        super().__init__()

        self.input_resolution = input_resolution  # Store input resolution (H, W)
        self.dim = dim  # Store input dimensionality
        self.out_dim = out_dim  # Store output dimensionality
        self.act = activation()  # Initialize activation function instance
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)  # 1x1 convolution layer
        stride_c = 1 if out_dim in {320, 448, 576} else 2
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)  # Depthwise separable convolution
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)  # 1x1 convolution layer

    def forward(self, x):
        """Performs forward pass through the PatchMerging module.

        Args:
            x (torch.Tensor): Input tensor, expected to have dimensions (B, C, H, W) or (B, H, W, C).

        Returns:
            torch.Tensor: Flattened and transposed tensor after convolution operations.
        """
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # Reshape input tensor to (B, C, H, W) format if initially in (B, H, W, C)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)  # Apply first convolution layer
        x = self.act(x)  # Apply activation function

        x = self.conv2(x)  # Apply second convolution layer
        x = self.act(x)  # Apply activation function
        x = self.conv3(x)  # Apply third convolution layer

        return x.flatten(2).transpose(1, 2)  # Flatten and transpose output tensor


class ConvLayer(nn.Module):
    """
    Convolutional Layer featuring multiple MobileNetV3-style inverted bottleneck convolutions (MBConv).

    Optionally applies downsample operations to the output, and provides support for gradient checkpointing.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
    ):
        """Initializes the ConvLayer module with specified parameters.

        Args:
            dim (int): Input dimensionality for the convolutional layer.
            input_resolution (tuple): Resolution of the input feature map (H, W).
            depth (int): Depth of the convolutional layer.
            activation (torch.nn.Module): Activation function instance.
            drop_path (float, optional): Dropout probability. Defaults to 0.0.
            downsample (str or None, optional): Downsample operation type. Defaults to None.
            use_checkpoint (bool, optional): Flag to use gradient checkpointing. Defaults to False.
            out_dim (int or None, optional): Output dimensionality. Defaults to None.
            conv_expand_ratio (float, optional): Expansion ratio for convolution layers. Defaults to 4.0.
        """
        super().__init__()

        # Initialize module attributes
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.activation = activation
        self.drop_path = drop_path
        self.downsample = downsample
        self.use_checkpoint = use_checkpoint
        self.out_dim = out_dim
        self.conv_expand_ratio = conv_expand_ratio
    ):
        """
        Initializes the ConvLayer with the given dimensions and settings.

        Args:
            dim (int): The dimensionality of the input and output.
            input_resolution (Tuple[int, int]): The resolution of the input image.
            depth (int): The number of MBConv layers in the block.
            activation (Callable): Activation function applied after each convolution.
            drop_path (Union[float, List[float]]): Drop path rate. Single float or a list of floats for each MBConv.
            downsample (Optional[Callable]): Function for downsampling the output. None to skip downsampling.
            use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
            out_dim (Optional[int]): The dimensionality of the output. None means it will be the same as `dim`.
            conv_expand_ratio (float): Expansion ratio for the MBConv layers.
        """
        super().__init__()  # 调用父类的初始化方法

        self.dim = dim  # 设置输入和输出的维度
        self.input_resolution = input_resolution  # 设置输入图像的分辨率
        self.depth = depth  # 设置 MBConv 块中的层数
        self.use_checkpoint = use_checkpoint  # 设置是否使用梯度检查点来节省内存

        # 构建块
        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # Patch merging layer
        self.downsample = (
            None  # 如果没有指定 downsample 函数，则设置为 None
            if downsample is None
            else downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)  # 否则调用 downsample 函数进行下采样
        )

    def forward(self, x):
        """Processes the input through a series of convolutional layers and returns the activated output."""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)  # 依次对输入 x 应用每个 MBConv 块
        return x if self.downsample is None else self.downsample(x)  # 如果定义了 downsample 函数，则对最终输出 x 进行下采样处理
        """
        Initializes the Multi-head Attention module with given parameters.

        Args:
            dim (int): Dimensionality of input embeddings.
            key_dim (int): Dimensionality of key and query vectors.
            num_heads (int, optional): Number of attention heads. Default is 8.
            attn_ratio (int, optional): Ratio of total spatial positions to use as attention biases. Default is 4.
            resolution (tuple, optional): Spatial resolution of the input. Default is (14, 14).
        """
        super().__init__()
        # Calculate the size of each head in the attention mechanism
        head_dim = key_dim // num_heads
        # Initialize the linear transformation of input into query, key, and value
        self.to_qkv = nn.Linear(dim, 3 * key_dim)
        # Cache the number of attention heads
        self.num_heads = num_heads
        # Set the spatial bias ratio for attention mechanism
        self.attn_ratio = attn_ratio
        # Determine the size of spatial grid for the attention biases
        self.resolution = resolution
        # Initialize cached attention biases for inference, to be deleted during training
        self.ab = None
    ):
        """
        Initializes the Attention module.

        Args:
            dim (int): The dimensionality of the input and output.
            key_dim (int): The dimensionality of the keys and queries.
            num_heads (int, optional): Number of attention heads. Default is 8.
            attn_ratio (float, optional): Attention ratio, affecting the dimensions of the value vectors. Default is 4.
            resolution (Tuple[int, int], optional): Spatial resolution of the input feature map. Default is (14, 14).

        Raises:
            AssertionError: If `resolution` is not a tuple of length 2.
        """
        super().__init__()

        # 检查并确保 `resolution` 是长度为 2 的元组
        assert isinstance(resolution, tuple) and len(resolution) == 2, "'resolution' argument not tuple of length 2"
        
        # 设置模块的属性
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        # 计算 `h`，作为后续线性层的输入维度
        h = self.dh + nh_kd * 2

        # Layer normalization 层
        self.norm = nn.LayerNorm(dim)
        
        # 线性变换层，将输入转换为 `h` 维度
        self.qkv = nn.Linear(dim, h)
        
        # 输出投影层，将注意力头的结果投影回 `dim` 维度
        self.proj = nn.Linear(self.dh, dim)

        # 生成所有空间位置的偏移量对应的索引
        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        
        # 初始化注意力偏置参数，并注册为模型的可学习参数
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        """Sets the module in training mode and handles attribute 'ab' based on the mode."""
        # 调用父类的 `train` 方法，设置模块的训练模式
        super().train(mode)
        
        # 根据训练模式处理 `ab` 属性
        if mode and hasattr(self, "ab"):
            del self.ab  # 如果是训练模式且存在 `ab` 属性，则删除它
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
            # 如果是测试模式或 `ab` 属性不存在，则将 `attention_biases` 与 `attention_bias_idxs` 结合起来存储在 `ab` 中
    def forward(self, x):  # x
        """对输入张量 'x' 执行前向传播，包括归一化和查询键/值操作。"""
        B, N, _ = x.shape  # B, N, C

        # 归一化处理
        x = self.norm(x)

        # 查询键值对
        qkv = self.qkv(x)
        # 将结果重塑为 (B, N, num_heads, d)，并分割为 q, k, v
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # 将维度重新排列为 (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 将 attention_biases 转移到合适的设备上
        self.ab = self.ab.to(self.attention_biases.device)

        # 计算注意力权重，包括缩放和偏置项
        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        attn = attn.softmax(dim=-1)
        
        # 计算加权后的值
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        
        # 应用投影层并返回结果
        return self.proj(x)
class TinyViTBlock(nn.Module):
    """TinyViT Block that applies self-attention and a local convolution to the input."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        activation=nn.GELU,
    ):
        """
        Initializes the TinyViTBlock.

        Args:
            dim (int): The dimensionality of the input and output.
            input_resolution (Tuple[int, int]): Spatial resolution of the input feature map.
            num_heads (int): Number of attention heads.
            window_size (int, optional): Window size for attention. Default is 7.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default is 4.
            drop (float, optional): Dropout rate. Default is 0.
            drop_path (float, optional): Stochastic depth rate. Default is 0.
            local_conv_size (int, optional): The kernel size of the local convolution. Default is 3.
            activation (torch.nn, optional): Activation function for MLP. Default is nn.GELU.

        Raises:
            AssertionError: If `window_size` is not greater than 0.
            AssertionError: If `dim` is not divisible by `num_heads`.
        """
        super().__init__()
        self.dim = dim  # 设置输入输出的维度
        self.input_resolution = input_resolution  # 设置输入特征图的空间分辨率
        self.num_heads = num_heads  # 设置注意力头的数量
        assert window_size > 0, "window_size must be greater than 0"  # 断言窗口大小必须大于0
        self.window_size = window_size  # 设置注意力机制的窗口大小
        self.mlp_ratio = mlp_ratio  # 设置MLP隐藏层维度与嵌入维度的比例

        # NOTE: `DropPath` is needed only for training.
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()  # 设置DropPath层，用于训练时的随机深度（如果drop_path大于0）

        assert dim % num_heads == 0, "dim must be divisible by num_heads"  # 断言维度必须能够被注意力头数整除
        head_dim = dim // num_heads  # 计算每个注意力头的维度

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)
        # 初始化注意力层，传入维度、头部维度、头部数量、注意力比例和窗口分辨率

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=mlp_activation, drop=drop)
        # 初始化MLP层，传入输入特征维度、隐藏层特征维度、激活函数和dropout率

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)
        # 初始化本地卷积层，传入输入和输出特征维度、卷积核大小、步长、填充、分组数
    def forward(self, x):
        """对输入的 'x' 进行基于注意力的转换或填充，然后通过本地卷积传递。

        Args:
            x (tensor): 输入张量，形状为 [batch, height*width, channels]。

        Returns:
            tensor: 经过处理后的张量，形状为 [batch, height*width, channels]。
        """
        h, w = self.input_resolution
        b, hw, c = x.shape  # batch, height*width, channels
        assert hw == h * w, "input feature has wrong size"  # 断言输入特征的尺寸是否正确
        res_x = x  # 保留原始输入张量

        # 如果输入分辨率等于窗口尺寸，则直接应用注意力模块
        if h == self.window_size and w == self.window_size:
            x = self.attn(x)
        else:
            # 否则，对输入进行重塑以便进行填充
            x = x.view(b, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0  # 检查是否需要填充

            # 如果需要填充，则对输入进行填充操作
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = h + pad_b, w + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size

            # 窗口分割
            x = (
                x.view(b, nH, self.window_size, nW, self.window_size, c)
                .transpose(2, 3)
                .reshape(b * nH * nW, self.window_size * self.window_size, c)
            )
            x = self.attn(x)  # 应用注意力模块

            # 窗口重组
            x = x.view(b, nH, nW, self.window_size, self.window_size, c).transpose(2, 3).reshape(b, pH, pW, c)
            if padding:
                x = x[:, :h, :w].contiguous()  # 移除填充部分

            x = x.view(b, hw, c)  # 恢复原始形状

        x = res_x + self.drop_path(x)  # 加入残差连接和DropPath操作
        x = x.transpose(1, 2).reshape(b, c, h, w)  # 转置和重塑张量形状
        x = self.local_conv(x)  # 应用本地卷积
        x = x.view(b, c, hw).transpose(1, 2)  # 重塑张量形状

        return x + self.drop_path(self.mlp(x))  # 加入残差连接和MLP操作

    def extra_repr(self) -> str:
        """返回一个格式化的字符串，表示TinyViTBlock的参数：维度、输入分辨率、注意力头数、窗口尺寸和MLP比例。

        Returns:
            str: 格式化后的参数信息字符串。
        """
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )
# 定义一个名为 BasicLayer 的类，用于 TinyViT 架构中的一个阶段的基本层次
class BasicLayer(nn.Module):
    """A basic TinyViT layer for one stage in a TinyViT architecture."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        local_conv_size=3,
        activation=nn.GELU,
        out_dim=None,
    ):
        """
        Initializes the BasicLayer.

        Args:
            dim (int): The dimensionality of the input and output.
            input_resolution (Tuple[int, int]): Spatial resolution of the input feature map.
            depth (int): Number of TinyViT blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default is 4.
            drop (float, optional): Dropout rate. Default is 0.
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default is 0.
            downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default is None.
            use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Default is False.
            local_conv_size (int, optional): Kernel size of the local convolution. Default is 3.
            activation (torch.nn, optional): Activation function for MLP. Default is nn.GELU.
            out_dim (int | None, optional): The output dimension of the layer. Default is None.

        Raises:
            ValueError: If `drop_path` is a list of float but its length doesn't match `depth`.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 设置类的属性
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建 TinyViTBlock 组成的模块列表
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    local_conv_size=local_conv_size,
                    activation=activation,
                )
                for i in range(depth)  # 根据 depth 参数循环创建 TinyViTBlock
            ]
        )

        # 如果指定了 downsample 参数，则创建对应的下采样层
        self.downsample = (
            None
            if downsample is None
            else downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        )
    # 执行输入张量的前向传播，并返回一个规范化的张量
    def forward(self, x):
        # 遍历网络中的每个块进行前向传播
        for blk in self.blocks:
            # 如果使用了检查点技术，则通过检查点执行块的前向传播，否则直接调用块的前向传播
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        # 如果存在下采样函数，则对输出张量进行下采样操作
        return x if self.downsample is None else self.downsample(x)

    # 返回一个描述层参数的字符串表示形式
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
    class LayerNorm2d(nn.Module):
        """A PyTorch implementation of Layer Normalization in 2D."""

        def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
            """Initialize LayerNorm2d with the number of channels and an optional epsilon."""
            super().__init__()
            # Define learnable parameters for scaling and shifting
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Perform a forward pass, normalizing the input tensor."""
            # Compute mean and standard deviation across channels
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            # Normalize the input tensor
            x = (x - u) / torch.sqrt(s + self.eps)
            # Scale and shift the normalized tensor
            return self.weight[:, None, None] * x + self.bias[:, None, None]


    class TinyViT(nn.Module):
        """
        The TinyViT architecture for vision tasks.

        Attributes:
            img_size (int): Input image size.
            in_chans (int): Number of input channels.
            num_classes (int): Number of classification classes.
            embed_dims (List[int]): List of embedding dimensions for each layer.
            depths (List[int]): List of depths for each layer.
            num_heads (List[int]): List of number of attention heads for each layer.
            window_sizes (List[int]): List of window sizes for each layer.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            drop_rate (float): Dropout rate for drop layers.
            drop_path_rate (float): Drop path rate for stochastic depth.
            use_checkpoint (bool): Use checkpointing for efficient memory usage.
            mbconv_expand_ratio (float): Expansion ratio for MBConv layer.
            local_conv_size (int): Local convolution kernel size.
            layer_lr_decay (float): Layer-wise learning rate decay.

        Note:
            This implementation is generalized to accept a list of depths, attention heads,
            embedding dimensions and window sizes, which allows you to create a
            "stack" of TinyViT models of varying configurations.
        """

        def __init__(
            self,
            img_size=224,
            in_chans=3,
            num_classes=1000,
            embed_dims=(96, 192, 384, 768),
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_sizes=(7, 7, 14, 7),
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.1,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=1.0,
    def set_layer_lr_decay(self, layer_lr_decay):
        """Sets the learning rate decay for each layer in the TinyViT model."""
        decay_rate = layer_lr_decay  # 设置每个层的学习率衰减率

        # Layers -> blocks (depth)
        depth = sum(self.depths)  # 计算总的层深度
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]  # 计算每个层的学习率缩放比例

        def _set_lr_scale(m, scale):
            """Sets the learning rate scale for each layer in the model based on the layer's depth."""
            for p in m.parameters():
                p.lr_scale = scale  # 设置每个模型层的学习率缩放比例

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))  # 对输入嵌入层设置学习率缩放比例
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))  # 对每个层块设置学习率缩放比例
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))  # 对下采样层设置学习率缩放比例
        assert i == depth  # 确保设置了所有层的学习率缩放比例
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))  # 对归一化头部和头部设置学习率缩放比例

        for k, p in self.named_parameters():
            p.param_name = k  # 为每个参数设置参数名属性

        def _check_lr_scale(m):
            """Checks if the learning rate scale attribute is present in module's parameters."""
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name  # 检查模块参数中是否存在学习率缩放属性

        self.apply(_check_lr_scale)  # 应用检查学习率缩放属性的函数到模型中的所有模块

    def _init_weights(self, m):
        """Initializes weights for linear layers and layer normalization in the given module."""
        if isinstance(m, nn.Linear):
            # NOTE: This initialization is needed only for training.
            # trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 初始化线性层的偏置为常数0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # 初始化层归一化的偏置为常数0
            nn.init.constant_(m.weight, 1.0)  # 初始化层归一化的权重为常数1.0

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Returns a dictionary of parameter names where weight decay should not be applied."""
        return {"attention_biases"}  # 返回不应用权重衰减的参数名称字典

    def forward_features(self, x):
        """Runs the input through the model layers and returns the transformed output."""
        x = self.patch_embed(x)  # x input is (N, C, H, W)

        x = self.layers[0](x)  # 对输入应用第一个层
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)  # 依次对每个层应用输入
        batch, _, channel = x.shape
        x = x.view(batch, 64, 64, channel)  # 调整输出的形状
        x = x.permute(0, 3, 1, 2)  # 调整输出的维度顺序
        return self.neck(x)  # 返回经过颈部处理后的输出

    def forward(self, x):
        """Executes a forward pass on the input tensor through the constructed model layers."""
        return self.forward_features(x)  # 执行输入张量通过构建模型层的前向传播
```