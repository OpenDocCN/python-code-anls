# `.\yolov8\ultralytics\models\sam\modules\encoders.py`

```py
# 导入所需模块和类，包括通用数据类型和模型定义
from typing import Any, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入自定义模块
from ultralytics.nn.modules import LayerNorm2d, MLPBlock

# 定义一个基于Vision Transformer（ViT）架构的图像编码器类
class ImageEncoderViT(nn.Module):
    """
    An image encoder using Vision Transformer (ViT) architecture for encoding an image into a compact latent space. The
    encoder takes an image, splits it into patches, and processes these patches through a series of transformer blocks.
    The encoded patches are then processed through a neck to generate the final encoded representation.

    This class and its supporting functions below lightly adapted from the ViTDet backbone available at
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py.

    Attributes:
        img_size (int): Dimension of input images, assumed to be square.
        patch_embed (PatchEmbed): Module for patch embedding.
        pos_embed (nn.Parameter, optional): Absolute positional embedding for patches.
        blocks (nn.ModuleList): List of transformer blocks for processing patch embeddings.
        neck (nn.Sequential): Neck module to further process the output.
    """

    def __init__(
        self,
        img_size: int = 1024,             # 输入图像的尺寸，默认为1024x1024像素
        patch_size: int = 16,            # 每个patch的尺寸，默认为16x16像素
        in_chans: int = 3,               # 输入图像的通道数，默认为RGB三通道
        embed_dim: int = 768,            # 嵌入维度，每个patch的嵌入维度，默认为768
        depth: int = 12,                 # Transformer块的深度（层数），默认为12层
        num_heads: int = 12,             # 注意力头的数量，默认为12个
        mlp_ratio: float = 4.0,          # MLP（多层感知机）部分的维度扩展比例，默认为4.0
        out_chans: int = 256,            # 输出通道数，默认为256
        qkv_bias: bool = True,           # 是否允许注意力机制中的查询、键、值偏置，默认为True
        norm_layer: Type[nn.Module] = nn.LayerNorm,  # 规范化层类型，默认为LayerNorm
        act_layer: Type[nn.Module] = nn.GELU,        # 激活函数类型，默认为GELU
        use_abs_pos: bool = True,        # 是否使用绝对位置编码，默认为True
        use_rel_pos: bool = False,       # 是否使用相对位置编码，默认为False
        rel_pos_zero_init: bool = True,  # 相对位置编码是否零初始化，默认为True
        window_size: int = 0,            # 窗口大小，用于局部注意力机制，默认为0表示全局注意力
        global_attn_indexes: Tuple[int, ...] = (),  # 全局注意力的索引列表，默认为空元组
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size  # 设置输入图像尺寸

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),  # 设置 patch 的大小
            stride=(patch_size, patch_size),  # 设置 patch 的步长
            in_chans=in_chans,  # 设置输入图像的通道数
            embed_dim=embed_dim,  # 设置 patch 嵌入的维度
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # 如果使用绝对位置编码，则初始化绝对位置嵌入，大小为预训练图像大小除以 patch 大小
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,  # 设置块的维度
                num_heads=num_heads,  # 设置块中注意力头的数量
                mlp_ratio=mlp_ratio,  # 设置 MLP 隐藏层维度与嵌入维度的比率
                qkv_bias=qkv_bias,  # 设置是否在查询、键、值上添加可学习偏置
                norm_layer=norm_layer,  # 设置归一化层
                act_layer=act_layer,  # 设置激活函数层
                use_rel_pos=use_rel_pos,  # 设置是否使用相对位置编码
                rel_pos_zero_init=rel_pos_zero_init,  # 设置是否将相对位置参数初始化为零
                window_size=window_size if i not in global_attn_indexes else 0,  # 设置窗口注意力块的窗口大小
                input_size=(img_size // patch_size, img_size // patch_size),  # 设置输入块的大小
            )
            self.blocks.append(block)  # 将块添加到模块列表中

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,  # 输入通道数为嵌入维度
                out_chans,  # 输出通道数为指定的输出通道数
                kernel_size=1,  # 设置卷积核大小为1x1
                bias=False,  # 不使用偏置
            ),
            LayerNorm2d(out_chans),  # 应用输出通道数的层归一化
            nn.Conv2d(
                out_chans,  # 输入通道数为上一层的输出通道数
                out_chans,  # 输出通道数为上一层的输出通道数
                kernel_size=3,  # 设置卷积核大小为3x3
                padding=1,  # 使用填充大小为1
                bias=False,  # 不使用偏置
            ),
            LayerNorm2d(out_chans),  # 应用输出通道数的层归一化
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through patch embedding, applies positional embedding if present, and passes through blocks
        and neck.
        """
        # 使用 patch embedding 处理输入 x，将其转换为特定形状
        x = self.patch_embed(x)
        
        # 如果存在位置编码，则将其加到 x 上
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        # 逐个应用每个块（block）
        for blk in self.blocks:
            x = blk(x)
        
        # 将张量维度进行置换，通常是为了适应后续操作的需要
        return self.neck(x.permute(0, 3, 1, 2))
class PromptEncoder(nn.Module):
    """
    Encodes different types of prompts, including points, boxes, and masks, for input to SAM's mask decoder. The encoder
    produces both sparse and dense embeddings for the input prompts.

    Attributes:
        embed_dim (int): Dimension of the embeddings.
        input_image_size (Tuple[int, int]): Size of the input image as (H, W).
        image_embedding_size (Tuple[int, int]): Spatial size of the image embedding as (H, W).
        pe_layer (PositionEmbeddingRandom): Module for random position embedding.
        num_point_embeddings (int): Number of point embeddings for different types of points.
        point_embeddings (nn.ModuleList): List of point embeddings.
        not_a_point_embed (nn.Embedding): Embedding for points that are not a part of any label.
        mask_input_size (Tuple[int, int]): Size of the input mask.
        mask_downscaling (nn.Sequential): Neural network for downscaling the mask.
        no_mask_embed (nn.Embedding): Embedding for cases where no mask is provided.
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Args:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim  # 存储嵌入维度信息
        self.input_image_size = input_image_size  # 存储输入图像的大小信息
        self.image_embedding_size = image_embedding_size  # 存储图像嵌入的空间尺寸信息
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)  # 创建一个随机位置编码模块

        self.num_point_embeddings: int = 4  # 点嵌入的数量，包括正负点和两个框角点
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)  # 创建点嵌入的列表
        self.not_a_point_embed = nn.Embedding(1, embed_dim)  # 创建一个用于不属于任何标签的点的嵌入

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])  # 设置输入掩码的大小
        self.mask_downscaling = nn.Sequential(  # 创建用于缩小掩码的神经网络序列
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),  # 第一层卷积层，将输入通道数缩小到 mask_in_chans // 4
            LayerNorm2d(mask_in_chans // 4),  # 对通道进行归一化
            activation(),  # 应用激活函数
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),  # 第二层卷积层，将通道数扩展回 mask_in_chans
            LayerNorm2d(mask_in_chans),  # 对通道进行归一化
            activation(),  # 应用激活函数
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),  # 最后一层卷积层，将通道数减少到嵌入维度
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)  # 创建一个用于没有提供掩码的情况的嵌入
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts, applied to a dense set of points the shape of the
        image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape 1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        # 调用位置编码层，返回对图像编码大小的位置编码张量，并在第一维度上增加一个维度
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        # 将点的坐标向中心偏移0.5，以便准确表示像素中心
        points = points + 0.5  # Shift to center of pixel
        if pad:
            # 如果需要填充，则创建零填充点和负标签
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            # 在点和标签的第一维度上连接填充点和填充标签
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        # 使用位置编码层将点坐标嵌入到输入图像大小中
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # 将标签为-1的点嵌入设为0
        point_embedding[labels == -1] = 0.0
        # 将标签为-1的点嵌入增加not_a_point_embed的权重
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        # 将标签为0的点嵌入增加point_embeddings[0]的权重
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        # 将标签为1的点嵌入增加point_embeddings[1]的权重
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        # 将框的坐标向中心偏移0.5，以便准确表示像素中心
        boxes = boxes + 0.5  # Shift to center of pixel
        # 将框的坐标重塑为(-1, 2, 2)的形状
        coords = boxes.reshape(-1, 2, 2)
        # 使用位置编码层将框的角坐标嵌入到输入图像大小中
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        # 将每个框的第一个角点的嵌入增加point_embeddings[2]的权重
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        # 将每个框的第二个角点的嵌入增加point_embeddings[3]的权重
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        # 使用mask_downscaling函数对输入的masks进行嵌入
        return self.mask_downscaling(masks)

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """Gets the batch size of the output given the batch size of the input prompts."""
        if points is not None:
            # 如果有点的输入，则返回点的批量大小
            return points[0].shape[0]
        elif boxes is not None:
            # 如果有框的输入，则返回框的批量大小
            return boxes.shape[0]
        elif masks is not None:
            # 如果有mask的输入，则返回mask的批量大小
            return masks.shape[0]
        else:
            # 如果没有输入，则默认返回批量大小为1
            return 1

    def _get_device(self) -> torch.device:
        """Returns the device of the first point embedding's weight tensor."""
        # 返回第一个点嵌入权重张量所在的设备
        return self.point_embeddings[0].weight.device
    # 定义函数签名及文档字符串，说明函数用途和返回值
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
          points (tuple(torch.Tensor, torch.Tensor), None): point coordinates and labels to embed.
              如果不为 None，则包含点的坐标和标签的元组
          boxes (torch.Tensor, None): boxes to embed
              如果不为 None，则包含要嵌入的框的张量
          masks (torch.Tensor, None): masks to embed
              如果不为 None，则包含要嵌入的掩码的张量

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape BxNx(embed_dim), where N is determined
            by the number of input points and boxes.
              稀疏嵌入（sparse embeddings）用于点和框，形状为 BxNx(embed_dim)，其中 N 取决于输入点和框的数量。
          torch.Tensor: dense embeddings for the masks, in the shape Bx(embed_dim)x(embed_H)x(embed_W)
              密集嵌入（dense embeddings）用于掩码，形状为 Bx(embed_dim)x(embed_H)x(embed_W)，其中 B 是批大小，embed_dim 是嵌入维度，embed_H 和 embed_W 是图像嵌入的高度和宽度。
        """
        # 获取批大小
        bs = self._get_batch_size(points, boxes, masks)
        # 初始化稀疏嵌入张量，形状为 (批大小, 0, embed_dim)，使用与设备相关的空设备
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        
        # 如果 points 不为 None，则嵌入点的坐标和标签
        if points is not None:
            coords, labels = points
            # 调用 _embed_points 方法嵌入点，根据 boxes 是否为 None 进行填充
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            # 在稀疏嵌入张量中拼接点的嵌入结果，按维度 1 连接
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        
        # 如果 boxes 不为 None，则嵌入框
        if boxes is not None:
            # 调用 _embed_boxes 方法嵌入框
            box_embeddings = self._embed_boxes(boxes)
            # 在稀疏嵌入张量中拼接框的嵌入结果，按维度 1 连接
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # 如果 masks 不为 None，则嵌入掩码
        if masks is not None:
            # 调用 _embed_masks 方法嵌入掩码
            dense_embeddings = self._embed_masks(masks)
        else:
            # 否则，使用预设的无掩码嵌入权重，重塑形状以匹配指定的图像嵌入大小
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        # 返回稀疏嵌入张量和密集嵌入张量
        return sparse_embeddings, dense_embeddings
class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies."""

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        """Initializes a position embedding using random spatial frequencies."""
        super().__init__()
        # 如果未指定或指定的尺度小于等于0，则将尺度设置为1.0
        if scale is None or scale <= 0.0:
            scale = 1.0
        # 创建一个随机高斯矩阵，并将其缩放到指定的尺度
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

        # 设置非确定性以避免前向传播时出现'cumsum_cuda_kernel does not have a deterministic implementation'错误
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # 假设坐标在[0, 1]^2的范围内，并且具有形状为 d_1 x ... x d_n x 2
        coords = 2 * coords - 1  # 将坐标映射到[-1, 1]^2范围内
        coords = coords @ self.positional_encoding_gaussian_matrix  # 用高斯矩阵对坐标进行编码
        coords = 2 * np.pi * coords  # 对编码后的坐标进行标准化
        # 输出形状为 d_1 x ... x d_n x C
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # 返回正弦和余弦的拼接结果

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)  # 创建一个大小为 h x w 的全1张量
        y_embed = grid.cumsum(dim=0) - 0.5  # 在垂直方向上累积和并偏移0.5
        x_embed = grid.cumsum(dim=1) - 0.5  # 在水平方向上累积和并偏移0.5
        y_embed = y_embed / h  # 归一化垂直方向的编码
        x_embed = x_embed / w  # 归一化水平方向的编码

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))  # 对编码后的坐标进行位置编码
        return pe.permute(2, 0, 1)  # 返回形状为 C x H x W 的编码结果

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # 将 x 坐标归一化到 [0, 1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]  # 将 y 坐标归一化到 [0, 1]
        return self._pe_encoding(coords.to(torch.float))  # 对归一化后的坐标进行位置编码，返回 B x N x C 的结果


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks."""

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
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.ReLU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = False,
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
            input_size (tuple(int, int), None): Input resolution for calculating the relative
                positional parameter size.
        """
        # Initialize the transformer block with parameters and layers
        super().__init__()
        # Layer normalization for the input data
        self.norm1 = norm_layer(dim)
        # Attention mechanism initialization
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        # Layer normalization after the attention mechanism
        self.norm2 = norm_layer(dim)
        # Multi-layer perceptron (MLP) block initialization
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        # Store the window size parameter
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes a forward pass through the transformer block with window attention and non-overlapping windows."""
        # Store the input tensor for residual connection
        shortcut = x
        # Layer normalization on the input tensor
        x = self.norm1(x)
        # Perform window partitioning if the window size is greater than 0
        if self.window_size > 0:
            # Retrieve dimensions H (height) and W (width) from the input tensor
            H, W = x.shape[1], x.shape[2]
            # Partition the input tensor into windows and calculate padding
            x, pad_hw = window_partition(x, self.window_size)

        # Apply attention mechanism on the input tensor
        x = self.attn(x)

        # Reverse window partitioning if the window size is greater than 0
        if self.window_size > 0:
            # Unpartition the tensor, using stored parameters
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # Add the shortcut connection to the transformed tensor
        x = shortcut + x
        # Apply layer normalization, MLP block, and return the transformed tensor
        return x + self.mlp(self.norm2(x))
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
        Initialize Attention module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int), None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Linear transformation for queries, keys, and values
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Linear transformation for projecting output
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            # Initialize relative positional embeddings for attention mechanism
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the forward operation including attention, normalization, MLP, and indexing within window limits."""
        B, H, W, _ = x.shape
        # Linear transformation for queries, keys, and values with reshaping and permutation
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # Separate queries, keys, and values
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # Compute attention scores
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            # Incorporate relative positional embeddings into attention scores
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        # Apply softmax to compute attention weights
        attn = attn.softmax(dim=-1)
        # Compute weighted sum of values based on attention weights
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        # Project output back to original dimension
        return self.proj(x)


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

    # Calculate padding required to make dimensions divisible by window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        # Pad the input tensor along height and width dimensions
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    # 计算添加填充后的高度和宽度
    Hp, Wp = H + pad_h, W + pad_w
    
    # 将输入张量 x 重新视图为多个窗口，每个窗口大小为 window_size x window_size，按照指定顺序重新排列
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    
    # 对重新排列后的张量进行维度置换，使得窗口的维度顺序为 (batch_size, rows, cols, window_size, window_size, channels)，并确保张量连续性
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    
    # 返回重组后的窗口数据以及添加填充后的高度和宽度元组
    return windows, (Hp, Wp)
def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from mvitv2 paper at
    https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py.

    Args:
        attn (Tensor): attention map.
            输入参数，表示注意力图的张量。

        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            输入参数，表示注意力层中的查询 q，形状为 (B, q_h * q_w, C)。

        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
            输入参数，表示高度轴上的相对位置嵌入，形状为 (Lh, C)。

        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
            输入参数，表示宽度轴上的相对位置嵌入，形状为 (Lw, C)。

        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            输入参数，表示查询 q 的空间序列大小，形状为 (q_h, q_w)。

        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
            输入参数，表示键 k 的空间序列大小，形状为 (k_h, k_w)。

    Returns:
        torch.Tensor: Updated attention map with decomposed relative positional embeddings.
            返回值，更新后的注意力图，包含分解后的相对位置嵌入。
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]
    # 获取查询和键的尺寸
    q_h, q_w = q_size
    k_h, k_w = k_size

    # 获取相对位置编码的矩阵
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    # 获取输入张量的批大小、通道数和维度
    B, _, dim = q.shape

    # 将查询张量重塑为四维张量，以便进行相对位置编码计算
    r_q = q.reshape(B, q_h, q_w, dim)

    # 使用 Einstein Summation Notation 计算高度方向和宽度方向的相对位置编码
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    # 将注意力张量重塑为期望的形状，并添加高度和宽度方向的相对位置编码
    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w
    )

    # 返回添加了相对位置编码的注意力张量
    return attn
class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),  # 定义卷积核大小为 (16, 16)
        stride: Tuple[int, int] = (16, 16),  # 定义卷积步长为 (16, 16)
        padding: Tuple[int, int] = (0, 0),  # 定义卷积填充为 (0, 0)
        in_chans: int = 3,  # 输入图片的通道数为 3
        embed_dim: int = 768,  # 嵌入维度为 768
    ) -> None:
        """
        Initialize PatchEmbed module.

        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        # 使用 nn.Conv2d 定义一个卷积层，用于将输入图像转换成嵌入表示
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes patch embedding by applying convolution and transposing resulting tensor."""
        # 将输入张量 x 经过卷积层 self.proj 处理，并对结果进行维度置换
        return self.proj(x).permute(0, 2, 3, 1)  # B C H W -> B H W C
```