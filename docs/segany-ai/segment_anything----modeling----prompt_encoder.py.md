# `.\segment-anything\segment_anything\modeling\prompt_encoder.py`

```
# 导入所需的库
import numpy as np
import torch
from torch import nn

# 导入类型提示相关的库
from typing import Any, Optional, Tuple, Type

# 从当前目录下的 common 模块中导入 LayerNorm2d 类
from .common import LayerNorm2d

# 定义 PromptEncoder 类，继承自 nn.Module
class PromptEncoder(nn.Module):
    # 初始化方法
    def __init__(
        self,
        embed_dim: int,  # 嵌入维度
        image_embedding_size: Tuple[int, int],  # 图像嵌入大小
        input_image_size: Tuple[int, int],  # 输入图像大小
        mask_in_chans: int,  # 掩码通道数
        activation: Type[nn.Module] = nn.GELU,  # 激活函数，默认为 GELU
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
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
        # 初始化类的实例
        super().__init__()
        # 设置属性值
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        # 创建位置编码层
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # 定义点的嵌入数量
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        # 创建点的嵌入层
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        # 创建非点的嵌入层
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # 设置输入 mask 的大小
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        # 创建 mask 下采样层
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        # 创建无 mask 的嵌入层
        self.no_mask_embed = nn.Embedding(1, embed_dim)
    def get_dense_pe(self) -> torch.Tensor:
        """
        返回用于编码点提示的位置编码，应用于图像编码形状的密集点集。

        Returns:
          torch.Tensor: 具有形状1x(embed_dim)x(embedding_h)x(embedding_w)的位置编码
        """
        返回通过self.pe_layer对self.image_embedding_size进行处理后的张量，然后在第0维度上增加一个维度

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """嵌入点提示。"""
        将点坐标加上0.5，将点移到像素中心
        如果pad为True：
            创建用于填充的零点和标签
            在点和标签的第1维度上连接填充点
        使用self.pe_layer.forward_with_coords对点进行编码，传入self.input_image_size
        将标签为-1的点编码设为0.0
        将标签为-1的点编码加上self.not_a_point_embed.weight
        将标签为0的点编码加上self.point_embeddings[0].weight
        将标签为1的点编码加上self.point_embeddings[1].weight
        返回点编码

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """嵌入框提示。"""
        将框坐标加上0.5，将框移到像素中心
        将框reshape为(-1, 2, 2)
        使用self.pe_layer.forward_with_coords对角点进行编码，传入self.input_image_size
        将第一维度为0的角点编码加上self.point_embeddings[2].weight
        将第一维度为1的角点编码加上self.point_embeddings[3].weight
        返回角点编码

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """嵌入掩码输入。"""
        使用self.mask_downscaling对掩码进行编码
        返回掩码编码
    # 获取输出的批量大小，根据输入提示的批量大小
    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        # 如果点不为空，则返回点的批量大小
        if points is not None:
            return points[0].shape[0]
        # 如果框不为空，则返回框的批量大小
        elif boxes is not None:
            return boxes.shape[0]
        # 如果掩码不为空，则返回掩码的批量大小
        elif masks is not None:
            return masks.shape[0]
        # 如果都为空，则返回默认批量大小为1
        else:
            return 1

    # 获取设备信息
    def _get_device(self) -> torch.device:
        # 返回点嵌入的设备信息
        return self.point_embeddings[0].weight.device

    # 前向传播函数
    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        # 获取批次大小
        bs = self._get_batch_size(points, boxes, masks)
        # 初始化稀疏嵌入张量
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        # 如果存在点坐标和标签
        if points is not None:
            coords, labels = points
            # 嵌入点坐标和标签
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            # 将点嵌入拼接到稀疏嵌入张量中
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        # 如果存在框
        if boxes is not None:
            # 嵌入框
            box_embeddings = self._embed_boxes(boxes)
            # 将框嵌入拼接到稀疏嵌入张量中
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # 如果存在掩码
        if masks is not None:
            # 嵌入掩码
            dense_embeddings = self._embed_masks(masks)
        else:
            # 使用预定义的权重嵌入掩码
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        # 返回稀疏和密集嵌入张量
        return sparse_embeddings, dense_embeddings
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        # 初始化函数，设置位置编码的参数
        super().__init__()
        # 如果未指定缩放因子或缩放因子小于等于0，则设为1.0
        if scale is None or scale <= 0.0:
            scale = 1.0
        # 创建一个缓冲区，存储高斯矩阵作为位置编码
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # 假设坐标在[0,1]的范围内，将坐标映射到[-1,1]范围
        coords = 2 * coords - 1
        # 使用高斯矩阵对坐标进行编码
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # 返回正弦和余弦编码后的坐标
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        # 生成指定大小的网格
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        # 对坐标进行编码并返回结果
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        # 对未归一化的坐标进行编码
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
```