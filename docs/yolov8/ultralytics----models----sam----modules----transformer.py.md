# `.\yolov8\ultralytics\models\sam\modules\transformer.py`

```py
# 导入所需的库
import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn

# 导入自定义模块MLPBlock
from ultralytics.nn.modules import MLPBlock

# 定义一个名为TwoWayTransformer的神经网络模块，继承自nn.Module类
class TwoWayTransformer(nn.Module):
    """
    A Two-Way Transformer module that enables the simultaneous attention to both image and query points. This class
    serves as a specialized transformer decoder that attends to an input image using queries whose positional embedding
    is supplied. This is particularly useful for tasks like object detection, image segmentation, and point cloud
    processing.

    Attributes:
        depth (int): The number of layers in the transformer.
        embedding_dim (int): The channel dimension for the input embeddings.
        num_heads (int): The number of heads for multihead attention.
        mlp_dim (int): The internal channel dimension for the MLP block.
        layers (nn.ModuleList): The list of TwoWayAttentionBlock layers that make up the transformer.
        final_attn_token_to_image (Attention): The final attention layer applied from the queries to the image.
        norm_final_attn (nn.LayerNorm): The layer normalization applied to the final queries.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
          attention_downsample_rate (int): downsample rate for attention
        """
        super().__init__()
        # 初始化模块参数
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        # 逐层添加TwoWayAttentionBlock，构建多层Transformer
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        # 定义最终的注意力层，从查询到图像的注意力
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        # 最终注意力层的层归一化
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        point_pe: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Defines the forward pass for the TwoWayTransformer module.

        Args:
            image_embedding (Tensor): The input image embedding.
            image_pe (Tensor): Positional encoding for the image.
            point_embedding (Tensor): The input point/query embedding.
            point_pe (Tensor): Positional encoding for the point/query.
            **kwargs: Additional arguments.

        Returns:
            Tuple[Tensor, Tensor]: Output of the transformer module.
        """
        # 略去了具体的前向传播过程，因为还未完整给出
        pass
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must have same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          (torch.Tensor): the processed point_embedding
          (torch.Tensor): the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        # 获取图像嵌入的维度信息
        bs, c, h, w = image_embedding.shape
        # 将图像嵌入展平成 B x (H * W) x C，并置换维度顺序为 B x (H * W) x C
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        # 将图像位置编码展平成 B x (H * W) x C，并置换维度顺序为 B x (H * W) x C
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # 准备查询点
        queries = point_embedding
        keys = image_embedding

        # 应用 Transformer 块和最终的 LayerNorm
        for layer in self.layers:
            # 通过每个层处理查询和键
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # 应用从点到图像的最终注意力层
        q = queries + point_embedding
        k = keys + image_pe
        # 对点到图像的注意力计算输出
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        # 将查询更新为原查询加上注意力输出
        queries = queries + attn_out
        # 对最终的注意力查询应用 LayerNorm
        queries = self.norm_final_attn(queries)

        return queries, keys
# 定义一个名为TwoWayAttentionBlock的新的神经网络模块，用于实现自注意力和交叉注意力的操作，
# 其中包括从查询到键和从键到查询的两个方向。该模块由四个主要层组成：
# (1) 自注意力层在稀疏输入上操作，
# (2) 将稀疏输入的交叉注意力应用于密集输入，
# (3) 对稀疏输入执行MLP块，
# (4) 将密集输入的交叉注意力应用于稀疏输入。

class TwoWayAttentionBlock(nn.Module):
    """
    An attention block that performs both self-attention and cross-attention in two directions: queries to keys and
    keys to queries. This block consists of four main layers: (1) self-attention on sparse inputs, (2) cross-attention
    of sparse inputs to dense inputs, (3) an MLP block on sparse inputs, and (4) cross-attention of dense inputs to
    sparse inputs.

    Attributes:
        self_attn (Attention): The self-attention layer for the queries.
        norm1 (nn.LayerNorm): Layer normalization following the first attention block.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization following the second attention block.
        mlp (MLPBlock): MLP block that transforms the query embeddings.
        norm3 (nn.LayerNorm): Layer normalization following the MLP block.
        norm4 (nn.LayerNorm): Layer normalization following the third attention block.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Whether to skip the positional encoding in the first layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        Initialize the transformer block with four layers: 
        (1) self-attention on sparse inputs,
        (2) cross attention of sparse inputs to dense inputs,
        (3) mlp block on sparse inputs,
        (4) cross attention of dense inputs to sparse inputs.

        Args:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        # 第一层：自注意力层，用于处理查询
        self.self_attn = Attention(embedding_dim, num_heads)
        # 第一层后的层归一化
        self.norm1 = nn.LayerNorm(embedding_dim)

        # 第二层：从查询到键的交叉注意力层
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        # 第二层后的层归一化
        self.norm2 = nn.LayerNorm(embedding_dim)

        # 第三层：MLP块，用于转换查询的嵌入
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        # 第三层后的层归一化
        self.norm3 = nn.LayerNorm(embedding_dim)

        # 第四层：从密集输入到稀疏输入的交叉注意力层
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        # 是否跳过第一层的位置编码
        self.skip_first_layer_pe = skip_first_layer_pe
    # 定义一个方法 `forward`，接收四个张量参数 `queries`、`keys`、`query_pe`、`key_pe`，返回两个张量。
    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply self-attention and cross-attention to queries and keys and return the processed embeddings."""
        
        # Self attention block
        # 如果设置了跳过第一层的位置编码，直接对 queries 执行自注意力机制
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            # 否则，将位置编码 query_pe 加到 queries 上，然后执行自注意力机制
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        # 对经过注意力机制后的 queries 执行 Layer Normalization
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        # 将位置编码 query_pe 加到 queries 上，同时将位置编码 key_pe 加到 keys 上，然后执行跨注意力机制
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        # 对经过注意力机制后的 queries 执行 Layer Normalization
        queries = self.norm2(queries)

        # MLP block
        # 将 queries 输入到 MLP 网络中进行处理
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        # 对 MLP 输出后的 queries 执行 Layer Normalization
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        # 将位置编码 query_pe 加到 queries 上，同时将位置编码 key_pe 加到 keys 上，然后执行反向的跨注意力机制
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        # 对经过反向注意力机制后的 keys 执行 Layer Normalization
        keys = self.norm4(keys)

        # 返回处理后的 queries 和 keys
        return queries, keys
class Attention(nn.Module):
    """An attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        """
        Initializes the Attention model with the given dimensions and settings.

        Args:
            embedding_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            downsample_rate (int, optional): The factor by which the internal dimensions are downsampled. Defaults to 1.

        Raises:
            AssertionError: If 'num_heads' does not evenly divide the internal dim (embedding_dim / downsample_rate).
        """
        super().__init__()
        # 设置嵌入维度和内部维度
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        # 检查 num_heads 是否能够整除内部维度
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        # 初始化线性投影层
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    @staticmethod
    def _separate_heads(x: Tensor, num_heads: int) -> Tensor:
        """Separate the input tensor into the specified number of attention heads."""
        # 获取输入张量的形状信息
        b, n, c = x.shape
        # 重塑张量，将通道维度分为 num_heads 份
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    @staticmethod
    def _recombine_heads(x: Tensor) -> Tensor:
        """Recombine the separated attention heads into a single tensor."""
        # 获取张量的形状信息
        b, n_heads, n_tokens, c_per_head = x.shape
        # 转置张量，重新组合注意力头部
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Compute the attention output given the input query, key, and value tensors."""
        
        # 输入的投影
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 分割为注意力头部
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # 注意力计算
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # 获取输出
        out = attn @ v
        out = self._recombine_heads(out)
        return self.out_proj(out)
```