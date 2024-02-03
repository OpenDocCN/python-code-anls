# `.\segment-anything\segment_anything\modeling\transformer.py`

```
# 导入 torch 库中的相关模块
import torch
# 从 torch 库中导入 Tensor 和 nn 模块
from torch import Tensor, nn

# 导入 math 模块和 Tuple、Type 类型
import math
from typing import Tuple, Type

# 从当前目录下的 common 模块中导入 MLPBlock 类
from .common import MLPBlock

# 定义一个名为 TwoWayTransformer 的 nn.Module 类
class TwoWayTransformer(nn.Module):
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
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        # 调用父类的构造函数
        super().__init__()
        # 初始化深度、嵌入维度、头数、MLP维度等参数
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        # 初始化层列表
        self.layers = nn.ModuleList()

        # 循环创建指定层数的 TwoWayAttentionBlock，并添加到层列表中
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

        # 创建最终的注意力机制，用于将令牌与图像进行关联
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        # 对最终的注意力结果进行归一化处理
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
    # 定义一个方法，用于将点的嵌入和图像的嵌入进行处理
    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # 获取图像嵌入的形状信息
        bs, c, h, w = image_embedding.shape
        # 将图像嵌入展平为 BxHWxC 的形状
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        # 将图像位置编码展平为 BxHWxC 的形状
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # 准备查询点
        queries = point_embedding
        keys = image_embedding

        # 应用 transformer blocks 和最终的 layernorm
        for layer in self.layers:
            # 通过每个 transformer block 处理查询和键
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # 应用最终的注意力层，从点到图像
        q = queries + point_embedding
        k = keys + image_pe
        # 使用最终的注意力层处理点的查询和键
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        # 对最终的查询进行 layernorm 处理
        queries = self.norm_final_attn(queries)

        return queries, keys
class TwoWayAttentionBlock(nn.Module):
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
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        # 初始化函数，定义了一个包含四个层的 Transformer 块
        super().__init__()
        # 初始化 self-attention 层
        self.self_attn = Attention(embedding_dim, num_heads)
        # 初始化 LayerNorm 层
        self.norm1 = nn.LayerNorm(embedding_dim)

        # 初始化 cross-attention 层（token 到 image）
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        # 初始化 LayerNorm 层
        self.norm2 = nn.LayerNorm(embedding_dim)

        # 初始化 MLP 层
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        # 初始化 LayerNorm 层
        self.norm3 = nn.LayerNorm(embedding_dim)

        # 初始化 LayerNorm 层
        self.norm4 = nn.LayerNorm(embedding_dim)
        # 初始化 cross-attention 层（image 到 token）
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        # 是否跳过第一层的位置编码
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # 定义函数的输入和输出类型注解，返回两个张量

        # Self attention block
        # 如果跳过第一层位置编码
        if self.skip_first_layer_pe:
            # 使用自注意力机制处理查询
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            # 将查询和位置编码相加
            q = queries + query_pe
            # 使用自注意力机制处理查询
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        # 对查询进行归一化
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        # 将查询和位置编码相加
        q = queries + query_pe
        # 将键和位置编码相加
        k = keys + key_pe
        # 使用交叉注意力机制处理查询和键
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        # 对查询进行归一化
        queries = self.norm2(queries)

        # MLP block
        # 使用多层感知机处理查询
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        # 对查询进行归一化
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        # 将查询和位置编码相加
        q = queries + query_pe
        # 将键和位置编码相加
        k = keys + key_pe
        # 使用交叉注意力机制处理查询和键
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        # 对键进行归一化
        keys = self.norm4(keys)

        # 返回处理后的查询和键
        return queries, keys
class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        # 初始化函数，定义注意力层的参数
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        # 定义线性变换层，用于将输入的embedding映射到内部维度
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        # 将输入张量x按照头数分割
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        # 将分割后的张量重新组合
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # 输入的线性变换
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 将输入分割成多个头
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
        out = self.out_proj(out)

        return out
```