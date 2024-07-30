# `.\yolov8\ultralytics\nn\modules\transformer.py`

```py
# 导入必要的库和模块
"""Transformer modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

# 导入自定义的模块和函数
from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = (
    "TransformerEncoderLayer",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
)

# 定义 Transformer 编码器层的类
class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        
        # 检查是否满足需要的 PyTorch 版本
        from ...utils.torch_utils import TORCH_1_9
        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        
        # 多头注意力机制
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        
        # 前向传播模型的实现
        self.fc1 = nn.Linear(c1, cm)  # 第一个全连接层
        self.fc2 = nn.Linear(cm, c1)  # 第二个全连接层

        # 层归一化
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.act = act

        # 是否在归一化之前进行层归一化
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        # 添加位置编码
        q = k = self.with_pos_embed(src, pos)
        
        # 多头注意力机制的前向传播
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        
        # 残差连接和 dropout
        src = src + self.dropout1(src2)
        
        # 层归一化
        src = self.norm1(src)
        
        # 前向传播的第二部分：全连接层和激活函数
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        
        # 残差连接和 dropout
        src = src + self.dropout2(src2)
        
        # 再次进行层归一化
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        # 层归一化
        src2 = self.norm1(src)
        
        # 添加位置编码
        q = k = self.with_pos_embed(src2, pos)
        
        # 多头注意力机制的前向传播
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        
        # 残差连接和 dropout
        src = src + self.dropout1(src2)
        
        # 再次进行层归一化
        src2 = self.norm2(src)
        
        # 前向传播的第二部分：全连接层和激活函数
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        
        # 残差连接
        return src + self.dropout2(src2)
    # 此方法用于在编码器模块中前向传播输入数据。
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        # 如果设置了 normalize_before 标志，则调用前向传播前处理方法
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # 否则调用前向传播后处理方法
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        # 构建二维的正弦-余弦位置编码
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # 将输入张量展平成[B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        # 将张量重新整形为原始形状[B, C, H, W]，并保持连续性
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        # 创建网格坐标
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        # 计算频率因子
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        # 计算位置编码
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        # 拼接正弦和余弦编码
        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        # 初始化查询、键、值的线性变换
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        # 多头注意力机制
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        # 两个线性层
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        # 使用自注意力机制和残差连接
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        # 两个线性层的前向传播，并添加残差连接
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        # 如果输入通道数和输出通道数不同，添加一个卷积层
        if c1 != c2:
            self.conv = Conv(c1, c2)
        # 可学习的位置编码
        self.linear = nn.Linear(c2, c2)
        # 多个Transformer层组成的序列
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2
    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        # 如果存在卷积层，则将输入 x 传递给卷积层进行处理
        if self.conv is not None:
            x = self.conv(x)
        # 获取输入张量 x 的形状信息：批量大小 b, 通道数 _, 宽度 w, 高度 h
        b, _, w, h = x.shape
        # 将 x 展平为二维张量，然后对维度进行置换，变换顺序为 (2, 0, 1)
        p = x.flatten(2).permute(2, 0, 1)
        # 将展平后的张量 p 输入到 self.linear 进行线性变换，并加上原始的 p
        # 然后再将结果进行置换，变换顺序为 (1, 2, 0)，最后将其重新形状为 (b, self.c2, w, h)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
# 实现多层感知机（MLP）的单个块
class MLPBlock(nn.Module):
    """Implements a single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function."""
        super().__init__()
        # 第一层线性变换，将输入的embedding_dim维度映射到mlp_dim维度
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        # 第二层线性变换，将mlp_dim维度的输入映射回embedding_dim维度
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        # 激活函数，默认为GELU
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock."""
        # 前向传播过程，先通过第一层线性变换和激活函数，再通过第二层线性变换
        return self.lin2(self.act(self.lin1(x)))


# 实现简单的多层感知机（MLP）模型，也称为前馈神经网络（FFN）
class MLP(nn.Module):
    """Implements a simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers."""
        super().__init__()
        self.num_layers = num_layers
        # 根据层数和各层的维度，创建多个线性层组成的层列表
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """Forward pass for the entire MLP."""
        # 逐层进行前向传播，对前num_layers-1层使用ReLU激活函数
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 实现二维的层归一化模块，灵感来自Detectron2和ConvNeXt的实现
class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    Original implementations in
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    and
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """

    def __init__(self, num_channels, eps=1e-6):
        """Initialize LayerNorm2d with the given parameters."""
        super().__init__()
        # 初始化归一化层的权重和偏置参数
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Perform forward pass for 2D layer normalization."""
        # 对输入进行二维的层归一化计算
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# 实现基于Deformable-DETR和PaddleDetection实现的多尺度可变形注意力模块
class MSDeformAttn(nn.Module):
    """
    Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        # 调用父类的初始化方法
        super().__init__()
        # 检查是否满足 d_model 可以被 n_heads 整除的条件，否则抛出错误
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        # 计算每个头部的维度
        _d_per_head = d_model // n_heads
        # 断言 d_model 必须能够被 n_heads 整除，用于检查计算的正确性
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        # 设置 im2col 操作的步长
        self.im2col_step = 64

        # 初始化各个模型参数
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 创建用于偏移量的线性层，输入维度为 d_model，输出维度为 n_heads * n_levels * n_points * 2
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 创建用于注意力权重的线性层，输入维度为 d_model，输出维度为 n_heads * n_levels * n_points
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 创建用于值投影的线性层，输入维度和输出维度都是 d_model
        self.value_proj = nn.Linear(d_model, d_model)
        # 创建用于输出投影的线性层，输入维度和输出维度都是 d_model
        self.output_proj = nn.Linear(d_model, d_model)

        # 调用内部方法，初始化模型参数
        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        # 将 sampling_offsets 的权重初始化为常数 0.0
        constant_(self.sampling_offsets.weight.data, 0.0)
        # 生成一组角度 thetas，用于初始化采样网格
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # 将初始化的网格归一化，并重复以适应不同的 levels 和 points
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        # 根据点的索引加权初始化网格的不同部分
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        # 使用无梯度的方式将初始化后的网格作为偏置参数赋给 sampling_offsets
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # 将 attention_weights 的权重初始化为常数 0.0
        constant_(self.attention_weights.weight.data, 0.0)
        # 将 attention_weights 的偏置初始化为常数 0.0
        constant_(self.attention_weights.bias.data, 0.0)
        # 使用 xavier_uniform 方法初始化 value_proj 的权重
        xavier_uniform_(self.value_proj.weight.data)
        # 将 value_proj 的偏置初始化为常数 0.0
        constant_(self.value_proj.bias.data, 0.0)
        # 使用 xavier_uniform 方法初始化 output_proj 的权重
        xavier_uniform_(self.output_proj.weight.data)
        # 将 output_proj 的偏置初始化为常数 0.0
        constant_(self.output_proj.bias.data, 0.0)
    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C] 输入的查询张量，形状为 [批大小, 查询长度, 通道数]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area 参考边界框张量，形状为 [批大小, 查询长度, 层级数, 2]，表示区域范围在 [0, 1] 之间，左上角为 (0,0)，右下角为 (1,1)，包含填充区域
            value (torch.Tensor): [bs, value_length, C] 输入的值张量，形状为 [批大小, 值长度, 通道数]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})] 不同层级的值张量形状列表
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements 值张量的掩码，形状为 [批大小, 值长度]，True 表示非填充元素，False 表示填充元素

        Returns:
            output (Tensor): [bs, Length_{query}, C] 输出张量，形状为 [批大小, 查询长度, 通道数]
        """
        bs, len_q = query.shape[:2]  # 获取批大小和查询长度
        len_v = value.shape[1]  # 获取值张量的长度
        assert sum(s[0] * s[1] for s in value_shapes) == len_v  # 确保所有层级的值张量形状乘积等于值张量长度

        value = self.value_proj(value)  # 使用值投影函数处理值张量

        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))  # 根据值张量的掩码进行填充

        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)  # 调整值张量的形状为 [批大小, 值长度, 头数, 模型维度//头数]

        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # 计算采样偏移量，形状为 [批大小, 查询长度, 头数, 层级数, 采样点数, 2]

        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        # 计算注意力权重，形状为 [批大小, 查询长度, 头数, 层级数 * 采样点数]

        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # 对注意力权重进行 softmax 归一化，形状为 [批大小, 查询长度, 头数, 层级数, 采样点数]

        num_points = refer_bbox.shape[-1]  # 获取参考边界框张量的最后一个维度大小

        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
            # 根据参考边界框张量和采样偏移量计算采样位置，形状为 [批大小, 查询长度, 1, 层级数, 1, 2]

        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
            # 根据参考边界框张量、采样偏移量和采样点数计算采样位置，形状为 [批大小, 查询长度, 1, 层级数, 1, 2]

        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
            # 如果参考边界框张量的最后一个维度不是 2 或 4，则引发值错误异常

        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        # 使用多尺度可变形注意力函数计算输出结果

        return self.output_proj(output)
        # 对输出结果进行投影处理并返回
class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.0, act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)  # 创建自注意力层
        self.dropout1 = nn.Dropout(dropout)  # 定义第一层dropout
        self.norm1 = nn.LayerNorm(d_model)  # 定义第一层Layer Normalization

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)  # 创建交叉注意力层
        self.dropout2 = nn.Dropout(dropout)  # 定义第二层dropout
        self.norm2 = nn.LayerNorm(d_model)  # 定义第二层Layer Normalization

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)  # 第一层线性变换
        self.act = act  # 激活函数
        self.dropout3 = nn.Dropout(dropout)  # 定义第三层dropout
        self.linear2 = nn.Linear(d_ffn, d_model)  # 第二层线性变换
        self.dropout4 = nn.Dropout(dropout)  # 定义第四层dropout
        self.norm3 = nn.LayerNorm(d_model)  # 定义第三层Layer Normalization

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos  # 如果提供了位置编码，则将其添加到输入张量中

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))  # 前向传播过程中的前馈网络部分
        tgt = tgt + self.dropout4(tgt2)  # 加上残差连接和最后一层dropout
        return self.norm3(tgt)  # 应用Layer Normalization

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""

        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)  # 添加位置编码后的查询和键
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[0].transpose(0, 1)  # 自注意力机制
        embed = embed + self.dropout1(tgt)  # 加上残差连接和第一层dropout
        embed = self.norm1(embed)  # 应用Layer Normalization

        # Cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )  # 交叉注意力机制
        embed = embed + self.dropout2(tgt)  # 加上残差连接和第二层dropout
        embed = self.norm2(embed)  # 应用Layer Normalization

        # FFN
        return self.forward_ffn(embed)  # 前向传播过程中的前馈网络
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        # 调用父类初始化方法
        super().__init__()
        # 使用 _get_clones 函数复制 decoder_layer，构建层列表
        self.layers = _get_clones(decoder_layer, num_layers)
        # 记录解码器层数
        self.num_layers = num_layers
        # 记录隐藏层维度
        self.hidden_dim = hidden_dim
        # 设置评估索引，如果未指定则为最后一层
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed,  # 解码器嵌入
        refer_bbox,  # 锚框
        feats,  # 图像特征
        shapes,  # 特征形状
        bbox_head,
        score_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
    ):
        """Perform the forward pass through the entire decoder."""
        # 初始化输出为解码器嵌入
        output = embed
        # 初始化解码器生成的边界框和类别
        dec_bboxes = []
        dec_cls = []
        # 初始化最后细化的参考边界框为 None
        last_refined_bbox = None
        # 对参考边界框进行 sigmoid 操作
        refer_bbox = refer_bbox.sigmoid()
        # 遍历所有层进行前向传播
        for i, layer in enumerate(self.layers):
            # 在当前层应用解码器操作
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            # 预测边界框
            bbox = bbox_head[i](output)
            # 计算细化的边界框
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            # 如果处于训练阶段
            if self.training:
                # 记录类别预测结果
                dec_cls.append(score_head[i](output))
                # 如果是第一层，直接记录细化后的边界框
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    # 否则记录上一次细化后的边界框
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            # 如果处于评估阶段且达到指定的评估层次
            elif i == self.eval_idx:
                # 记录类别预测结果
                dec_cls.append(score_head[i](output))
                # 记录细化后的边界框
                dec_bboxes.append(refined_bbox)
                break

            # 更新上一次细化后的边界框为当前细化后的边界框
            last_refined_bbox = refined_bbox
            # 更新参考边界框为当前细化后的边界框的分离版本（在训练阶段）
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        # 返回堆叠的边界框和类别预测结果
        return torch.stack(dec_bboxes), torch.stack(dec_cls)
```