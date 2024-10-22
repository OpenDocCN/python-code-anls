# `.\cogvideo-finetune\sat\dit_video_concat.py`

```py
# 导入 functools 模块中的 partial 函数，用于部分应用函数
from functools import partial
# 从 einops 模块导入 rearrange 和 repeat 函数，用于张量重排和重复
from einops import rearrange, repeat
# 导入 numpy 库
import numpy as np

# 导入 PyTorch 库
import torch
# 从 torch 库导入神经网络模块
from torch import nn
# 导入 PyTorch 的功能性模块
import torch.nn.functional as F

# 从自定义模型库中导入基础模型和非冲突函数
from sat.model.base_model import BaseModel, non_conflict
# 从自定义模型库中导入基础混入类
from sat.model.mixins import BaseMixin
# 导入 transformer 的默认钩子和注意力函数
from sat.transformer_defaults import HOOKS_DEFAULT, attention_fn_default
# 从自定义模块中导入列并行线性层
from sat.mpu.layers import ColumnParallelLinear
# 从配置中实例化对象的工具函数
from sgm.util import instantiate_from_config

# 从扩散模块中导入时间步类
from sgm.modules.diffusionmodules.openaimodel import Timestep
# 从扩散模块中导入线性和时间嵌入的实用函数
from sgm.modules.diffusionmodules.util import (
    linear,
    timestep_embedding,
)
# 从自定义层归一化模块中导入层归一化和 RMS 归一化
from sat.ops.layernorm import LayerNorm, RMSNorm


# 定义图像补丁嵌入混入类，继承自基础混入类
class ImagePatchEmbeddingMixin(BaseMixin):
    # 初始化函数，接收输入通道数、隐藏层大小、补丁大小和其他可选参数
    def __init__(
        self,
        in_channels,
        hidden_size,
        patch_size,
        bias=True,
        text_hidden_size=None,
    ):
        # 调用父类构造函数
        super().__init__()
        # 创建卷积层以实现补丁嵌入
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        # 如果提供文本隐藏层大小，则创建线性层以处理文本嵌入
        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        # 否则将文本投影设置为 None
        else:
            self.text_proj = None

    # 定义词嵌入前向传播方法
    def word_embedding_forward(self, input_ids, **kwargs):
        # 获取 3D 图像补丁
        images = kwargs["images"]  # (b,t,c,h,w)
        # 获取批大小 B 和时间步 T
        B, T = images.shape[:2]
        # 将图像展平为 2D 形式以进行卷积操作
        emb = images.view(-1, *images.shape[2:])
        # 使用卷积层进行嵌入转换
        emb = self.proj(emb)  # ((b t),d,h/2,w/2)
        # 将嵌入重塑为三维形状
        emb = emb.view(B, T, *emb.shape[1:])
        # 扁平化嵌入并转置维度
        emb = emb.flatten(3).transpose(2, 3)  # (b,t,n,d)
        # 使用 rearrange 函数重排嵌入
        emb = rearrange(emb, "b t n d -> b (t n) d")

        # 如果存在文本投影，则计算文本嵌入
        if self.text_proj is not None:
            text_emb = self.text_proj(kwargs["encoder_outputs"])
            # 将文本嵌入与图像嵌入连接
            emb = torch.cat((text_emb, emb), dim=1)  # (b,n_t+t*n_i,d)

        # 确保嵌入在内存中是连续的
        emb = emb.contiguous()
        # 返回最终嵌入
        return emb  # (b,n_t+t*n_i,d)

    # 定义重新初始化函数
    def reinit(self, parent_model=None):
        # 获取卷积层的权重数据
        w = self.proj.weight.data
        # 使用 Xavier 均匀分布初始化权重
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # 将卷积层的偏置初始化为 0
        nn.init.constant_(self.proj.bias, 0)
        # 删除变压器的词嵌入
        del self.transformer.word_embeddings


# 定义获取 3D 正弦余弦位置嵌入的函数
def get_3d_sincos_pos_embed(
    embed_dim,
    grid_height,
    grid_width,
    t_size,
    cls_token=False,
    height_interpolation=1.0,
    width_interpolation=1.0,
    time_interpolation=1.0,
):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # 断言嵌入维度能够被 4 整除
    assert embed_dim % 4 == 0
    # 计算空间嵌入维度
    embed_dim_spatial = embed_dim // 4 * 3
    # 计算时间嵌入维度
    embed_dim_temporal = embed_dim // 4

    # 计算空间位置嵌入
    grid_h = np.arange(grid_height, dtype=np.float32) / height_interpolation
    grid_w = np.arange(grid_width, dtype=np.float32) / width_interpolation
    # 创建网格坐标，宽度优先
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # 将网格堆叠成一个数组
    grid = np.stack(grid, axis=0)

    # 重塑网格以适应后续计算
    grid = grid.reshape([2, 1, grid_height, grid_width])
    # 从网格获取 2D 正弦余弦位置嵌入
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # 处理时间位置嵌入
    # 创建一个一维数组 grid_t，值范围从 0 到 t_size-1，类型为 float32，并进行时间插值
    grid_t = np.arange(t_size, dtype=np.float32) / time_interpolation
    # 从 grid_t 生成一维的正弦余弦位置嵌入
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # 将位置嵌入的维度调整为 [T, 1, D] 形式，以便后续拼接
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    # 重复位置嵌入，以匹配网格高度和宽度，形成 [T, H*W, D // 4]
    pos_embed_temporal = np.repeat(pos_embed_temporal, grid_height * grid_width, axis=1)  # [T, H*W, D // 4]
    # 将空间位置嵌入调整为 [1, H*W, D // 4 * 3] 形式
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    # 重复空间位置嵌入，以匹配时间维度，形成 [T, H*W, D // 4 * 3]
    pos_embed_spatial = np.repeat(pos_embed_spatial, t_size, axis=0)  # [T, H*W, D // 4 * 3]

    # 将时间和空间位置嵌入在最后一个维度上进行拼接，形成最终的位置嵌入
    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    # pos_embed = pos_embed.reshape([-1, embed_dim])  # 将位置嵌入重塑为 [T*H*W, D]

    # 返回最终位置嵌入，维度为 [T, H*W, D]
    return pos_embed  # [T, H*W, D]
# 获取二维正弦余弦位置嵌入
def get_2d_sincos_pos_embed(embed_dim, grid_height, grid_width, cls_token=False, extra_tokens=0):
    # 定义网格高度和宽度的正弦余弦嵌入
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # 创建网格高度的数组
    grid_h = np.arange(grid_height, dtype=np.float32)
    # 创建网格宽度的数组
    grid_w = np.arange(grid_width, dtype=np.float32)
    # 生成网格的网格坐标，宽度在前
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # 将网格坐标堆叠成一个数组
    grid = np.stack(grid, axis=0)

    # 调整网格数组形状以便后续处理
    grid = grid.reshape([2, 1, grid_height, grid_width])
    # 从网格中获取二维正弦余弦位置嵌入
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # 如果需要类别标记且有额外的标记，进行拼接
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    # 返回位置嵌入
    return pos_embed


# 从网格中获取二维正弦余弦位置嵌入
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # 确保嵌入维度是偶数
    assert embed_dim % 2 == 0

    # 使用一半的维度来编码网格高度
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    # 使用一半的维度来编码网格宽度
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # 将高度和宽度的嵌入拼接在一起
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    # 返回嵌入
    return emb


# 从给定位置获取一维正弦余弦位置嵌入
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    # 定义每个位置的输出维度
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    # 确保嵌入维度是偶数
    assert embed_dim % 2 == 0
    # 生成频率因子数组
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    # 计算频率的倒数
    omega = 1.0 / 10000**omega  # (D/2,)

    # 将位置数组重塑为一维
    pos = pos.reshape(-1)  # (M,)
    # 计算外积以获取正弦和余弦的输入
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # 计算正弦值
    emb_sin = np.sin(out)  # (M, D/2)
    # 计算余弦值
    emb_cos = np.cos(out)  # (M, D/2)

    # 将正弦和余弦值拼接在一起
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    # 返回嵌入
    return emb


# 基础二维位置嵌入混合类
class Basic2DPositionEmbeddingMixin(BaseMixin):
    def __init__(self, height, width, compressed_num_frames, hidden_size, text_length=0):
        # 初始化基础混合类
        super().__init__()
        # 保存高度
        self.height = height
        # 保存宽度
        self.width = width
        # 计算空间长度
        self.spatial_length = height * width
        # 初始化位置嵌入参数
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.spatial_length), int(hidden_size)), requires_grad=False
        )

    # 前向传播位置嵌入
    def position_embedding_forward(self, position_ids, **kwargs):
        return self.pos_embedding

    # 重新初始化位置嵌入
    def reinit(self, parent_model=None):
        # 删除原位置嵌入
        del self.transformer.position_embeddings
        # 生成新的二维正弦余弦位置嵌入
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], self.height, self.width)
        # 将新嵌入拷贝到参数中
        self.pos_embedding.data[:, -self.spatial_length :].copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


# 基础三维位置嵌入混合类
class Basic3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        text_length=0,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
    ):
        # 调用父类构造函数
        super().__init__()
        # 设置实例的高度属性
        self.height = height
        # 设置实例的宽度属性
        self.width = width
        # 设置文本长度属性
        self.text_length = text_length
        # 设置压缩帧数属性
        self.compressed_num_frames = compressed_num_frames
        # 计算空间长度（高度乘以宽度）
        self.spatial_length = height * width
        # 计算补丁数量（高度乘以宽度乘以压缩帧数）
        self.num_patches = height * width * compressed_num_frames
        # 创建位置嵌入参数，初始化为零，形状为 (1, 文本长度 + 补丁数量, 隐藏层大小)，不需要梯度
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.num_patches), int(hidden_size)), requires_grad=False
        )
        # 设置高度插值属性
        self.height_interpolation = height_interpolation
        # 设置宽度插值属性
        self.width_interpolation = width_interpolation
        # 设置时间插值属性
        self.time_interpolation = time_interpolation

    def position_embedding_forward(self, position_ids, **kwargs):
        # 检查输入图像的通道数是否为 1
        if kwargs["images"].shape[1] == 1:
            # 返回位置嵌入，包含文本长度和空间长度的部分
            return self.pos_embedding[:, : self.text_length + self.spatial_length]

        # 返回位置嵌入，包含文本长度和序列长度的部分
        return self.pos_embedding[:, : self.text_length + kwargs["seq_length"]]

    def reinit(self, parent_model=None):
        # 删除当前模型的变换器位置嵌入
        del self.transformer.position_embeddings
        # 获取新的三维正弦余弦位置嵌入
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embedding.shape[-1],
            self.height,
            self.width,
            self.compressed_num_frames,
            height_interpolation=self.height_interpolation,
            width_interpolation=self.width_interpolation,
            time_interpolation=self.time_interpolation,
        )
        # 将位置嵌入转换为张量并转为浮点型
        pos_embed = torch.from_numpy(pos_embed).float()
        # 重新排列位置嵌入的形状
        pos_embed = rearrange(pos_embed, "t n d -> (t n) d")
        # 更新位置嵌入的最后一部分为新的位置嵌入
        self.pos_embedding.data[:, -self.num_patches :].copy_(pos_embed)
# 定义一个用于广播连接张量的函数
def broadcat(tensors, dim=-1):
    # 获取输入张量的数量
    num_tensors = len(tensors)
    # 创建一个集合，存储每个张量的维度长度
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    # 确保所有张量都有相同数量的维度
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    # 获取张量的维度长度
    shape_len = list(shape_lens)[0]
    # 如果 dim 是负数，则将其转换为正索引
    dim = (dim + shape_len) if dim < 0 else dim
    # 获取每个张量的形状，按维度进行打包
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    # 创建一个可扩展的维度列表，排除目标维度
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    # 确保可扩展的维度中每个维度的值不超过2种
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    # 获取每个可扩展维度的最大值
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    # 扩展维度，将最大值对应的维度扩展到张量数量
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    # 将目标维度插入到扩展维度中
    expanded_dims.insert(dim, (dim, dims[dim]))
    # 将扩展维度的形状打包
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    # 扩展每个张量到对应的形状
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    # 沿指定维度连接所有张量
    return torch.cat(tensors, dim=dim)


# 定义一个旋转半分的函数
def rotate_half(x):
    # 重新排列输入张量，将最后一维拆分成两个维度
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    # 按最后一维拆分为两个张量
    x1, x2 = x.unbind(dim=-1)
    # 将第二个张量取负并与第一个张量堆叠
    x = torch.stack((-x2, x1), dim=-1)
    # 重新排列堆叠后的张量，将其合并回一维
    return rearrange(x, "... d r -> ... (d r)")


# 定义一个类，用于混合旋转三维位置嵌入
class Rotary3DPositionEmbeddingMixin(BaseMixin):
    # 初始化类的构造函数
    def __init__(
        # 高度
        height,
        # 宽度
        width,
        # 压缩帧数
        compressed_num_frames,
        # 隐藏层大小
        hidden_size,
        # 每个头的隐藏层大小
        hidden_size_head,
        # 文本长度
        text_length,
        # theta参数，默认为10000
        theta=10000,
        # 是否使用旋转向量，默认为False
        rot_v=False,
        # 是否使用可学习的位置嵌入，默认为False
        learnable_pos_embed=False,
    # 定义构造函数的参数
        ):
            # 调用父类的构造函数
            super().__init__()
            # 初始化旋转向量
            self.rot_v = rot_v
    
            # 计算时间维度的尺寸
            dim_t = hidden_size_head // 4
            # 计算高度维度的尺寸
            dim_h = hidden_size_head // 8 * 3
            # 计算宽度维度的尺寸
            dim_w = hidden_size_head // 8 * 3
    
            # 计算时间频率
            freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
            # 计算高度频率
            freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
            # 计算宽度频率
            freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
    
            # 创建时间维度的网格
            grid_t = torch.arange(compressed_num_frames, dtype=torch.float32)
            # 创建高度维度的网格
            grid_h = torch.arange(height, dtype=torch.float32)
            # 创建宽度维度的网格
            grid_w = torch.arange(width, dtype=torch.float32)
    
            # 计算时间频率与网格的外积
            freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
            # 计算高度频率与网格的外积
            freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
            # 计算宽度频率与网格的外积
            freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
    
            # 扩展时间频率的维度
            freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
            # 扩展高度频率的维度
            freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
            # 扩展宽度频率的维度
            freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
    
            # 将时间、空间频率合并到一起
            freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
            # 重新排列频率的维度
            freqs = rearrange(freqs, "t h w d -> (t h w) d")
    
            # 确保频率数据在内存中的连续性
            freqs = freqs.contiguous()
            # 计算频率的正弦值
            freqs_sin = freqs.sin()
            # 计算频率的余弦值
            freqs_cos = freqs.cos()
            # 注册频率的正弦值为缓冲区
            self.register_buffer("freqs_sin", freqs_sin)
            # 注册频率的余弦值为缓冲区
            self.register_buffer("freqs_cos", freqs_cos)
    
            # 保存文本长度
            self.text_length = text_length
            # 如果学习位置嵌入
            if learnable_pos_embed:
                # 计算补丁数量
                num_patches = height * width * compressed_num_frames + text_length
                # 创建可学习的位置信息嵌入参数
                self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, int(hidden_size)), requires_grad=True)
            else:
                # 如果不学习位置嵌入，设置为 None
                self.pos_embedding = None
    
        # 定义旋转函数
        def rotary(self, t, **kwargs):
            # 获取序列长度
            seq_len = t.shape[2]
            # 提取对应序列长度的余弦频率
            freqs_cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
            # 提取对应序列长度的正弦频率
            freqs_sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
            # 返回旋转后的结果
            return t * freqs_cos + rotate_half(t) * freqs_sin
    
        # 定义位置嵌入前向传播函数
        def position_embedding_forward(self, position_ids, **kwargs):
            # 如果存在位置嵌入
            if self.pos_embedding is not None:
                # 返回对应位置的嵌入
                return self.pos_embedding[:, :self.text_length + kwargs["seq_length"]]
            else:
                # 如果没有位置嵌入，返回 None
                return None
    
        # 定义注意力函数
        def attention_fn(
            self,
            # 查询层
            query_layer,
            # 键层
            key_layer,
            # 值层
            value_layer,
            # 注意力掩码
            attention_mask,
            # 可选的注意力丢弃
            attention_dropout=None,
            # 可选的记录注意力权重
            log_attention_weights=None,
            # 是否缩放注意力得分
            scaling_attention_score=True,
            **kwargs,
    # 结束函数的定义部分
        ):
            # 从默认钩子中获取注意力函数
            attention_fn_default = HOOKS_DEFAULT["attention_fn"]
    
            # 对查询层的特定部分应用旋转操作
            query_layer[:, :, self.text_length :] = self.rotary(query_layer[:, :, self.text_length :])
            # 对键层的特定部分应用旋转操作
            key_layer[:, :, self.text_length :] = self.rotary(key_layer[:, :, self.text_length :])
            # 如果启用了旋转值，则对值层的特定部分应用旋转操作
            if self.rot_v:
                value_layer[:, :, self.text_length :] = self.rotary(value_layer[:, :, self.text_length :])
    
            # 返回默认的注意力函数，传入相关参数
            return attention_fn_default(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                attention_dropout=attention_dropout,
                log_attention_weights=log_attention_weights,
                scaling_attention_score=scaling_attention_score,
                **kwargs,
            )
# 定义调制函数，接受输入 x、偏移量 shift 和缩放因子 scale
def modulate(x, shift, scale):
    # 返回调制后的结果，通过 x、scale 和 shift 计算
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# 定义 unpatchify 函数，用于将输入 x 转换为图像格式
def unpatchify(x, c, p, w, h, rope_position_ids=None, **kwargs):
    """
    x: 输入形状为 (N, T/2 * S, patch_size**3 * C)
    imgs: 输出形状为 (N, T, H, W, C)
    """
    # 如果存在 rope_position_ids，执行未实现的检查
    if rope_position_ids is not None:
        assert NotImplementedError
        # 处理 pix2struct unpatchify
        L = x.shape[1]  # 获取 x 的第二维度大小
        x = x.reshape(shape=(x.shape[0], L, p, p, c))  # 重塑 x 以符合新的维度
        x = torch.einsum("nlpqc->ncplq", x)  # 使用爱因斯坦求和约定重新排列维度
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))  # 重塑为图像形状
    else:
        b = x.shape[0]  # 获取批次大小
        # 使用 rearrange 函数将 x 重新排列为图像格式
        imgs = rearrange(x, "b (t h w) (c p q) -> b t c (h p) (w q)", b=b, h=h, w=w, c=c, p=p, q=p)

    # 返回生成的图像
    return imgs


# 定义 FinalLayerMixin 类，继承自 BaseMixin
class FinalLayerMixin(BaseMixin):
    # 初始化类，设置各种参数和层
    def __init__(
        self,
        hidden_size,
        time_embed_dim,
        patch_size,
        out_channels,
        latent_width,
        latent_height,
        elementwise_affine,
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 存储隐藏层大小
        self.patch_size = patch_size  # 存储补丁大小
        self.out_channels = out_channels  # 存储输出通道数
        # 初始化 LayerNorm 层
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=1e-6)
        # 初始化全连接层
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # 初始化调制层
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 2 * hidden_size, bias=True))

        # 计算空间长度
        self.spatial_length = latent_width * latent_height // patch_size**2
        self.latent_width = latent_width  # 存储潜在宽度
        self.latent_height = latent_height  # 存储潜在高度

    # 定义 final_forward 方法
    def final_forward(self, logits, **kwargs):
        # 从 logits 中提取 x 和 emb
        x, emb = logits[:, kwargs["text_length"] :, :], kwargs["emb"]  # x:(b,(t n),d)
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)  # 调制得到 shift 和 scale
        x = modulate(self.norm_final(x), shift, scale)  # 对 x 进行调制
        x = self.linear(x)  # 通过线性层转换 x

        # 调用 unpatchify 生成最终图像
        return unpatchify(
            x,
            c=self.out_channels,
            p=self.patch_size,
            w=self.latent_width // self.patch_size,
            h=self.latent_height // self.patch_size,
            rope_position_ids=kwargs.get("rope_position_ids", None),
            **kwargs,
        )

    # 定义 reinit 方法
    def reinit(self, parent_model=None):
        # 初始化线性层权重
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)  # 将偏置初始化为 0


# 定义 SwiGLUMixin 类，继承自 BaseMixin
class SwiGLUMixin(BaseMixin):
    # 初始化类，设置层的数量和特征
    def __init__(self, num_layers, in_features, hidden_features, bias=False):
        super().__init__()  # 调用父类初始化
        # 创建一个模块列表，包含多个 ColumnParallelLinear 层
        self.w2 = nn.ModuleList(
            [
                ColumnParallelLinear(
                    in_features,
                    hidden_features,
                    gather_output=False,
                    bias=bias,
                    module=self,
                    name="dense_h_to_4h_gate",
                )
                for i in range(num_layers)  # 根据层数生成相应数量的层
            ]
        )
    # 前向传播函数，接受隐藏状态和其他参数
        def mlp_forward(self, hidden_states, **kw_args):
            # 将输入的隐藏状态赋值给 x
            x = hidden_states
            # 获取指定层的 MLP（多层感知机）模块
            origin = self.transformer.layers[kw_args["layer_id"]].mlp
            # 通过第一层全连接将 x 映射到 4h 维度
            x1 = origin.dense_h_to_4h(x)
            # 使用权重矩阵将 x 映射到另一个维度
            x2 = self.w2[kw_args["layer_id"]](x)
            # 应用激活函数并乘以第一层输出，生成隐藏状态
            hidden = origin.activation_func(x2) * x1
            # 将隐藏状态通过最后一层全连接映射回原始维度
            x = origin.dense_4h_to_h(hidden)
            # 返回最终输出
            return x
# 定义一个混合类 AdaLNMixin，继承自 BaseMixin
class AdaLNMixin(BaseMixin):
    # 初始化方法，定义类的构造参数
    def __init__(
        self,
        width,  # 宽度参数
        height,  # 高度参数
        hidden_size,  # 隐藏层大小
        num_layers,  # 层数
        time_embed_dim,  # 时间嵌入维度
        compressed_num_frames,  # 压缩帧数
        qk_ln=True,  # 是否使用查询和键的层归一化
        hidden_size_head=None,  # 每个头的隐藏层大小
        elementwise_affine=True,  # 是否使用逐元素仿射变换
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存层数到实例变量
        self.num_layers = num_layers
        # 保存宽度到实例变量
        self.width = width
        # 保存高度到实例变量
        self.height = height
        # 保存压缩帧数到实例变量
        self.compressed_num_frames = compressed_num_frames

        # 创建一个包含多个线性层和激活函数的模块列表
        self.adaLN_modulations = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size)) for _ in range(num_layers)]
        )

        # 保存是否使用层归一化的标志
        self.qk_ln = qk_ln
        # 如果使用层归一化，则初始化查询和键的层归一化列表
        if qk_ln:
            # 创建查询层归一化的模块列表
            self.query_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
            # 创建键层归一化的模块列表
            self.key_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )

    # 定义前向传播方法
    def layer_forward(
        self,
        hidden_states,  # 输入的隐藏状态
        mask,  # 输入的掩码
        *args,  # 其他可变参数
        **kwargs,  # 其他关键字参数
    ):
        pass  # 该方法未实现，留作将来的扩展

    # 定义重初始化方法
    def reinit(self, parent_model=None):
        # 对每个 adaLN 调制层进行初始化
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)  # 将最后一层的权重初始化为0
            nn.init.constant_(layer[-1].bias, 0)  # 将最后一层的偏置初始化为0

    # 定义注意力函数，带有非冲突装饰器
    @non_conflict
    def attention_fn(
        self,
        query_layer,  # 查询层
        key_layer,  # 键层
        value_layer,  # 值层
        attention_mask,  # 注意力掩码
        attention_dropout=None,  # 注意力的丢弃率
        log_attention_weights=None,  # 日志注意力权重
        scaling_attention_score=True,  # 是否缩放注意力得分
        old_impl=attention_fn_default,  # 默认的注意力实现
        **kwargs,  # 其他关键字参数
    ):
        # 如果使用查询和键的层归一化
        if self.qk_ln:
            # 获取当前层的查询层归一化模块
            query_layernorm = self.query_layernorm_list[kwargs["layer_id"]]
            # 获取当前层的键层归一化模块
            key_layernorm = self.key_layernorm_list[kwargs["layer_id"]]
            # 对查询层进行层归一化
            query_layer = query_layernorm(query_layer)
            # 对键层进行层归一化
            key_layer = key_layernorm(key_layer)

        # 返回注意力函数的结果
        return old_impl(
            query_layer,  # 归一化后的查询层
            key_layer,  # 归一化后的键层
            value_layer,  # 值层
            attention_mask,  # 注意力掩码
            attention_dropout=attention_dropout,  # 注意力的丢弃率
            log_attention_weights=log_attention_weights,  # 日志注意力权重
            scaling_attention_score=scaling_attention_score,  # 缩放注意力得分
            **kwargs,  # 其他关键字参数
        )

# 定义数据类型到 PyTorch 数据类型的映射
str_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# 定义扩散变换器类，继承自 BaseModel
class DiffusionTransformer(BaseModel):
    # 初始化方法，用于创建类的实例
    def __init__(
        # 变换器参数
        self,
        transformer_args,
        # 帧数
        num_frames,
        # 时间压缩率
        time_compressed_rate,
        # 潜在宽度
        latent_width,
        # 潜在高度
        latent_height,
        # 补丁大小
        patch_size,
        # 输入通道数
        in_channels,
        # 输出通道数
        out_channels,
        # 隐藏层大小
        hidden_size,
        # 层数
        num_layers,
        # 注意力头数
        num_attention_heads,
        # 是否进行逐元素仿射变换
        elementwise_affine,
        # 时间嵌入维度，默认为 None
        time_embed_dim=None,
        # 类别数量，默认为 None
        num_classes=None,
        # 模块配置，默认为空字典
        modules={},
        # 输入时间格式，默认为 "adaln"
        input_time="adaln",
        # 自适应输入通道数，默认为 None
        adm_in_channels=None,
        # 是否并行输出，默认为 True
        parallel_output=True,
        # 高度插值因子，默认为 1.0
        height_interpolation=1.0,
        # 宽度插值因子，默认为 1.0
        width_interpolation=1.0,
        # 时间插值因子，默认为 1.0
        time_interpolation=1.0,
        # 是否使用 SwiGLU 激活函数，默认为 False
        use_SwiGLU=False,
        # 是否使用 RMSNorm 归一化，默认为 False
        use_RMSNorm=False,
        # 是否将 y 嵌入初始化为零，默认为 False
        zero_init_y_embed=False,
        # 其他参数，使用关键字参数收集
        **kwargs,
    ):
        # 设置潜在宽度
        self.latent_width = latent_width
        # 设置潜在高度
        self.latent_height = latent_height
        # 设置补丁大小
        self.patch_size = patch_size
        # 设置帧数
        self.num_frames = num_frames
        # 设置时间压缩率
        self.time_compressed_rate = time_compressed_rate
        # 计算空间长度
        self.spatial_length = latent_width * latent_height // patch_size**2
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置输出通道数
        self.out_channels = out_channels
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型通道数，等于隐藏层大小
        self.model_channels = hidden_size
        # 设置时间嵌入维度，如果未提供则使用隐藏层大小
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else hidden_size
        # 设置类别数量
        self.num_classes = num_classes
        # 设置自适应输入通道数
        self.adm_in_channels = adm_in_channels
        # 设置输入时间格式
        self.input_time = input_time
        # 设置层数
        self.num_layers = num_layers
        # 设置注意力头数
        self.num_attention_heads = num_attention_heads
        # 设置是否为解码器
        self.is_decoder = transformer_args.is_decoder
        # 设置是否进行逐元素仿射变换
        self.elementwise_affine = elementwise_affine
        # 设置高度插值因子
        self.height_interpolation = height_interpolation
        # 设置宽度插值因子
        self.width_interpolation = width_interpolation
        # 设置时间插值因子
        self.time_interpolation = time_interpolation
        # 计算内部隐藏层大小，等于隐藏层大小的四倍
        self.inner_hidden_size = hidden_size * 4
        # 设置是否将 y 嵌入初始化为零
        self.zero_init_y_embed = zero_init_y_embed
        # 尝试从关键字参数中提取数据类型
        try:
            self.dtype = str_to_dtype[kwargs.pop("dtype")]
        except:
            # 默认数据类型为 float32
            self.dtype = torch.float32

        # 如果使用 SwiGLU 激活函数，将其添加到关键字参数中
        if use_SwiGLU:
            kwargs["activation_func"] = F.silu
        # 如果没有指定激活函数，使用近似 GELU 激活函数
        elif "activation_func" not in kwargs:
            approx_gelu = nn.GELU(approximate="tanh")
            kwargs["activation_func"] = approx_gelu

        # 如果使用 RMSNorm 归一化，添加到关键字参数中
        if use_RMSNorm:
            kwargs["layernorm"] = RMSNorm
        else:
            # 否则使用带有逐元素仿射变换的 LayerNorm
            kwargs["layernorm"] = partial(LayerNorm, elementwise_affine=elementwise_affine, eps=1e-6)

        # 更新变换器参数中的层数、隐藏层大小和注意力头数
        transformer_args.num_layers = num_layers
        transformer_args.hidden_size = hidden_size
        transformer_args.num_attention_heads = num_attention_heads
        transformer_args.parallel_output = parallel_output
        # 调用父类的初始化方法
        super().__init__(args=transformer_args, transformer=None, **kwargs)

        # 模块配置
        module_configs = modules
        # 构建模块
        self._build_modules(module_configs)

        # 如果使用 SwiGLU 激活函数，添加混合层
        if use_SwiGLU:
            self.add_mixin(
                "swiglu", SwiGLUMixin(num_layers, hidden_size, self.inner_hidden_size, bias=False), reinit=True
            )
    # 前向传播函数，接收输入 x 及其他可选参数
        def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
            # 获取输入 x 的形状，分别为批量大小 b、时间步 t、通道 d、高 h 和宽 w
            b, t, d, h, w = x.shape
            # 如果输入 x 的数据类型与模型的数据类型不匹配，则转换为模型的数据类型
            if x.dtype != self.dtype:
                x = x.to(self.dtype)
    
            # 此部分在推理时不使用
            if "concat_images" in kwargs and kwargs["concat_images"] is not None:
                # 如果 concat_images 的批量大小与 x 不匹配，则重复 concat_images
                if kwargs["concat_images"].shape[0] != x.shape[0]:
                    concat_images = kwargs["concat_images"].repeat(2, 1, 1, 1, 1)
                else:
                    # 否则直接使用 concat_images
                    concat_images = kwargs["concat_images"]
                # 将 x 和 concat_images 在时间维度上进行拼接
                x = torch.cat([x, concat_images], dim=2)
    
            # 断言 y 的存在性与 num_classes 的存在性相对应，确保一致性
            assert (y is not None) == (
                self.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            # 生成时间步的嵌入向量
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
            # 通过时间嵌入获取最终的嵌入
            emb = self.time_embed(t_emb)
    
            if self.num_classes is not None:
                # 确保 y 的形状与 x 的批量大小相符
                # assert y.shape[0] == x.shape[0]
                # 确保 x 的批量大小能被 y 的批量大小整除
                assert x.shape[0] % y.shape[0] == 0
                # 重复 y 以匹配 x 的批量大小
                y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
                # 将类别嵌入添加到 emb 中
                emb = emb + self.label_emb(y)
    
            # 在 kwargs 中存储序列长度、图像、嵌入、编码器输出及文本长度
            kwargs["seq_length"] = t * h * w // (self.patch_size**2)
            kwargs["images"] = x
            kwargs["emb"] = emb
            kwargs["encoder_outputs"] = context
            kwargs["text_length"] = context.shape[1]
    
            # 初始化输入 ID、位置 ID 和注意力掩码为全 1 的张量
            kwargs["input_ids"] = kwargs["position_ids"] = kwargs["attention_mask"] = torch.ones((1, 1)).to(x.dtype)
            # 调用父类的 forward 方法，并获取第一个输出
            output = super().forward(**kwargs)[0]
            # 返回最终输出
            return output
```