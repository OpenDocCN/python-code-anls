# `.\lucidrains\meshgpt-pytorch\meshgpt_pytorch\meshgpt_pytorch.py`

```py
# 导入所需的模块
from pathlib import Path
from functools import partial
from math import ceil, pi, sqrt

import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

# 导入自定义的类型注解
from torchtyping import TensorType

# 导入自定义的工具函数
from pytorch_custom_utils import save_load

# 导入类型注解相关的模块
from beartype import beartype
from beartype.typing import Union, Tuple, Callable, Optional, List, Dict, Any

# 导入 einops 库中的函数
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# 导入 einx 库中的函数
from einx import get_at

# 导入 x_transformers 库中的模块和函数
from x_transformers import Decoder
from x_transformers.attend import Attend
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates

# 导入自动回归包装器相关的函数
from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
    top_p,
)

# 导入本地注意力相关的函数
from local_attention import LocalMHA

# 导入向量量化相关的函数
from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)

# 导入 meshgpt_pytorch 库中的函数
from meshgpt_pytorch.data import derive_face_edges_from_faces
from meshgpt_pytorch.version import __version__

# 导入 Taylor 级数线性注意力相关的函数
from taylor_series_linear_attention import TaylorSeriesLinearAttn

# 导入无分类器引导相关的函数
from classifier_free_guidance_pytorch import (
    classifier_free_guidance,
    TextEmbeddingReturner
)

# 导入 torch_geometric 库中的函数
from torch_geometric.nn.conv import SAGEConv

# 导入 gateloop_transformer 库中的函数
from gateloop_transformer import SimpleGateLoopLayer

# 导入 tqdm 库中的函数
from tqdm import tqdm

# 定义一些辅助函数

# 检查变量是否存在
def exists(v):
    return v is not None

# 返回默认值
def default(v, d):
    return v if exists(v) else d

# 返回迭代器的第一个元素
def first(it):
    return it[0]

# 检查一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0

# 检查一个数是否为奇数
def is_odd(n):
    return not divisible_by(n, 2)

# 检查列表是否为空
def is_empty(l):
    return len(l) == 0

# 检查张量是否为空
def is_tensor_empty(t: Tensor):
    return t.numel() == 0

# 设置模块的 requires_grad 属性
def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

# 计算张量的 L1 范数
def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

# 计算张量的 L2 范数
def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

# 安全地拼接张量
def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim = dim)

# 在指定维度上填充张量
def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)

# 将张量填充到指定长度
def pad_to_length(t, length, dim = -1, value = 0, right = True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim = dim, value = value)

# 连续嵌入

def ContinuousEmbed(dim_cont):
    return nn.Sequential(
        Rearrange('... -> ... 1'),
        nn.Linear(1, dim_cont),
        nn.SiLU(),
        nn.Linear(dim_cont, dim_cont),
        nn.LayerNorm(dim_cont)
    )

# 获取派生的面特征
# 1. 角度 (3), 2. 面积 (1), 3. 法线 (3)

# 计算两个向量之间的夹角
def derive_angle(x, y, eps = 1e-5):
    z = einsum('... d, ... d -> ...', l2norm(x), l2norm(y))
    return z.clip(-1 + eps, 1 - eps).arccos()

# 获取派生的面特征
@torch.no_grad()
def get_derived_face_features(
    face_coords: TensorType['b', 'nf', 'nvf', 3, float]  # 3 or 4 vertices with 3 coordinates
):
    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2)

    angles  = derive_angle(face_coords, shifted_face_coords)

    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2)

    normals = l2norm(torch.cross(edge1, edge2, dim = -1))
    area = normals.norm(dim = -1, keepdim = True) * 0.5

    return dict(
        angles = angles,
        area = area,
        normals = normals
    )   

# 张量辅助函数

# 将连续值离散化
@beartype
def discretize(
    t: Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    # 断言高值大于低值，确保输入的范围是有效的
    assert hi > lo
    
    # 将输入值 t 根据给定的范围进行归一化处理
    t = (t - lo) / (hi - lo)
    # 将归一化后的值映射到离散值范围内
    t *= num_discrete
    # 将映射后的值进行偏移，使得离散值范围从0开始
    t -= 0.5
    
    # 将处理后的值四舍五入取整，并转换为长整型，然后限制在离散值范围内
    return t.round().long().clamp(min=0, max=num_discrete - 1)
# 使用 beartype 装饰器对 undiscretize 函数进行类型检查
@beartype
# 将连续值转换为离散值
def undiscretize(
    t: Tensor,  # 输入张量
    *,
    continuous_range = Tuple[float, float],  # 连续值范围
    num_discrete: int = 128  # 离散值数量
) -> Tensor:  # 返回张量
    lo, hi = continuous_range  # 解包连续值范围
    assert hi > lo  # 断言确保上限大于下限

    t = t.float()  # 将输入张量转换为浮点型

    t += 0.5  # 加上0.5
    t /= num_discrete  # 除以离散值数量
    return t * (hi - lo) + lo  # 返回转换后的张量

# 使用 beartype 装饰器对 gaussian_blur_1d 函数进行类型检查
@beartype
# 一维高斯模糊
def gaussian_blur_1d(
    t: Tensor,  # 输入张量
    *,
    sigma: float = 1.  # 高斯模糊的标准差
) -> Tensor:  # 返回张量

    _, _, channels, device, dtype = *t.shape, t.device, t.dtype  # 解包张量的形状、设备和数据类型

    width = int(ceil(sigma * 5))  # 计算模糊核的宽度
    width += (width + 1) % 2  # 确保宽度为奇数
    half_width = width // 2  # 计算宽度的一半

    distance = torch.arange(-half_width, half_width + 1, dtype = dtype, device = device)  # 生成距离张量

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))  # 计算高斯权重
    gaussian = l1norm(gaussian)  # 对高斯权重进行 L1 归一化

    kernel = repeat(gaussian, 'n -> c 1 n', c = channels)  # 重复高斯权重以匹配通道数

    t = rearrange(t, 'b n c -> b c n')  # 重新排列输入张量的维度
    out = F.conv1d(t, kernel, padding = half_width, groups = channels)  # 一维卷积操作
    return rearrange(out, 'b c n -> b n c')  # 重新排列输出张量的维度

# 使用 beartype 装饰器对 scatter_mean 函数进行类型检查
@beartype
# 对张量进行均值散点
def scatter_mean(
    tgt: Tensor,  # 目标张量
    indices: Tensor,  # 索引张量
    src = Tensor,  # 源张量
    *,
    dim: int = -1,  # 维度
    eps: float = 1e-5  # 防止除零的小值
):
    """
    todo: update to pytorch 2.1 and try https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
    """
    num = tgt.scatter_add(dim, indices, src)  # 使用索引张量将源张量的值加到目标张量上
    den = torch.zeros_like(tgt).scatter_add(dim, indices, torch.ones_like(src))  # 计算分母
    return num / den.clamp(min = eps)  # 返回均值

# resnet block

# 像素归一化模块
class PixelNorm(Module):
    def __init__(self, dim, eps = 1e-4):  # 初始化函数
        super().__init__()  # 调用父类初始化函数
        self.dim = dim  # 维度
        self.eps = eps  # 小值

    def forward(self, x):  # 前向传播函数
        dim = self.dim  # 获取维度
        return F.normalize(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])  # 返回归一化后的张量

# Squeeze-and-Excitation 模块
class SqueezeExcite(Module):
    def __init__(
        self,
        dim,
        reduction_factor = 4,  # 缩减因子
        min_dim = 16  # 最小维度
    ):
        super().__init__()  # 调用父类初始化函数
        dim_inner = max(dim // reduction_factor, min_dim)  # 计算内部维度

        self.net = nn.Sequential(  # 定义神经网络
            nn.Linear(dim, dim_inner),  # 线性层
            nn.SiLU(),  # SiLU 激活函数
            nn.Linear(dim_inner, dim),  # 线性层
            nn.Sigmoid(),  # Sigmoid 激活函数
            Rearrange('b c -> b c 1')  # 重新排列维度
        )

    def forward(self, x, mask = None):  # 前向传播函数
        if exists(mask):  # 如果存在掩码
            x = x.masked_fill(~mask, 0.)  # 使用掩码填充张量

            num = reduce(x, 'b c n -> b c', 'sum')  # 沿指定维度求和
            den = reduce(mask.float(), 'b 1 n -> b 1', 'sum')  # 沿指定维度求和
            avg = num / den.clamp(min = 1e-5)  # 计算均值
        else:
            avg = reduce(x, 'b c n -> b c', 'mean')  # 沿指定维度求均值

        return x * self.net(avg)  # 返回加权后的张量

# 基本块
class Block(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        dropout = 0.
    ):
        super().__init__()  # 调用父类初始化函数
        dim_out = default(dim_out, dim)  # 设置输出维度为输入维度

        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)  # 一维卷积层
        self.norm = PixelNorm(dim = 1)  # 像素归一化
        self.dropout = nn.Dropout(dropout)  # 随机失活层
        self.act = nn.SiLU()  # SiLU 激活函数

    def forward(self, x, mask = None):  # 前向传播函数
        if exists(mask):  # 如果存在掩码
            x = x.masked_fill(~mask, 0.)  # 使用掩码填充张量

        x = self.proj(x)  # 卷积操作

        if exists(mask):  # 如果存在掩码
            x = x.masked_fill(~mask, 0.)  # 使用掩码填充张量

        x = self.norm(x)  # 像素归一化
        x = self.act(x)  # 激活函数
        x = self.dropout(x)  # 随机失活

        return x  # 返回处理后的张量

# ResNet 块
class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        dropout = 0.
    ):
        super().__init__()  # 调用父类初始化函数
        dim_out = default(dim_out, dim)  # 设置输出维度为输入维度
        self.block1 = Block(dim, dim_out, dropout = dropout)  # 基本块1
        self.block2 = Block(dim_out, dim_out, dropout = dropout)  # 基本块2
        self.excite = SqueezeExcite(dim_out)  # Squeeze-and-Excitation 模块
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()  # 残差卷积层

    def forward(
        self,
        x,
        mask = None
    ):
        res = self.residual_conv(x)  # 残差连接
        h = self.block1(x, mask = mask)  # 第一个基本块
        h = self.block2(h, mask = mask)  # 第二个基本块
        h = self.excite(h, mask = mask)  # Squeeze-and-Excitation
        return h + res  # 返回残差连接结果

# gateloop 层

# 门循环块
class GateLoopBlock(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        use_heinsen = True
    # 初始化函数，继承父类的初始化方法
    ):
        # 初始化一个空的模块列表
        super().__init__()
        self.gateloops = ModuleList([])

        # 根据深度循环创建 SimpleGateLoopLayer 层，并添加到模块列表中
        for _ in range(depth):
            gateloop = SimpleGateLoopLayer(dim = dim, use_heinsen = use_heinsen)
            self.gateloops.append(gateloop)

    # 前向传播函数
    def forward(
        self,
        x,
        cache = None
    ):
        # 检查是否接收到缓存
        received_cache = exists(cache)

        # 检查输入张量是否为空
        if is_tensor_empty(x):
            return x, None

        # 如果接收到缓存，则将输入张量分为前面部分和最后一个元素
        if received_cache:
            prev, x = x[:, :-1], x[:, -1:]

        # 如果缓存为空，则初始化为空列表
        cache = default(cache, [])
        # 将缓存转换为迭代器
        cache = iter(cache)

        # 存储新的缓存
        new_caches = []
        # 遍历每个 SimpleGateLoopLayer 层
        for gateloop in self.gateloops:
            # 从缓存中获取当前层的缓存
            layer_cache = next(cache, None)
            # 调用当前层的前向传播方法，返回输出和新的缓存
            out, new_cache = gateloop(x, cache = layer_cache, return_cache = True)
            new_caches.append(new_cache)
            # 更新输入张量
            x = x + out

        # 如果接收到缓存，则将之前分离的部分和当前输出拼接在一起
        if received_cache:
            x = torch.cat((prev, x), dim = -2)

        # 返回更新后的输入张量和新的缓存列表
        return x, new_caches
# 主要类

# 使用装饰器 @save_load(version = __version__)，保存和加载模型版本信息
class MeshAutoencoder(Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        num_discrete_coors = 128,  # 离散坐标数量
        coor_continuous_range: Tuple[float, float] = (-1., 1.),  # 连续坐标范围
        dim_coor_embed = 64,  # 坐标嵌入维度
        num_discrete_area = 128,  # 离散区域数量
        dim_area_embed = 16,  # 区域嵌入维度
        num_discrete_normals = 128,  # 离散法线数量
        dim_normal_embed = 64,  # 法线嵌入维度
        num_discrete_angle = 128,  # 离散角度数量
        dim_angle_embed = 16,  # 角度嵌入维度
        encoder_dims_through_depth: Tuple[int, ...] = (  # 编码器深度维度
            64, 128, 256, 256, 576
        ),
        init_decoder_conv_kernel = 7,  # 初始化解码器卷积核大小
        decoder_dims_through_depth: Tuple[int, ...] = (  # 解码器深度维度
            128, 128, 128, 128,
            192, 192, 192, 192,
            256, 256, 256, 256, 256, 256,
            384, 384, 384
        ),
        dim_codebook = 192,  # 代码簿维度
        num_quantizers = 2,  # 量化器数量
        codebook_size = 16384,  # 代码簿大小
        use_residual_lfq = True,  # 是否使用最新的无查找量化
        rq_kwargs: dict = dict(  # 量化器关键字参数
            quantize_dropout = True,
            quantize_dropout_cutoff_index = 1,
            quantize_dropout_multiple_of = 1,
        ),
        rvq_kwargs: dict = dict(  # RVQ关键字参数
            kmeans_init = True,
            threshold_ema_dead_code = 2,
        ),
        rlfq_kwargs: dict = dict(  # RLFQ关键字参数
            frac_per_sample_entropy = 1.
        ),
        rvq_stochastic_sample_codes = True,  # RVQ是否随机采样代码
        sageconv_kwargs: dict = dict(  # SageConv关键字参数
            normalize = True,
            project = True
        ),
        commit_loss_weight = 0.1,  # 提交损失权重
        bin_smooth_blur_sigma = 0.4,  # 模糊离散坐标位置的独热编码
        attn_encoder_depth = 0,  # 注意力编码器深度
        attn_decoder_depth = 0,  # 注意力解码器深度
        local_attn_kwargs: dict = dict(  # 本地注意力关键字参数
            dim_head = 32,
            heads = 8
        ),
        local_attn_window_size = 64,  # 本地注意力窗口大小
        linear_attn_kwargs: dict = dict(  # 线性注意力关键字参数
            dim_head = 8,
            heads = 16
        ),
        use_linear_attn = True,  # 是否使用线性注意力
        pad_id = -1,  # 填充ID
        flash_attn = True,  # 闪光注意力
        attn_dropout = 0.,  # 注意力丢弃率
        ff_dropout = 0.,  # 前馈丢弃率
        resnet_dropout = 0,  # ResNet丢弃率
        checkpoint_quantizer = False,  # 检查点量化器
        quads = False  # 四边形
    @beartype
    def encode(
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],  # 顶点
        faces:            TensorType['b', 'nf', 'nvf', int],  # 面
        face_edges:       TensorType['b', 'e', 2, int],  # 面边
        face_mask:        TensorType['b', 'nf', bool],  # 面掩码
        face_edges_mask:  TensorType['b', 'e', bool],  # 面边掩码
        return_face_coordinates = False  # 返回面坐标
        """
        einops:
        b - batch
        nf - number of faces
        nv - number of vertices (3)
        nvf - number of vertices per face (3 or 4) - triangles vs quads
        c - coordinates (3)
        d - embed dim
        """

        # 获取顶点的批次、数量、坐标和设备信息
        batch, num_vertices, num_coors, device = *vertices.shape, vertices.device
        # 获取面的批次、数量和每个面的顶点数
        _, num_faces, num_vertices_per_face = faces.shape

        # 断言每个面的顶点数与预设的相同
        assert self.num_vertices_per_face == num_vertices_per_face

        # 根据 face_mask 对 faces 进行填充，将非有效面的值设为 0
        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)

        # 获取连续的面坐标
        face_coords = get_at('b [nv] c, b nf mv -> b nf mv c', vertices, face_without_pad)

        # 计算派生特征并嵌入
        derived_features = get_derived_face_features(face_coords)

        # 将角度离散化并嵌入
        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)

        # 将面积离散化并嵌入
        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)

        # 将法线离散化并嵌入
        discrete_normal = self.discretize_normals(derived_features['normals'])
        normal_embed = self.normal_embed(discrete_normal)

        # 为面坐标嵌入离散化顶点
        discrete_face_coords = self.discretize_face_coords(face_coords)
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)')

        # 对所有特征进行组合并投影到模型维度
        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed], 'b nf *')
        face_embed = self.project_in(face_embed)

        # 处理变长的特征，使用 masked_select 和 masked_scatter
        face_index_offsets = reduce(face_mask.long(), 'b nf -> b', 'sum')
        face_index_offsets = F.pad(face_index_offsets.cumsum(dim = 0), (1, -1), value = 0)
        face_index_offsets = rearrange(face_index_offsets, 'b -> b 1 1')

        face_edges = face_edges + face_index_offsets
        face_edges = face_edges[face_edges_mask]
        face_edges = rearrange(face_edges, 'be ij -> ij be')

        orig_face_embed_shape = face_embed.shape[:2]

        face_embed = face_embed[face_mask]

        # 初始 sage conv 后跟激活和规范化
        face_embed = self.init_sage_conv(face_embed, face_edges)
        face_embed = self.init_encoder_act_and_norm(face_embed)

        # 对每个编码器进行操作
        for conv in self.encoders:
            face_embed = conv(face_embed, face_edges)

        shape = (*orig_face_embed_shape, face_embed.shape[-1])

        face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed)

        # 对每个编码器的注意力块进行操作
        for linear_attn, attn, ff in self.encoder_attn_blocks:
            if exists(linear_attn):
                face_embed = linear_attn(face_embed, mask = face_mask) + face_embed

            face_embed = attn(face_embed, mask = face_mask) + face_embed
            face_embed = ff(face_embed) + face_embed

        # 如果不需要返回面坐标，则返回 face_embed
        if not return_face_coordinates:
            return face_embed

        # 否则返回 face_embed 和离散面坐标
        return face_embed, discrete_face_coords

    @beartype
    def quantize(
        self,
        *,
        faces: TensorType['b', 'nf', 'nvf', int],
        face_mask: TensorType['b', 'n', bool],
        face_embed: TensorType['b', 'nf', 'd', float],
        pad_id = None,
        rvq_sample_codebook_temp = 1.
    ):
        # 设置 pad_id 为默认值或者 self.pad_id
        pad_id = default(pad_id, self.pad_id)
        # 获取 batch, num_faces, device
        batch, num_faces, device = *faces.shape[:2], faces.device

        # 获取 faces 中最大的顶点索引
        max_vertex_index = faces.amax()
        # 计算顶点数量
        num_vertices = int(max_vertex_index.item() + 1)

        # 对 face_embed 进行维度投影
        face_embed = self.project_dim_codebook(face_embed)
        # 重新排列 face_embed 的维度
        face_embed = rearrange(face_embed, 'b nf (nvf d) -> b nf nvf d', nvf = self.num_vertices_per_face)

        # 获取顶点维度
        vertex_dim = face_embed.shape[-1]
        # 创建全零的顶点张量
        vertices = torch.zeros((batch, num_vertices, vertex_dim), device = device)

        # 创建 pad 顶点，用于变长的面
        pad_vertex_id = num_vertices
        vertices = pad_at_dim(vertices, (0, 1), dim = -2, value = 0.)

        # 根据 face_mask 对 faces 进行填充
        faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_vertex_id)

        # 准备用于 scatter mean 的 faces_with_dim
        faces_with_dim = repeat(faces, 'b nf nvf -> b (nf nvf) d', d = vertex_dim)

        # 重新排列 face_embed 的维度
        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        # scatter mean
        averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim = -2)

        # 掩码掉空顶点令牌
        mask = torch.ones((batch, num_vertices + 1), device = device, dtype = torch.bool)
        mask[:, -1] = False

        # rvq 特定的参数
        quantize_kwargs = dict(mask = mask)

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp = rvq_sample_codebook_temp)

        # 一个使其可内存检查点的量化函数
        def quantize_wrapper_fn(inp):
            unquantized, quantize_kwargs = inp
            return self.quantizer(unquantized, **quantize_kwargs)

        # 可能检查点量化函数
        if self.checkpoint_quantizer:
            quantize_wrapper_fn = partial(checkpoint, quantize_wrapper_fn, use_reentrant = False)

        # 剩余 VQ
        quantized, codes, commit_loss = quantize_wrapper_fn((averaged_vertices, quantize_kwargs))

        # 将量化后的顶点收集回 faces 进行解码
        face_embed_output = get_at('b [n] d, b nf nvf -> b nf (nvf d)', quantized, faces)

        # 顶点代码也需要被收集以便按面序组织，用于自回归学习
        codes_output = get_at('b [n] q, b nf nvf -> b (nf nvf) q', codes, faces)

        # 确保输出的代码具有此填��
        face_mask = repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf = self.num_vertices_per_face)
        codes_output = codes_output.masked_fill(~face_mask, self.pad_id)

        # 输出量化、代码以及承诺损失
        return face_embed_output, codes_output, commit_loss

    @beartype
    def decode(
        self,
        quantized: TensorType['b', 'n', 'd', float],
        face_mask:  TensorType['b', 'n', bool]
    ):
        # 重新排列 face_mask 的维度
        conv_face_mask = rearrange(face_mask, 'b n -> b 1 n')

        x = quantized

        for linear_attn, attn, ff in self.decoder_attn_blocks:
            if exists(linear_attn):
                x = linear_attn(x, mask = face_mask) + x

            x = attn(x, mask = face_mask) + x
            x = ff(x) + x

        # 重新排列 x 的维度
        x = rearrange(x, 'b n d -> b d n')
        x = x.masked_fill(~conv_face_mask, 0.)
        x = self.init_decoder_conv(x)

        for resnet_block in self.decoders:
            x = resnet_block(x, mask = conv_face_mask)

        return rearrange(x, 'b d n -> b n d')

    @beartype
    @torch.no_grad()
    def decode_from_codes_to_faces(
        self,
        codes: Tensor,
        face_mask: Optional[TensorType['b', 'n', bool]] = None,
        return_discrete_codes = False
    ):
        # 重新排列代码，将 'b ...' 转换为 'b (...)'
        codes = rearrange(codes, 'b ... -> b (...)')

        # 如果 face_mask 不存在，则将其设为不等于 self.pad_id 的代码
        if not exists(face_mask):
            face_mask = reduce(codes != self.pad_id, 'b (nf nvf q) -> b nf', 'all', nvf = self.num_vertices_per_face, q = self.num_quantizers)

        # 处理不同的代码形状

        # 重新排列代码，将 'b (n q)' 转换为 'b n q'
        codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)

        # 解码

        # 从索引获取量化值
        quantized = self.quantizer.get_output_from_indices(codes)
        # 重新排列量化值，将 'b (nf nvf) d' 转换为 'b nf (nvf d)'
        quantized = rearrange(quantized, 'b (nf nvf) d -> b nf (nvf d)', nvf = self.num_vertices_per_face)

        # 解码
        decoded = self.decode(
            quantized,
            face_mask = face_mask
        )

        # 将未被 face_mask 遮罩的部分填充为 0
        decoded = decoded.masked_fill(~face_mask[..., None], 0.)
        # 将 decoded 转换为坐标 logits
        pred_face_coords = self.to_coor_logits(decoded)

        # 取最大值的索引
        pred_face_coords = pred_face_coords.argmax(dim = -1)

        # 重新排列 pred_face_coords，将 '... (v c)' 转换为 '... v c', v 为 self.num_vertices_per_face
        pred_face_coords = rearrange(pred_face_coords, '... (v c) -> ... v c', v = self.num_vertices_per_face)

        # 转换回连续空间

        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )

        # 使用 nan 进行遮罩处理

        continuous_coors = continuous_coors.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1 1'), float('nan'))

        # 如果不返回离散代码，则返回 continuous_coors 和 face_mask
        if not return_discrete_codes:
            return continuous_coors, face_mask

        # 返回 continuous_coors、pred_face_coords 和 face_mask

        return continuous_coors, pred_face_coords, face_mask

    @torch.no_grad()
    def tokenize(self, vertices, faces, face_edges = None, **kwargs):
        # 确保 kwargs 中不存在 'return_codes'
        assert 'return_codes' not in kwargs

        inputs = [vertices, faces, face_edges]
        inputs = [*filter(exists, inputs)]
        ndims = {i.ndim for i in inputs}

        # 确保输入的张量维度相同
        assert len(ndims) == 1
        batch_less = first(list(ndims)) == 2

        # 如果 batch_less 为 True，则将输入转换为批量大小为 1 的形式
        if batch_less:
            inputs = [rearrange(i, '... -> 1 ...') for i in inputs]

        input_kwargs = dict(zip(['vertices', 'faces', 'face_edges'], inputs))

        self.eval()

        # 调用 forward 方法，返回代码

        codes = self.forward(
            **input_kwargs,
            return_codes = True,
            **kwargs
        )

        # 如果 batch_less 为 True，则重新排列代码
        if batch_less:
            codes = rearrange(codes, '1 ... -> ...')

        return codes

    @beartype
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, float],
        faces:          TensorType['b', 'nf', 'nvf', int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        return_codes = False,
        return_loss_breakdown = False,
        return_recon_faces = False,
        only_return_recon_faces = False,
        rvq_sample_codebook_temp = 1.
        ):
        # 如果未提供面边缘数据，则从面数据中推导出面边缘数据
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)

        # 获取面的数量、面边缘的数量以及设备信息
        num_faces, num_face_edges, device = faces.shape[1], face_edges.shape[1], faces.device

        # 创建面的掩码，标记哪些位置是有效的面
        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        # 创建面边缘的掩码，标记哪些位置是有效的面边缘
        face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')

        # 编码输入数据，获取编码结果和面坐标
        encoded, face_coordinates = self.encode(
            vertices = vertices,
            faces = faces,
            face_edges = face_edges,
            face_edges_mask = face_edges_mask,
            face_mask = face_mask,
            return_face_coordinates = True
        )

        # 量化编码结果，获取量化后的数据、编码和损失
        quantized, codes, commit_loss = self.quantize(
            face_embed = encoded,
            faces = faces,
            face_mask = face_mask,
            rvq_sample_codebook_temp = rvq_sample_codebook_temp
        )

        # 如果需要返回编码结果
        if return_codes:
            assert not return_recon_faces, 'cannot return reconstructed faces when just returning raw codes'

            # 将编码结果填充到掩码之外的位置
            codes = codes.masked_fill(~repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf = self.num_vertices_per_face), self.pad_id)
            return codes

        # 解码量化后的数据，获取解码结果
        decode = self.decode(
            quantized,
            face_mask = face_mask
        )

        # 将解码结果转换为坐标概率
        pred_face_coords = self.to_coor_logits(decode)

        # 如果需要计算重构的面
        if return_recon_faces or only_return_recon_faces:

            # 将坐标概率反离散化为坐标
            recon_faces = undiscretize(
                pred_face_coords.argmax(dim = -1),
                num_discrete = self.num_discrete_coors,
                continuous_range = self.coor_continuous_range,
            )

            # 重排重构的面数据
            recon_faces = rearrange(recon_faces, 'b nf (nvf c) -> b nf nvf c', nvf = self.num_vertices_per_face)
            face_mask = rearrange(face_mask, 'b nf -> b nf 1 1')
            recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))
            face_mask = rearrange(face_mask, 'b nf 1 1 -> b nf')

        # 如果只需要返回重构的面数据
        if only_return_recon_faces:
            return recon_faces

        # 准备重构损失
        pred_face_coords = rearrange(pred_face_coords, 'b ... c -> b c (...)')
        face_coordinates = rearrange(face_coordinates, 'b ... -> b 1 (...)')

        # 计算重构损失，使用局部平滑
        with autocast(enabled = False):
            pred_log_prob = pred_face_coords.log_softmax(dim = 1)

            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, face_coordinates, 1.)

            if self.bin_smooth_blur_sigma >= 0.:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma = self.bin_smooth_blur_sigma)

            # 使用局部平滑的交叉熵损失
            recon_losses = (-target_one_hot * pred_log_prob).sum(dim = 1)

            face_mask = repeat(face_mask, 'b nf -> b (nf r)', r = self.num_vertices_per_face * 3)
            recon_loss = recon_losses[face_mask].mean()

        # 计算总损失
        total_loss = recon_loss + \
                     commit_loss.sum() * self.commit_loss_weight

        # 如果需要计算损失细分
        loss_breakdown = (recon_loss, commit_loss)

        # 返回逻辑
        if not return_loss_breakdown:
            if not return_recon_faces:
                return total_loss

            return recon_faces, total_loss

        if not return_recon_faces:
            return total_loss, loss_breakdown

        return recon_faces, total_loss, loss_breakdown
# 定义一个 MeshTransformer 类，用于处理网格数据的转换
@save_load(version = __version__)
class MeshTransformer(Module):
    # 初始化 MeshTransformer 类
    @beartype
    def __init__(
        self,
        autoencoder: MeshAutoencoder, # 接收一个 MeshAutoencoder 对象作为参数
        *,
        dim: Union[int, Tuple[int, int]] = 512, # 维度参数，默认为 512
        max_seq_len = 8192, # 最大序列长度，默认为 8192
        flash_attn = True, # 是否使用 flash attention，默认为 True
        attn_depth = 12, # 注意力层的深度，默认为 12
        attn_dim_head = 64, # 注意力头的维度，默认为 64
        attn_heads = 16, # 注意力头的数量，默认为 16
        attn_kwargs: dict = dict( # 注意力参数字典，默认包含 ff_glu 和 num_mem_kv 两个键值对
            ff_glu = True,
            num_mem_kv = 4
        ),
        cross_attn_num_mem_kv = 4, # 用于防止在丢弃文本条件时出现 NaN 的交叉注意力数目
        dropout = 0., # 丢弃概率，默认为 0
        coarse_pre_gateloop_depth = 2, # 粗粒度预门循环的深度，默认为 2
        fine_pre_gateloop_depth = 2, # 细粒度预门循环的深度，默认为 2
        gateloop_use_heinsen = False, # 是否使用 Heinsen 门循环，默认为 False
        fine_attn_depth = 2, # 细粒度注意力层的深度，默认为 2
        fine_attn_dim_head = 32, # 细粒度注意力头的维度，默认为 32
        fine_attn_heads = 8, # 细粒度注意力头的数量，默认为 8
        pad_id = -1, # 填充 ID，默认为 -1
        condition_on_text = False, # 是否基于文本条件，默认为 False
        text_condition_model_types = ('t5',), # 文本条件模型类型，默认为 ('t5',)
        text_condition_cond_drop_prob = 0.25, # 文本条件丢弃概率，默认为 0.25
        quads = False # 是否使用四元组，默认为 False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 如果不是四边形，则每个面的顶点数为3，否则为4
        self.num_vertices_per_face = 3 if not quads else 4

        # 断言自动编码器和转换器必须都支持相同类型的网格（全三角形或全四边形）
        assert autoencoder.num_vertices_per_face == self.num_vertices_per_face, 'autoencoder and transformer must both support the same type of mesh (either all triangles, or all quads)'

        # 如果维度是整数，则将其转换为元组
        dim, dim_fine = (dim, dim) if isinstance(dim, int) else dim

        # 设置自动编码器，并将其梯度设置为False
        self.autoencoder = autoencoder
        set_module_requires_grad_(autoencoder, False)

        # 获取自动编码器的代码本大小和量化器数量
        self.codebook_size = autoencoder.codebook_size
        self.num_quantizers = autoencoder.num_quantizers

        # 初始化起始标记和结束标记
        self.sos_token = nn.Parameter(torch.randn(dim_fine))
        self.eos_token_id = self.codebook_size

        # 断言最大序列长度必须能够被（3 x self.num_quantizers）整除
        assert divisible_by(max_seq_len, self.num_vertices_per_face * self.num_quantizers), f'max_seq_len ({max_seq_len}) must be divisible by (3 x {self.num_quantizers}) = {3 * self.num_quantizers}' # 3 or 4 vertices per face, with D codes per vertex

        # 初始化标记嵌入
        self.token_embed = nn.Embedding(self.codebook_size + 1, dim)

        # 初始化量化级别嵌入和顶点嵌入
        self.quantize_level_embed = nn.Parameter(torch.randn(self.num_quantizers, dim))
        self.vertex_embed = nn.Parameter(torch.randn(self.num_vertices_per_face, dim))

        # 初始化绝对位置嵌入
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        # 设置最大序列长度
        self.max_seq_len = max_seq_len

        # 文本条件
        self.condition_on_text = condition_on_text
        self.conditioner = None

        cross_attn_dim_context = None

        # 如果有文本条件，则初始化文本嵌入返回器
        if condition_on_text:
            self.conditioner = TextEmbeddingReturner(
                model_types = text_condition_model_types,
                cond_drop_prob = text_condition_cond_drop_prob
            )
            cross_attn_dim_context = self.conditioner.dim_latent

        # 用于总结每个面的顶点
        self.to_face_tokens = nn.Sequential(
            nn.Linear(self.num_quantizers * self.num_vertices_per_face * dim, dim),
            nn.LayerNorm(dim)
        )

        # 初始化粗粒度门环块
        self.coarse_gateloop_block = GateLoopBlock(dim, depth = coarse_pre_gateloop_depth, use_heinsen = gateloop_use_heinsen) if coarse_pre_gateloop_depth > 0 else None

        # 主自回归注意力网络，关注面标记
        self.decoder = Decoder(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            cross_attend = condition_on_text,
            cross_attn_dim_context = cross_attn_dim_context,
            cross_attn_num_mem_kv = cross_attn_num_mem_kv,
            **attn_kwargs
        )

        # 如果需要，从粗到细的投影
        self.maybe_project_coarse_to_fine = nn.Linear(dim, dim_fine) if dim != dim_fine else nn.Identity()

        # 解决注意力中的一个弱点
        self.fine_gateloop_block = GateLoopBlock(dim, depth = fine_pre_gateloop_depth) if fine_pre_gateloop_depth > 0 else None

        # 解码顶点，两阶层次
        self.fine_decoder = Decoder(
            dim = dim_fine,
            depth = fine_attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            **attn_kwargs
        )

        # 转换为逻辑值
        self.to_logits = nn.Linear(dim_fine, self.codebook_size + 1)

        # 填充ID，强制自动编码器使用转换器中给定的相同填充ID
        self.pad_id = pad_id
        autoencoder.pad_id = pad_id

    @property
    def device(self):
        # 返回参数的设备
        return next(self.parameters()).device

    @beartype
    @torch.no_grad()
    # 将文本嵌入到向量空间中
    def embed_texts(self, texts: Union[str, List[str]]):
        # 检查是否为单个文本
        single_text = not isinstance(texts, list)
        # 如果是单个文本，则转换为列表
        if single_text:
            texts = [texts]

        # 断言条件器存在
        assert exists(self.conditioner)
        # 嵌入文本到向量空间中并分离计算图
        text_embeds = self.conditioner.embed_texts(texts).detach()

        # 如果是单个文本，则取第一个文本的嵌入向量
        if single_text:
            text_embeds = text_embeds[0]

        # 返回文本的嵌入向量
        return text_embeds

    # 生成文本
    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        prompt: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        return_codes = False,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_scale = 1.,
        cache_kv = True,
        max_seq_len = None,
        face_coords_to_file: Optional[Callable[[Tensor], Any]] = None
    ):
        # 设置最大序列长度为默认值或者给定的值
        max_seq_len = default(max_seq_len, self.max_seq_len)

        # 如果存在提示信息
        if exists(prompt):
            # 确保批处理大小不存在
            assert not exists(batch_size)

            # 重新排列提示信息的维度
            prompt = rearrange(prompt, 'b ... -> b (...)')
            # 确保提示信息的长度不超过最大序列长度
            assert prompt.shape[-1] <= self.max_seq_len

            # 设置批处理大小为提示信息的批次大小
            batch_size = prompt.shape[0]

        # 如果需要根据文本条件生成
        if self.condition_on_text:
            # 确保文本或文本嵌入存在，且二者只能存在一个
            assert exists(texts) ^ exists(text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'
            # 如果文本存在，则生成文本嵌入
            if exists(texts):
                text_embeds = self.embed_texts(texts)

            # 设置批处理大小为文本嵌入的批次大小
            batch_size = default(batch_size, text_embeds.shape[0])

        # 设置批处理大小为默认值或者1
        batch_size = default(batch_size, 1)

        # 初始化代码张量
        codes = default(prompt, torch.empty((batch_size, 0), dtype = torch.long, device = self.device))

        # 获取当前代码长度
        curr_length = codes.shape[-1]

        # 初始化缓存
        cache = (None, None)

        # 循环生成序列
        for i in tqdm(range(curr_length, max_seq_len)):

            # 判断是否可以生成结束符
            can_eos = i != 0 and divisible_by(i, self.num_quantizers * self.num_vertices_per_face)  # 只允许在每个面的末尾生成结束符，定义为具有 D 个残差 VQ 代码的 3 或 4 个顶点的面

            # 在代码上进行前向传播
            output = self.forward_on_codes(
                codes,
                text_embeds = text_embeds,
                return_loss = False,
                return_cache = cache_kv,
                append_eos = False,
                cond_scale = cond_scale,
                cfg_routed_kwargs = dict(
                    cache = cache
                )
            )

            # 如果使用缓存
            if cache_kv:
                logits, cache = output

                if cond_scale == 1.:
                    cache = (cache, None)
            else:
                logits = output

            logits = logits[:, -1]

            # ���果不能生成结束符，则将结束符位置的概率设为最小值
            if not can_eos:
                logits[:, -1] = -torch.finfo(logits.dtype).max

            # 过滤logits
            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            # 根据温度参数进行采样
            if temperature == 0.:
                sample = filtered_logits.argmax(dim = -1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            # 将采样结果添加到代码中
            codes, _ = pack([codes, sample], 'b *')

            # 检查是否所有行都有结束符以终止
            is_eos_codes = (codes == self.eos_token_id)

            if is_eos_codes.any(dim = -1).all():
                break

        # 掩盖第一个结束符后的所有内容
        mask = is_eos_codes.float().cumsum(dim = -1) >= 1
        codes = codes.masked_fill(mask, self.pad_id)

        # 移除可能存在的额外结束符
        code_len = codes.shape[-1]
        round_down_code_len = code_len // self.num_quantizers * self.num_quantizers
        codes = codes[:, :round_down_code_len]

        # 如果需要返回代码，则返回原始残差量化器代码
        if return_codes:
            codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)
            return codes

        # 将自动编码器设置为评估模式
        self.autoencoder.eval()
        # 从代码解码到面的坐标
        face_coords, face_mask = self.autoencoder.decode_from_codes_to_faces(codes)

        # 如果不存在面坐标到文件的映射，则返回面坐标和面掩码
        if not exists(face_coords_to_file):
            return face_coords, face_mask

        # 生成面坐标到文件的映射
        files = [face_coords_to_file(coords[mask]) for coords, mask in zip(face_coords, face_mask)]
        return files

    # 前向传播函数
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, int],
        faces:          TensorType['b', 'nf', 'nvf', int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        codes:          Optional[Tensor] = None,
        cache:          Optional[LayerIntermediates] = None,
        **kwargs
    # 如果未提供codes，则调用autoencoder的tokenize方法生成codes
    ):
        # 如果codes不存在，则使用autoencoder的tokenize方法生成codes
        if not exists(codes):
            codes = self.autoencoder.tokenize(
                vertices = vertices,
                faces = faces,
                face_edges = face_edges
            )

        # 调用forward_on_codes方法，传入codes和其他参数
        return self.forward_on_codes(codes, cache = cache, **kwargs)

    # 标记为classifier_free_guidance的方法，用于在codes上执行前向传播
    def forward_on_codes(
        self,
        codes = None,
        return_loss = True,
        return_cache = False,
        append_eos = True,
        cache = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob = None
```