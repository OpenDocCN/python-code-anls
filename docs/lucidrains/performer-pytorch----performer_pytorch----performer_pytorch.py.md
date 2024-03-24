# `.\lucidrains\performer-pytorch\performer_pytorch\performer_pytorch.py`

```py
# 导入数学库
import math
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 torch 中导入 nn 模块
from torch import nn
# 从 torch.cuda.amp 中导入 autocast 函数
from torch.cuda.amp import autocast
# 从 einops 中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 从 functools 中导入 partial 函数
from functools import partial
# 从 contextlib 中导入 contextmanager 函数
from contextlib import contextmanager

# 导入自定义的 local_attention 模块
from local_attention import LocalAttention
# 导入自定义的 axial_positional_embedding 模块
from axial_positional_embedding import AxialPositionalEmbedding
# 导入 performer_pytorch 中的 reversible 模块
from performer_pytorch.reversible import ReversibleSequence, SequentialSequence

# 从 distutils.version 中导入 LooseVersion 类
from distutils.version import LooseVersion

# 检查 torch 版本是否大于等于 1.8.0
TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')

try:
    # 尝试导入 apex 库中的 amp 模块
    from apex import amp
    APEX_AVAILABLE = True
except:
    # 如果导入失败，则将 APEX_AVAILABLE 设为 False
    APEX_AVAILABLE = False

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 判断张量是否为空
def empty(tensor):
    return tensor.numel() == 0

# 返回 val 或默认值 d
def default(val, d):
    return val if exists(val) else d

# 空上下文管理器
@contextmanager
def null_context():
    yield

# 将 val 转换为元组
def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

# 获取模块的设备
def get_module_device(module):
    return next(module.parameters()).device

# 查找 nn_module 中的指定类型的模块
def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

# 始终返回指定值的模块
class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

# token 移动的辅助函数和类

# 将张量 t 沿指定方向移动指定量 amount
def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

# 预先移动 token 的类
class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# 核函数

# 从 jax 转录到 pytorch 的 softmax 核函数
def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)

# 通用核函数
def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    # 将data的数据类型转换为与data_prime相同的数据类型，并返回结果
    return data_prime.type_as(data)
# 生成一个正交矩阵块
def orthogonal_matrix_chunk(cols, device = None):
    # 生成一个随机的矩阵
    unstructured_block = torch.randn((cols, cols), device = device)
    # 使用 QR 分解得到正交矩阵 q
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some = True)
    # 将 q 和 r 移动到指定设备上
    q, r = map(lambda t: t.to(device), (q, r))
    # 返回 q 的转置
    return q.t()

# 生成一个高斯正交随机矩阵
def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    # 计算完整块的数量
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    # 生成完整块
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    # 处理剩余的行
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    # 拼接所有块
    final_matrix = torch.cat(block_list)

    # 根据 scaling 参数生成 multiplier
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    # 返回乘积结果
    return torch.diag(multiplier) @ final_matrix

# 线性注意力类，使用 softmax 核

# 非因果线性注意力
def linear_attention(q, k, v):
    # 计算 k 的累加和
    k_cumsum = k.sum(dim = -2)
    # 计算 D_inv
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    # 计算上下文
    context = torch.einsum('...nd,...ne->...de', k, v)
    # 计算输出
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

# 高效因果线性注意力，由 EPFL 创建
# TODO: 重写 EPFL 的 CUDA 核以进行混合精度，并删除半精度到单精度的转换
def causal_linear_attention(q, k, v, eps = 1e-6):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out

# 低效因果线性注意力，不包含 CUDA 代码，供读者参考
# 未被使用
def causal_linear_attention_noncuda(q, k, v, chunk_size = 128, eps = 1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim = -2)

class FastAttention(nn.Module):
    # 初始化函数，设置注意力头的维度、特征数量、正交缩放、是否因果、是否使用广义注意力、核函数、是否不使用投影
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        # 调用父类的初始化函数
        super().__init__()
        # 如果未指定特征数量，则默认为注意力头维度乘以注意力头维度的对数
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        # 设置注意力头的维度、特征数量、正交缩放
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        # 创建投影矩阵的函数，使用高斯正交随机矩阵
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        # 生成投影矩阵并注册为缓冲区
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        # 设置是否使用广义注意力、核函数
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # 如果设置为不使用投影，则不进行投影，直接对查询和键进行 softmax 处理
        if this is turned on, no projection will be used
        queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        # 设置是否因果，如果是因果的则使用因果线性注意力函数
        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    # 重新生成投影矩阵的函数，用于在训练过程中更新投影矩阵
    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        # 生成新的投影矩阵并复制到原有的投影矩阵中
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    # 前向传播函数，接收查询、键、值作为输入，返回注意力计算结果
    def forward(self, q, k, v):
        device = q.device

        # 如果设置为不使用投影，则直接对查询和键进行 softmax 处理
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        # 如果设置为使用广义注意力，则使用广义核函数进行计算
        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        # 否则使用 softmax 核函数进行计算
        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        # 根据是否因果选择不同的注意力函数进行计算
        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out
# 用于跟踪何时更新投影的模块

class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projections_(self):
        # 修正投影
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance

        if not self.training:
            return

        # 如果存在特征重绘间隔并且自上次重绘以来的调用次数大于等于特征重绘间隔
        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(model)

            # 查找模型中的 FastAttention 模块
            fast_attentions = find_modules(model, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplemented

# 类

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class PreScaleNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)

class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        heads = 8,
        dim_head = 64,
        local_heads = 0,
        local_window_size = 256,
        nb_features = None,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        dropout = 0.,
        no_projection = False,
        qkv_bias = False,
        attn_out_bias = True
    # 初始化函数，继承父类的初始化方法
    def __init__(
        super().__init__()
        # 断言维度必须能够被头数整除
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        # 计算每个头的维度
        dim_head = default(dim_head, dim // heads)
        # 计算内部维度
        inner_dim = dim_head * heads
        # 创建快速注意力对象
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)

        # 设置头数和全局头数
        self.heads = heads
        self.global_heads = heads - local_heads
        # 如果有局部头数，则创建局部注意力对象
        self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        # 创建线性层，用于将输入转换为查询、键、值
        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    # 前向传播函数
    def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, **kwargs):
        # 获取输入张量的形状信息
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        # 判断是否存在上下文信息
        cross_attend = exists(context)

        # 设置默认上下文和上下文掩码
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        # 将输入张量转换为查询、键、值
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        # 重排查询、键、值张量的维度
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        # 存储注意力输出
        attn_outs = []

        # 如果查询不为空
        if not empty(q):
            # 如果存在上下文掩码，则对值进行掩码
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            # 如果存在位置编码且不是交叉注意力，则应用旋转位置编码
            if exists(pos_emb) and not cross_attend:
                q, k = apply_rotary_pos_emb(q, k, pos_emb)

            # 使用快速注意力计算输出
            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        # 如果局部查询不为空
        if not empty(lq):
            # 断言不支持交叉注意力和局部注意力同时存在
            assert not cross_attend, 'local attention is not compatible with cross attention'
            # 使用局部注意力计算输出
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        # 拼接所有注意力输出
        out = torch.cat(attn_outs, dim = 1)
        # 重排输出张量的维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 经过输出线性层
        out =  self.to_out(out)
        # 使用丢弃层
        return self.dropout(out)
# 定义 SelfAttention 类，继承自 Attention 类
class SelfAttention(Attention):
    # 重写 forward 方法，接收任意参数和关键字参数 context，默认为 None
    def forward(self, *args, context = None, **kwargs):
        # 断言 context 不存在，即 self attention 不应该接收 context
        assert not exists(context), 'self attention should not receive context'
        # 调用父类的 forward 方法，传入参数和关键字参数
        return super().forward(*args, **kwargs)

# 定义 CrossAttention 类，继承自 Attention 类
class CrossAttention(Attention):
    # 重写 forward 方法，接收任意参数和关键字参数 context，默认为 None
    def forward(self, *args, context = None, **kwargs):
        # 断言 context 存在，即 cross attention 应该接收 context
        assert exists(context), 'cross attention should receive context'
        # 调用父类的 forward 方法，传入参数、context 和关键字参数
        return super().forward(*args, context = context, **kwargs)

# positional embeddings

# 定义 AbsolutePositionalEmbedding 类，继承自 nn.Module 类
class AbsolutePositionalEmbedding(nn.Module):
    # 初始化方法，接收维度 dim 和最大序列长度 max_seq_len
    def __init__(self, dim, max_seq_len):
        super().__init__()
        # 创建一个 Embedding 层，将最大序列长度和维度作为参数
        self.emb = nn.Embedding(max_seq_len, dim)

    # 前向传播方法，接收输入 x
    def forward(self, x):
        # 生成一个序列长度的张量 t，设备为 x 的设备
        t = torch.arange(x.shape[1], device=x.device)
        # 返回 Embedding 层对 t 的嵌入结果
        return self.emb(t)

# rotary positional embedding helpers

# 定义 rotate_every_two 函数，接收输入 x
def rotate_every_two(x):
    # 重新排列 x 的维度，将最后一维拆分为两个维度
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    # 将 x 拆分为两部分 x1 和 x2
    x1, x2 = x.unbind(dim = -1)
    # 将 x1 和 x2 交换位置并合并成新的张量 x
    x = torch.stack((-x2, x1), dim = -1)
    # 重新排列 x 的维度，将最后两维合并为一维
    return rearrange(x, '... d j -> ... (d j)')

# 定义 apply_rotary_pos_emb 函数，接收查询向量 q、键向量 k 和正弦位置编码 sinu_pos
def apply_rotary_pos_emb(q, k, sinu_pos):
    # 重新排列 sinu_pos 的维度，将第二维拆分为两个维度
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    # 拆分 sinu_pos 为 sin 和 cos
    sin, cos = sinu_pos.unbind(dim = -2)
    # 将 sin 和 cos 扩展为与 q、k 相同的维度
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    # 对 q、k 应用正弦和余弦位置编码
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    # 返回处理后的 q 和 k
    return q, k

# sinusoidal positional embeddings

# 定义 FixedPositionalEmbedding 类，继承自 nn.Module 类
class FixedPositionalEmbedding(nn.Module):
    # 初始化方法，接收维度 dim 和最大序列长度 max_seq_len
    def __init__(self, dim, max_seq_len):
        super().__init__()
        # 计算频率的倒数
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 生成位置张量和频率张量的乘积
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        # 拼接正弦和余弦结果作为位置编码
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        # 将位置编码作为缓冲区注册到模型中
        self.register_buffer('emb', emb)

    # 前向传播方法，接收输入 x
    def forward(self, x):
        # 返回位置编码的子集，维度与输入 x 相匹配
        return self.emb[None, :x.shape[1], :].to(x)

# performer

# 定义 Performer 类，继承自 nn.Module 类
class Performer(nn.Module):
    # 初始化方法，接收多个参数设置
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        ff_glu = False,
        ff_dropout = 0.,
        attn_dropout = 0.,
        cross_attend = False,
        no_projection = False,
        auto_check_redraw = True,
        qkv_bias = True,
        attn_out_bias = True,
        shift_tokens = False
    # 初始化函数，继承父类的初始化方法
    ):
        # 初始化一个空的模块列表
        super().__init__()
        layers = nn.ModuleList([])
        # 将本地注意力头数转换为元组
        local_attn_heads = cast_tuple(local_attn_heads)
        # 如果只有一个本地注意力头数，则复制到每一层
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        # 确保本地注意力头数的长度等于深度
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        # 确保本地注意力头数的值小于总头数
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        # 根据使用的归一化方法选择包装函数
        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        # 遍历每一层
        for _, local_heads in zip(range(depth), local_attn_heads):

            # 创建自注意力层和前馈层
            attn = SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_heads, local_window_size = local_window_size, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)
            ff = Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1)

            # 如果需要移动标记，则对自注意力层和前馈层进行移动
            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))

            # 对自注意力层和前馈层应用包装函数
            attn, ff = map(wrapper_fn, (attn, ff))
            # 将自注意力层和前馈层添加到模块列表中
            layers.append(nn.ModuleList([attn, ff]))

            # 如果不需要跨层注意力，则继续下一层
            if not cross_attend:
                continue

            # 添加跨层注意力和前馈层到模块列表中
            layers.append(nn.ModuleList([
                wrapper_fn(CrossAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

        # 根据是否可逆选择执行类型
        execute_type = ReversibleSequence if reversible else SequentialSequence

        # 设置自注意力和上下文的路由映射
        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        # 创建网络结构
        self.net = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        # 记录何时重新绘制所有注意力层的投影矩阵
        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    # 修正投影矩阵
    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 如果需要自动检查重新绘制，则重新绘制投影矩阵
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)
class PerformerLM(nn.Module):
    # 定义 PerformerLM 类，继承自 nn.Module
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        heads,
        dim_head = 64,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        tie_embed = False,
        rotary_position_emb = True,
        axial_position_emb = False,
        axial_position_shape = None,
        auto_check_redraw = True,
        qkv_bias = False,
        attn_out_bias = False,
        shift_tokens = False
    ):
        # 初始化函数，接收多个参数
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建 token embedding 层

        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)
        # 根据不同的位置编码方式创建位置编码层

        self.dropout = nn.Dropout(emb_dropout)
        # 创建 dropout 层

        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias, attn_out_bias, shift_tokens)
        # 创建 Performer 模型

        self.norm = nn.LayerNorm(dim)
        # 创建 LayerNorm 层

        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None
        # 创建线性层，如果 tie_embed 为 False，则创建线性层，否则为 None

    def check_redraw_projections(self):
        # 检查是否需要重新绘制投影矩阵
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        # 修正投影矩阵
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings = False, **kwargs):
        # 前向传播函数，接收输入 x 和是否返回编码的标志
        b, n, device = *x.shape, x.device
        # 获取输入 x 的形状和设备信息
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        # 断言序列长度小于等于最大序列长度

        # token and positional embeddings
        x = self.token_emb(x)
        # 获取 token embedding
        x += self.pos_emb(x)
        # 添加位置编码

        x = self.dropout(x)
        # 应用 dropout

        # performer layers

        layer_pos_emb = self.layer_pos_emb(x)
        # 获取层级位置编码
        x = self.performer(x, pos_emb = layer_pos_emb, **kwargs)
        # 使用 Performer 模型进行计算

        # norm and to logits
        x = self.norm(x)
        # 应用 LayerNorm

        if return_encodings:
            return x
        # 如果需要返回编码，则直接返回编码

        if exists(self.to_out):
            return self.to_out(x)
        # 如果存在输出层，则返回输出

        return x @ self.token_emb.weight.t()
        # 返回结果
```