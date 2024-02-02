# `so-vits-svc\modules\F0Predictor\fcpe\pcmer.py`

```py
# 导入数学库
import math
# 从 functools 库中导入 partial 函数
from functools import partial
# 导入 torch 库
import torch
# 从 torch.nn.functional 中导入 F
import torch.nn.functional as F
# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 local_attention 中导入 LocalAttention 类
from local_attention import LocalAttention
# 从 torch.nn 中导入 nn
from torch import nn

# 导入 fast_transformers.causal_product.causal_product_cuda 库
#import fast_transformers.causal_product.causal_product_cuda

# 定义 softmax_kernel 函数，接受 data, projection_matrix, is_query, normalize_data, eps, device 参数
def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    # 获取 data 的形状
    b, h, *_ = data.shape
    # (batch size, head, length, model_dim)

    # 根据是否需要归一化数据，计算数据的归一化因子
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    # 计算 ratio
    ratio = (projection_matrix.shape[0] ** -0.5)

    # 将 projection_matrix 重复到与 data 相同的形状
    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    # 将 projection 转换为与 data 相同的数据类型
    projection = projection.type_as(data)

    # 计算 data_dash = w^T x
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    # 计算 diag_data = D**2 
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    # 根据 is_query 判断计算 data_dash
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data + eps))#- torch.max(data_dash)) + eps)

    return data_dash.type_as(data)

# 定义 orthogonal_matrix_chunk 函数，接受 cols, qr_uniform_q, device 参数
def orthogonal_matrix_chunk(cols, qr_uniform_q = False, device = None):
    # 生成随机的 unstructured_block
    unstructured_block = torch.randn((cols, cols), device = device)
    # 对 unstructured_block 进行 QR 分解
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))

    # 如果需要保证 Q 是均匀分布的，则进行调整
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()

# 定义 exists 函数，接受 val 参数
def exists(val):
    return val is not None

# 定义 empty 函数，接受 tensor 参数
def empty(tensor):
    return tensor.numel() == 0

# 定义 default 函数，接受 val, d 参数
def default(val, d):
    # 如果val存在，则返回val，否则返回d
    return val if exists(val) else d
# 定义一个函数，用于将输入值转换为元组，如果输入值不是元组的话
def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

# 定义一个名为 PCmer 的类，继承自 nn.Module
class PCmer(nn.Module):
    """The encoder that is used in the Transformer model."""
    
    # 初始化方法，接受多个参数
    def __init__(self, 
                num_layers,
                num_heads,
                dim_model,
                dim_keys,
                dim_values,
                residual_dropout,
                attention_dropout):
        super().__init__()
        # 初始化类的属性
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout

        # 创建一个包含多个 _EncoderLayer 实例的列表
        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])
        
    #  METHODS  ########################################################################################################
    
    # 前向传播方法，接受输入 phone 和可选的 mask 参数
    def forward(self, phone, mask=None):
        
        # 对输入应用所有层的操作
        for (i, layer) in enumerate(self._layers):
            phone = layer(phone, mask)
        # 返回最终的序列
        return phone


# ==================================================================================================================== #
#  CLASS  _ E N C O D E R  L A Y E R                                                                                   #
# ==================================================================================================================== #


# 定义一个名为 _EncoderLayer 的类，继承自 nn.Module
class _EncoderLayer(nn.Module):
    """One layer of the encoder.
    
    Attributes:
        attn: (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read the input sequence.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanism.
    """
    def __init__(self, parent: PCmer):
        """Creates a new instance of ``_EncoderLayer``.
        
        Args:
            parent (Encoder): The encoder that the layers is created for.
        """
        # 调用父类的构造函数
        super().__init__()
        
        # 初始化 ConformerConvModule 模块
        self.conformer = ConformerConvModule(parent.dim_model)
        # 初始化 LayerNorm 模块
        self.norm = nn.LayerNorm(parent.dim_model)
        # 初始化 Dropout 模块
        self.dropout = nn.Dropout(parent.residual_dropout)
        
        # 初始化 SelfAttention 模块，设置维度和头数
        # causal 参数表示是否使用因果注意力
        self.attn = SelfAttention(dim = parent.dim_model,
                                  heads = parent.num_heads,
                                  causal = False)
        
    #  METHODS  ########################################################################################################

    def forward(self, phone, mask=None):
        # 计算注意力子层
        # 对输入 phone 进行 LayerNorm 处理，然后传入 SelfAttention 模块计算注意力
        phone = phone + (self.attn(self.norm(phone), mask=mask))
        
        # 对输入 phone 进行 ConformerConvModule 处理
        phone = phone + (self.conformer(phone))
        
        # 返回处理后的结果
        return phone 
# 计算卷积核大小的相同填充量
def calc_same_padding(kernel_size):
    # 计算填充量，使得卷积后的输出大小与输入大小相同
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# 辅助类

# Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

# 转置操作
class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

# GLU 激活函数
class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

# 深度可分离卷积
class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# Conformer 模块中的卷积模块
class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            #nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 线性注意力函数
def linear_attention(q, k, v):
    if v is None:
        # 如果输入的 v 为空，则执行以下操作
        out = torch.einsum('...ed,...nd->...ne', k, q)
        # 使用 Einstein Summation Notation 计算张量 k 和 q 的乘积，并返回结果
        return out

    else:
        k_cumsum = k.sum(dim = -2) 
        # 沿着倒数第二个维度对张量 k 进行求和，得到 k_cumsum
        D_inv = 1. / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-8)
        # 使用 Einstein Summation Notation 计算张量 q 和 k_cumsum 的乘积，并对结果取倒数，避免除零错误

        context = torch.einsum('...nd,...ne->...de', k, v)
        # 使用 Einstein Summation Notation 计算张量 k 和 v 的乘积，并返回结果
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        # 使用 Einstein Summation Notation 计算张量 context、q 和 D_inv 的乘积，并返回结果
        return out
def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, qr_uniform_q = False, device = None):
    # 计算完整的块数
    nb_full_blocks = int(nb_rows / nb_columns)
    #print (nb_full_blocks)
    block_list = []

    for _ in range(nb_full_blocks):
        # 生成正交矩阵块并添加到列表中
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        block_list.append(q)
    # block_list[n] is a orthogonal matrix ... (model_dim * model_dim)
    #print (block_list[0].size(), torch.einsum('...nd,...nd->...n', block_list[0], torch.roll(block_list[0],1,1)))
    #print (nb_rows, nb_full_blocks, nb_columns)
    # 计算剩余的行数
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    #print (remaining_rows)
    if remaining_rows > 0:
        # 生成正交矩阵块并添加到列表中
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        #print (q[:remaining_rows].size())
        block_list.append(q[:remaining_rows])

    # 将所有块连接成最终的矩阵
    final_matrix = torch.cat(block_list)
    
    if scaling == 0:
        # 生成随机的乘数并计算其范数
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        # 生成指定值的乘数
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        # 抛出异常，指定的缩放值无效
        raise ValueError(f'Invalid scaling {scaling}')

    # 返回乘数矩阵与最终矩阵的乘积
    return torch.diag(multiplier) @ final_matrix

class FastAttention(nn.Module):
    # 初始化函数，设置注意力机制的参数
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, no_projection = False):
        super().__init__()
        # 如果未指定特征数量，则默认为维度头数乘以以 e 为底的对数
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        # 设置维度头数和特征数量
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        # 创建投影矩阵的函数
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling, qr_uniform_q = qr_uniform_q)
        # 生成投影矩阵并注册为缓冲区
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        # 设置是否使用广义注意力机制和核函数
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # 如果开启此选项，将不使用投影矩阵
        # 查询和键将按照原始高效注意力论文中的方式进行 softmax 处理
        self.no_projection = no_projection

        # 设置是否使用因果注意力机制
        self.causal = causal

    # 重新生成投影矩阵的函数，使用 torch.no_grad() 修饰
    @torch.no_grad()
    def redraw_projection_matrix(self):
        # 重新生成投影矩阵并复制到缓冲区中
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    # 前向传播函数，接收查询、键、值作为输入
    def forward(self, q, k, v):
        # 获取输入张量所在的设备
        device = q.device

        # 如果不使用投影矩阵
        if self.no_projection:
            # 对查询进行 softmax 处理
            q = q.softmax(dim = -1)
            # 如果开启因果注意力，对键进行指数处理，否则对键进行 softmax 处理
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)
        else:
            # 创建核函数，使用投影矩阵和设备信息
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            # 对查询和键应用核函数
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        # 如果不是因果注意力，使用线性注意力函数，否则使用因果线性函数
        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        # 如果值为空，则只返回注意力分布
        if v is None:
            out = attn_fn(q, k, None)
            return out
        # 否则返回注意力加权后的值
        else:
            out = attn_fn(q, k, v)
            return out
class SelfAttention(nn.Module):
    def __init__(self, dim, causal = False, heads = 8, dim_head = 64, local_heads = 0, local_window_size = 256, nb_features = None, feature_redraw_interval = 1000, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, dropout = 0., no_projection = False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        # 创建一个 FastAttention 对象
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, qr_uniform_q = qr_uniform_q, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        # 如果 local_heads 大于 0，则创建一个 LocalAttention 对象，否则为 None
        self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        # 创建一个线性层，用于将输入映射到查询向量
        self.to_q = nn.Linear(dim, inner_dim)
        # 创建一个线性层，用于将输入映射到键向量
        self.to_k = nn.Linear(dim, inner_dim)
        # 创建一个线性层，用于将输入映射到值向量
        self.to_v = nn.Linear(dim, inner_dim)
        # 创建一个线性层，用于将内部维度映射回原始维度
        self.to_out = nn.Linear(inner_dim, dim)
        # 创建一个 dropout 层
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        # 重新绘制投影矩阵
        self.fast_attention.redraw_projection_matrix()
        # 初始化 name_embedding 为零
        #torch.nn.init.zeros_(self.name_embedding)
        # 打印 name_embedding 的和
        #print (torch.sum(self.name_embedding))
    # 定义一个前向传播函数，接受输入 x，上下文 context，掩码 mask，上下文掩码 context_mask，名称 name，推断标志 inference，以及其他关键字参数
    def forward(self, x, context = None, mask = None, context_mask = None, name=None, inference=False, **kwargs):
        # 获取输入 x 的形状信息，并提取出其中的维度信息
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads
        
        # 检查是否存在上下文信息
        cross_attend = exists(context)

        # 如果未提供上下文信息，则使用输入 x 作为上下文
        context = default(context, x)
        # 如果未提供上下文掩码信息，并且不是跨上下文关注，则使用输入的掩码信息作为上下文掩码
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        
        # 将输入 x 转换为查询（q）、键（k）、值（v）的形式
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        # 将查询（q）、键（k）、值（v）重排为多头形式
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # 将查询（q）、键（k）、值（v）分割为全局部分和局部部分
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        # 初始化注意力输出列表
        attn_outs = []

        # 如果查询（q）不为空
        if not empty(q):
            # 如果存在上下文掩码
            if exists(context_mask):
                # 创建全局掩码
                global_mask = context_mask[:, None, :, None]
                # 对值（v）进行掩码处理
                v.masked_fill_(~global_mask, 0.)
            # 如果是跨上下文关注
            if cross_attend:
                pass
                # 执行跨上下文关注操作
                #out = self.fast_attention(q,self.name_embedding[name],None)
            else:
                # 执行快速注意力操作
                out = self.fast_attention(q, k, v)
            # 将注意力输出添加到列表中
            attn_outs.append(out)

        # 如果局部查询（lq）不为空
        if not empty(lq):
            # 断言不兼容跨上下文关注和局部关注
            assert not cross_attend, 'local attention is not compatible with cross attention'
            # 执行局部注意力操作
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            # 将局部注意力输出添加到列表中
            attn_outs.append(out)

        # 将所有注意力输出连接起来
        out = torch.cat(attn_outs, dim = 1)
        # 重排输出的形状
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 将输出传递给输出层
        out =  self.to_out(out)
        # 对输出进行 dropout 处理
        return self.dropout(out)
```