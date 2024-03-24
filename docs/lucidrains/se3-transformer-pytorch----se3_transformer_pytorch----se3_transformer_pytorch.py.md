# `.\lucidrains\se3-transformer-pytorch\se3_transformer_pytorch\se3_transformer_pytorch.py`

```
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 从 itertools 模块中导入 product 函数
from itertools import product
# 从 collections 模块中导入 namedtuple 类
from collections import namedtuple

# 导入 torch 库
import torch
# 从 torch.nn.functional 模块中导入 F 别名
import torch.nn.functional as F
# 从 torch 模块中导入 nn 和 einsum 函数
from torch import nn, einsum

# 导入自定义模块
from se3_transformer_pytorch.basis import get_basis
from se3_transformer_pytorch.utils import exists, default, uniq, map_values, batched_index_select, masked_mean, to_order, fourier_encode, cast_tuple, safe_cat, fast_split, rand_uniform, broadcat
from se3_transformer_pytorch.reversible import ReversibleSequence, SequentialSequence
from se3_transformer_pytorch.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

# 从 einops 模块中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 定义命名元组 FiberEl，包含 degrees 和 dim 两个字段
FiberEl = namedtuple('FiberEl', ['degrees', 'dim'])

# 定义 Fiber 类
class Fiber(nn.Module):
    def __init__(
        self,
        structure
    ):
        super().__init__()
        # 如果 structure 是字典，则转换为列表形式
        if isinstance(structure, dict):
            structure = [FiberEl(degree, dim) for degree, dim in structure.items()]
        self.structure = structure

    # 返回所有维度的列表
    @property
    def dims(self):
        return uniq(map(lambda t: t[1], self.structure))

    # 返回所有度数的生成器
    @property
    def degrees(self):
        return map(lambda t: t[0], self.structure)

    # 创建 Fiber 实例
    @staticmethod
    def create(num_degrees, dim):
        dim_tuple = dim if isinstance(dim, tuple) else ((dim,) * num_degrees)
        return Fiber([FiberEl(degree, dim) for degree, dim in zip(range(num_degrees), dim_tuple)])

    # 获取指定度数的元素
    def __getitem__(self, degree):
        return dict(self.structure)[degree]

    # 迭代器方法
    def __iter__(self):
        return iter(self.structure)

    # 定义乘法操作
    def __mul__(self, fiber):
        return product(self.structure, fiber.structure)

    # 定义与操作
    def __and__(self, fiber):
        out = []
        degrees_out = fiber.degrees
        for degree, dim in self:
            if degree in fiber.degrees:
                dim_out = fiber[degree]
                out.append((degree, dim, dim_out))
        return out

# 获取张量的设备和数据类型
def get_tensor_device_and_dtype(features):
    first_tensor = next(iter(features.items()))[1]
    return first_tensor.device, first_tensor.dtype

# 定义 ResidualSE3 类
class ResidualSE3(nn.Module):
    """ only support instance where both Fibers are identical """
    def forward(self, x, res):
        out = {}
        for degree, tensor in x.items():
            degree = str(degree)
            out[degree] = tensor
            if degree in res:
                out[degree] = out[degree] + res[degree]
        return out

# 定义 LinearSE3 类
class LinearSE3(nn.Module):
    def __init__(
        self,
        fiber_in,
        fiber_out
    ):
        super().__init__()
        self.weights = nn.ParameterDict()

        for (degree, dim_in, dim_out) in (fiber_in & fiber_out):
            key = str(degree)
            self.weights[key]  = nn.Parameter(torch.randn(dim_in, dim_out) / sqrt(dim_in))

    def forward(self, x):
        out = {}
        for degree, weight in self.weights.items():
            out[degree] = einsum('b n d m, d e -> b n e m', x[degree], weight)
        return out

# 定义 NormSE3 类
class NormSE3(nn.Module):
    """Norm-based SE(3)-equivariant nonlinearity.
    
    Nonlinearities are important in SE(3) equivariant GCNs. They are also quite 
    expensive to compute, so it is convenient for them to share resources with
    other layers, such as normalization. The general workflow is as follows:

    > for feature type in features:
    >    norm, phase <- feature
    >    output = fnc(norm) * phase
    
    where fnc: {R+}^m -> R^m is a learnable map from m norms to m scalars.
    """
    def __init__(
        self,
        fiber,
        nonlin = nn.GELU(),
        gated_scale = False,
        eps = 1e-12,
    # 初始化函数，设置初始参数
    def __init__(
        self,
        fiber,
        nonlin = nn.ReLU(),
        eps = 1e-12,
        gated_scale = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 将参数赋值给对象属性
        self.fiber = fiber
        self.nonlin = nonlin
        self.eps = eps

        # Norm mappings: 1 per feature type
        # 创建一个 ModuleDict 对象，用于存储每种特征类型的规范化映射
        self.transform = nn.ModuleDict()
        # 遍历 fiber 中的每个元素
        for degree, chan in fiber:
            # 为每种特征类型创建一个参数字典
            self.transform[str(degree)] = nn.ParameterDict({
                'scale': nn.Parameter(torch.ones(1, 1, chan)) if not gated_scale else None,
                'w_gate': nn.Parameter(rand_uniform((chan, chan), -1e-3, 1e-3)) if gated_scale else None
            })

    # 前向传播函数
    def forward(self, features):
        # 初始化输出字典
        output = {}
        # 遍历输入的特征字典
        for degree, t in features.items():
            # 计算规范化和归一化特征
            norm = t.norm(dim = -1, keepdim = True).clamp(min = self.eps)
            phase = t / norm

            # Transform on norms
            # 获取当前特征类型对应的参数
            parameters = self.transform[degree]
            gate_weights, scale = parameters['w_gate'], parameters['scale']

            # 重排特征
            transformed = rearrange(norm, '... () -> ...')

            # 如果缺少 scale 参数，则使用 gate_weights 进行计算
            if not exists(scale):
                scale = einsum('b n d, d e -> b n e', transformed, gate_weights)

            # 对特征进行非线性变换
            transformed = self.nonlin(transformed * scale)
            transformed = rearrange(transformed, '... -> ... ()')

            # 对规范化特征进行非线性变换
            output[degree] = (transformed * phase).view(*t.shape)

        # 返回输出字典
        return output
class ConvSE3(nn.Module):
    """定义一个张量场网络层
    
    ConvSE3代表一个SE(3)-等变卷积层。它相当于MLP中的线性层，CNN中的卷积层，或者GCN中的图卷积层。

    在每个节点上，激活被分成不同的“特征类型”，由SE(3)表示类型索引：非负整数0, 1, 2, ..
    """
    def __init__(
        self,
        fiber_in,
        fiber_out,
        self_interaction = True,
        pool = True,
        edge_dim = 0,
        fourier_encode_dist = False,
        num_fourier_features = 4,
        splits = 4
    ):
        super().__init__()
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction

        self.num_fourier_features = num_fourier_features
        self.fourier_encode_dist = fourier_encode_dist

        # radial function will assume a dimension of at minimum 1, for the relative distance - extra fourier features must be added to the edge dimension
        edge_dim += (0 if not fourier_encode_dist else (num_fourier_features * 2))

        # Neighbor -> center weights
        self.kernel_unary = nn.ModuleDict()

        self.splits = splits # for splitting the computation of kernel and basis, to reduce peak memory usage

        for (di, mi), (do, mo) in (self.fiber_in * self.fiber_out):
            self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo, edge_dim = edge_dim, splits = splits)

        self.pool = pool

        # Center -> center weights
        if self_interaction:
            assert self.pool, 'must pool edges if followed with self interaction'
            self.self_interact = LinearSE3(fiber_in, fiber_out)
            self.self_interact_sum = ResidualSE3()

    def forward(
        self,
        inp,
        edge_info,
        rel_dist = None,
        basis = None
        ):
            # 获取拆分信息
            splits = self.splits
            neighbor_indices, neighbor_masks, edges = edge_info
            # 重新排列相对距离的维度
            rel_dist = rearrange(rel_dist, 'b m n -> b m n ()')

            kernels = {}
            outputs = {}

            if self.fourier_encode_dist:
                # 对相对距离进行傅立叶编码
                rel_dist = fourier_encode(rel_dist[..., None], num_encodings = self.num_fourier_features)

            # 拆分基础

            basis_keys = basis.keys()
            split_basis_values = list(zip(*list(map(lambda t: fast_split(t, splits, dim = 1), basis.values())))
            split_basis = list(map(lambda v: dict(zip(basis_keys, v)), split_basis_values))

            # 遍历每种输入度类型到输出度类型的排列组合

            for degree_out in self.fiber_out.degrees:
                output = 0
                degree_out_key = str(degree_out)

                for degree_in, m_in in self.fiber_in:
                    etype = f'({degree_in},{degree_out})'

                    x = inp[str(degree_in)]

                    x = batched_index_select(x, neighbor_indices, dim = 1)
                    x = x.view(*x.shape[:3], to_order(degree_in) * m_in, 1)

                    kernel_fn = self.kernel_unary[etype]
                    edge_features = torch.cat((rel_dist, edges), dim = -1) if exists(edges) else rel_dist

                    output_chunk = None
                    split_x = fast_split(x, splits, dim = 1)
                    split_edge_features = fast_split(edge_features, splits, dim = 1)

                    # 沿着序列维度对输入、边缘和基础进行分块处理

                    for x_chunk, edge_features, basis in zip(split_x, split_edge_features, split_basis):
                        kernel = kernel_fn(edge_features, basis = basis)
                        chunk = einsum('... o i, ... i c -> ... o c', kernel, x_chunk)
                        output_chunk = safe_cat(output_chunk, chunk, dim = 1)

                    output = output + output_chunk

                if self.pool:
                    output = masked_mean(output, neighbor_masks, dim = 2) if exists(neighbor_masks) else output.mean(dim = 2)

                leading_shape = x.shape[:2] if self.pool else x.shape[:3]
                output = output.view(*leading_shape, -1, to_order(degree_out))

                outputs[degree_out_key] = output

            if self.self_interaction:
                self_interact_out = self.self_interact(inp)
                outputs = self.self_interact_sum(outputs, self_interact_out)

            return outputs
class RadialFunc(nn.Module):
    """定义一个神经网络参数化的径向函数。"""
    def __init__(
        self,
        num_freq,
        in_dim,
        out_dim,
        edge_dim = None,
        mid_dim = 128
    ):
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.edge_dim = default(edge_dim, 0)

        self.net = nn.Sequential(
            nn.Linear(self.edge_dim + 1, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, num_freq * in_dim * out_dim)
        )

    def forward(self, x):
        y = self.net(x)
        return rearrange(y, '... (o i f) -> ... o () i () f', i = self.in_dim, o = self.out_dim)

class PairwiseConv(nn.Module):
    """两种单一类型特征之间的SE(3)-等变卷积。"""
    def __init__(
        self,
        degree_in,
        nc_in,
        degree_out,
        nc_out,
        edge_dim = 0,
        splits = 4
    ):
        super().__init__()
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        self.num_freq = to_order(min(degree_in, degree_out))
        self.d_out = to_order(degree_out)
        self.edge_dim = edge_dim

        self.rp = RadialFunc(self.num_freq, nc_in, nc_out, edge_dim)

        self.splits = splits

    def forward(self, feat, basis):
        splits = self.splits
        R = self.rp(feat)
        B = basis[f'{self.degree_in},{self.degree_out}']

        out_shape = (*R.shape[:3], self.d_out * self.nc_out, -1)

        # torch.sum(R * B, dim = -1) is too memory intensive
        # needs to be chunked to reduce peak memory usage

        out = 0
        for i in range(R.shape[-1]):
            out += R[..., i] * B[..., i]

        out = rearrange(out, 'b n h s ... -> (b n h s) ...')

        # reshape and out
        return out.view(*out_shape)

# feed forwards

class FeedForwardSE3(nn.Module):
    def __init__(
        self,
        fiber,
        mult = 4
    ):
        super().__init__()
        self.fiber = fiber
        fiber_hidden = Fiber(list(map(lambda t: (t[0], t[1] * mult), fiber)))

        self.project_in  = LinearSE3(fiber, fiber_hidden)
        self.nonlin      = NormSE3(fiber_hidden)
        self.project_out = LinearSE3(fiber_hidden, fiber)

    def forward(self, features):
        outputs = self.project_in(features)
        outputs = self.nonlin(outputs)
        outputs = self.project_out(outputs)
        return outputs

class FeedForwardBlockSE3(nn.Module):
    def __init__(
        self,
        fiber,
        norm_gated_scale = False
    ):
        super().__init__()
        self.fiber = fiber
        self.prenorm = NormSE3(fiber, gated_scale = norm_gated_scale)
        self.feedforward = FeedForwardSE3(fiber)
        self.residual = ResidualSE3()

    def forward(self, features):
        res = features
        out = self.prenorm(features)
        out = self.feedforward(out)
        return self.residual(out, res)

# attention

class AttentionSE3(nn.Module):
    def __init__(
        self,
        fiber,
        dim_head = 64,
        heads = 8,
        attend_self = False,
        edge_dim = None,
        fourier_encode_dist = False,
        rel_dist_num_fourier_features = 4,
        use_null_kv = False,
        splits = 4,
        global_feats_dim = None,
        linear_proj_keys = False,
        tie_key_values = False
        ):
        # 调用父类的构造函数
        super().__init__()
        # 计算隐藏层维度
        hidden_dim = dim_head * heads
        # 创建隐藏层的 Fiber 对象
        hidden_fiber = Fiber(list(map(lambda t: (t[0], hidden_dim), fiber)))
        # 判断是否需要进行输出投影
        project_out = not (heads == 1 and len(fiber.dims) == 1 and dim_head == fiber.dims[0])

        # 设置缩放因子
        self.scale = dim_head ** -0.5
        self.heads = heads

        # 是否对特征进行线性投影以获得 keys
        self.linear_proj_keys = linear_proj_keys
        # 创建 LinearSE3 对象用于处理 queries
        self.to_q = LinearSE3(fiber, hidden_fiber)
        # 创建 ConvSE3 对象用于处理 values
        self.to_v = ConvSE3(fiber, hidden_fiber, edge_dim = edge_dim, pool = False, self_interaction = False, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)

        # 检查是否同时进行线性投影 keys 和共享 key / values
        assert not (linear_proj_keys and tie_key_values), 'you cannot do linear projection of keys and have shared key / values turned on at the same time'

        # 根据不同情况创建 keys 处理对象
        if linear_proj_keys:
            self.to_k = LinearSE3(fiber, hidden_fiber)
        elif not tie_key_values:
            self.to_k = ConvSE3(fiber, hidden_fiber, edge_dim = edge_dim, pool = False, self_interaction = False, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)
        else:
            self.to_k = None

        # 创建输出处理对象
        self.to_out = LinearSE3(hidden_fiber, fiber) if project_out else nn.Identity()

        # 是否使用空的 keys 和 values
        self.use_null_kv = use_null_kv
        if use_null_kv:
            self.null_keys = nn.ParameterDict()
            self.null_values = nn.ParameterDict()

            # 初始化空的 keys 和 values
            for degree in fiber.degrees:
                m = to_order(degree)
                degree_key = str(degree)
                self.null_keys[degree_key] = nn.Parameter(torch.zeros(heads, dim_head, m))
                self.null_values[degree_key] = nn.Parameter(torch.zeros(heads, dim_head, m))

        # 是否自我关注
        self.attend_self = attend_self
        if attend_self:
            # 创建自我关注的 keys 处理对象
            self.to_self_k = LinearSE3(fiber, hidden_fiber)
            # 创建自我关注的 values 处理对象
            self.to_self_v = LinearSE3(fiber, hidden_fiber)

        # 是否接受全局特征
        self.accept_global_feats = exists(global_feats_dim)
        if self.accept_global_feats:
            # 创建全局特征的 keys 处理对象
            global_input_fiber = Fiber.create(1, global_feats_dim)
            global_output_fiber = Fiber.create(1, hidden_fiber[0])
            self.to_global_k = LinearSE3(global_input_fiber, global_output_fiber)
            # 创建全局特征的 values 处理对象
            self.to_global_v = LinearSE3(global_input_fiber, global_output_fiber)
    # 定义前向传播函数，接收特征、边信息、相对距离、基础信息、全局特征、位置嵌入和掩码作为输入
    def forward(self, features, edge_info, rel_dist, basis, global_feats = None, pos_emb = None, mask = None):
        # 获取头数和是否自我关注的标志
        h, attend_self = self.heads, self.attend_self
        # 获取设备和数据类型
        device, dtype = get_tensor_device_and_dtype(features)
        # 解包边信息
        neighbor_indices, neighbor_mask, edges = edge_info

        # 如果邻居掩码存在，则重排维度
        if exists(neighbor_mask):
            neighbor_mask = rearrange(neighbor_mask, 'b i j -> b () i j')

        # 将特征转换为查询、值和键
        queries = self.to_q(features)
        values  = self.to_v(features, edge_info, rel_dist, basis)

        # 如果使用线性投影的键，则将键映射到邻居索引
        if self.linear_proj_keys:
            keys = self.to_k(features)
            keys = map_values(lambda val: batched_index_select(val, neighbor_indices, dim = 1), keys)
        # 如果没有定义键转换函数，则将键设置为值
        elif not exists(self.to_k):
            keys = values
        else:
            keys = self.to_k(features, edge_info, rel_dist, basis)

        # 如果允许自我关注，则获取自我键和自我值
        if attend_self:
            self_keys, self_values = self.to_self_k(features), self.to_self_v(features)

        # 如果全局特征存在，则获取全局键和全局值
        if exists(global_feats):
            global_keys, global_values = self.to_global_k(global_feats), self.to_global_v(global_feats)

        # 初始化输出字典
        outputs = {}
        # 遍历特征的度
        for degree in features.keys():
            # 获取当前度的查询、键和值
            q, k, v = map(lambda t: t[degree], (queries, keys, values))

            # 重排查询、键和值的维度
            q = rearrange(q, 'b i (h d) m -> b h i d m', h = h)
            k, v = map(lambda t: rearrange(t, 'b i j (h d) m -> b h i j d m', h = h), (k, v))

            # 如果允许自我关注，则处理自我键和自我值
            if attend_self:
                self_k, self_v = map(lambda t: t[degree], (self_keys, self_values))
                self_k, self_v = map(lambda t: rearrange(t, 'b n (h d) m -> b h n () d m', h = h), (self_k, self_v))
                k = torch.cat((self_k, k), dim = 3)
                v = torch.cat((self_v, v), dim = 3)

            # 如果位置嵌入存在且度为'0'，则应用旋转位置嵌入
            if exists(pos_emb) and degree == '0':
                query_pos_emb, key_pos_emb = pos_emb
                query_pos_emb = rearrange(query_pos_emb, 'b i d -> b () i d ()')
                key_pos_emb = rearrange(key_pos_emb, 'b i j d -> b () i j d ()')
                q = apply_rotary_pos_emb(q, query_pos_emb)
                k = apply_rotary_pos_emb(k, key_pos_emb)
                v = apply_rotary_pos_emb(v, key_pos_emb)

            # 如果使用空键值对，则处理空键和空值
            if self.use_null_kv:
                null_k, null_v = map(lambda t: t[degree], (self.null_keys, self.null_values))
                null_k, null_v = map(lambda t: repeat(t, 'h d m -> b h i () d m', b = q.shape[0], i = q.shape[2]), (null_k, null_v))
                k = torch.cat((null_k, k), dim = 3)
                v = torch.cat((null_v, v), dim = 3)

            # 如果全局特征存在且度为'0'，则处理全局键和全局值
            if exists(global_feats) and degree == '0':
                global_k, global_v = map(lambda t: t[degree], (global_keys, global_values))
                global_k, global_v = map(lambda t: repeat(t, 'b j (h d) m -> b h i j d m', h = h, i = k.shape[2]), (global_k, global_v))
                k = torch.cat((global_k, k), dim = 3)
                v = torch.cat((global_v, v), dim = 3)

            # 计算注意力权重
            sim = einsum('b h i d m, b h i j d m -> b h i j', q, k) * self.scale

            # 如果邻居掩码存在，则进行掩码处理
            if exists(neighbor_mask):
                num_left_pad = sim.shape[-1] - neighbor_mask.shape[-1]
                mask = F.pad(neighbor_mask, (num_left_pad, 0), value = True)
                sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

            # 计算注意力输出
            attn = sim.softmax(dim = -1)
            out = einsum('b h i j, b h i j d m -> b h i d m', attn, v)
            outputs[degree] = rearrange(out, 'b h n d m -> b n (h d) m')

        # 返回输出结果
        return self.to_out(outputs)
# 定义一个带有一个键/值投影的注意力机制类，该投影在所有查询头之间共享
class OneHeadedKVAttentionSE3(nn.Module):
    def __init__(
        self,
        fiber,
        dim_head = 64,
        heads = 8,
        attend_self = False,
        edge_dim = None,
        fourier_encode_dist = False,
        rel_dist_num_fourier_features = 4,
        use_null_kv = False,
        splits = 4,
        global_feats_dim = None,
        linear_proj_keys = False,
        tie_key_values = False
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        hidden_fiber = Fiber(list(map(lambda t: (t[0], hidden_dim), fiber)))
        kv_hidden_fiber = Fiber(list(map(lambda t: (t[0], dim_head), fiber)))
        project_out = not (heads == 1 and len(fiber.dims) == 1 and dim_head == fiber.dims[0])

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.linear_proj_keys = linear_proj_keys # 是否对键进行线性投影，而不是与基卷积

        # 创建查询线性层
        self.to_q = LinearSE3(fiber, hidden_fiber)
        # 创建值卷积层
        self.to_v = ConvSE3(fiber, kv_hidden_fiber, edge_dim = edge_dim, pool = False, self_interaction = False, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)

        assert not (linear_proj_keys and tie_key_values), 'you cannot do linear projection of keys and have shared key / values turned on at the same time'

        if linear_proj_keys:
            # 如果进行线性投影，则创建键的线性层
            self.to_k = LinearSE3(fiber, kv_hidden_fiber)
        elif not tie_key_values:
            # 如果不共享键/值，则创建键的卷积层
            self.to_k = ConvSE3(fiber, kv_hidden_fiber, edge_dim = edge_dim, pool = False, self_interaction = False, fourier_encode_dist = fourier_encode_dist, num_fourier_features = rel_dist_num_fourier_features, splits = splits)
        else:
            self.to_k = None

        # 创建输出线性层
        self.to_out = LinearSE3(hidden_fiber, fiber) if project_out else nn.Identity()

        self.use_null_kv = use_null_kv
        if use_null_kv:
            # 如果使用空键/值，则创建空键和值的参数字典
            self.null_keys = nn.ParameterDict()
            self.null_values = nn.ParameterDict()

            for degree in fiber.degrees:
                m = to_order(degree)
                degree_key = str(degree)
                self.null_keys[degree_key] = nn.Parameter(torch.zeros(dim_head, m))
                self.null_values[degree_key] = nn.Parameter(torch.zeros(dim_head, m))

        self.attend_self = attend_self
        if attend_self:
            # 如果自我关注，则创建自我键和值的线性层
            self.to_self_k = LinearSE3(fiber, kv_hidden_fiber)
            self.to_self_v = LinearSE3(fiber, kv_hidden_fiber)

        self.accept_global_feats = exists(global_feats_dim)
        if self.accept_global_feats:
            # 如果接受全局特征，则创建全局键和值的线性层
            global_input_fiber = Fiber.create(1, global_feats_dim)
            global_output_fiber = Fiber.create(1, kv_hidden_fiber[0])
            self.to_global_k = LinearSE3(global_input_fiber, global_output_fiber)
            self.to_global_v = LinearSE3(global_input_fiber, global_output_fiber)
    # 定义前向传播函数，接收特征、边信息、相对距离、基础信息、全局特征、位置嵌入和掩码作为输入
    def forward(self, features, edge_info, rel_dist, basis, global_feats = None, pos_emb = None, mask = None):
        # 获取头数和是否自我关注的标志
        h, attend_self = self.heads, self.attend_self
        # 获取设备和数据类型
        device, dtype = get_tensor_device_and_dtype(features)
        # 解包边信息
        neighbor_indices, neighbor_mask, edges = edge_info

        # 如果存在邻居掩码，则重排维度
        if exists(neighbor_mask):
            neighbor_mask = rearrange(neighbor_mask, 'b i j -> b () i j')

        # 将特征转换为查询、值和键
        queries = self.to_q(features)
        values  = self.to_v(features, edge_info, rel_dist, basis)

        # 如果使用线性投影的键，则将键映射到相应的位置
        if self.linear_proj_keys:
            keys = self.to_k(features)
            keys = map_values(lambda val: batched_index_select(val, neighbor_indices, dim = 1), keys)
        # 如果没有定义键转换函数，则将键设置为值
        elif not exists(self.to_k):
            keys = values
        else:
            keys = self.to_k(features, edge_info, rel_dist, basis)

        # 如果允许自我关注，则获取自我关注的键和值
        if attend_self:
            self_keys, self_values = self.to_self_k(features), self.to_self_v(features)

        # 如果存在全局特征，则获取全局键和值
        if exists(global_feats):
            global_keys, global_values = self.to_global_k(global_feats), self.to_global_v(global_feats)

        # 初始化输出字典
        outputs = {}
        # 遍历特征的度
        for degree in features.keys():
            # 获取当前度的查询、键和值
            q, k, v = map(lambda t: t[degree], (queries, keys, values))

            # 重排查询的维度
            q = rearrange(q, 'b i (h d) m -> b h i d m', h = h)

            # 如果允许自我关注，则处理自我关注的键和值
            if attend_self:
                self_k, self_v = map(lambda t: t[degree], (self_keys, self_values))
                self_k, self_v = map(lambda t: rearrange(t, 'b n d m -> b n () d m'), (self_k, self_v))
                k = torch.cat((self_k, k), dim = 2)
                v = torch.cat((self_v, v), dim = 2)

            # 如果存在位置嵌入并且度为 '0'，则应用旋转位置嵌入
            if exists(pos_emb) and degree == '0':
                query_pos_emb, key_pos_emb = pos_emb
                query_pos_emb = rearrange(query_pos_emb, 'b i d -> b () i d ()')
                key_pos_emb = rearrange(key_pos_emb, 'b i j d -> b i j d ()')
                q = apply_rotary_pos_emb(q, query_pos_emb)
                k = apply_rotary_pos_emb(k, key_pos_emb)
                v = apply_rotary_pos_emb(v, key_pos_emb)

            # 如果使用空键值对，则将空键值对与当前键值对拼接
            if self.use_null_kv:
                null_k, null_v = map(lambda t: t[degree], (self.null_keys, self.null_values))
                null_k, null_v = map(lambda t: repeat(t, 'd m -> b i () d m', b = q.shape[0], i = q.shape[2]), (null_k, null_v))
                k = torch.cat((null_k, k), dim = 2)
                v = torch.cat((null_v, v), dim = 2)

            # 如果存在全局特征并且度为 '0'，则将全局键值对与当前键值对拼接
            if exists(global_feats) and degree == '0':
                global_k, global_v = map(lambda t: t[degree], (global_keys, global_values))
                global_k, global_v = map(lambda t: repeat(t, 'b j d m -> b i j d m', i = k.shape[1]), (global_k, global_v))
                k = torch.cat((global_k, k), dim = 2)
                v = torch.cat((global_v, v), dim = 2)

            # 计算注意力权重
            sim = einsum('b h i d m, b i j d m -> b h i j', q, k) * self.scale

            # 如果存在邻居掩码，则进行掩码操作
            if exists(neighbor_mask):
                num_left_pad = sim.shape[-1] - neighbor_mask.shape[-1]
                mask = F.pad(neighbor_mask, (num_left_pad, 0), value = True)
                sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

            # 计算注意力分布并进行加权求和
            attn = sim.softmax(dim = -1)
            out = einsum('b h i j, b i j d m -> b h i d m', attn, v)
            outputs[degree] = rearrange(out, 'b h n d m -> b n (h d) m')

        # 将输出转换为最终输出
        return self.to_out(outputs)
# 定义一个注意力块类，继承自 nn.Module
class AttentionBlockSE3(nn.Module):
    def __init__(
        self,
        fiber,
        dim_head = 24,
        heads = 8,
        attend_self = False,
        edge_dim = None,
        use_null_kv = False,
        fourier_encode_dist = False,
        rel_dist_num_fourier_features = 4,
        splits = 4,
        global_feats_dim = False,
        linear_proj_keys = False,
        tie_key_values = False,
        attention_klass = AttentionSE3,
        norm_gated_scale = False
    ):
        super().__init__()
        # 初始化注意力机制
        self.attn = attention_klass(fiber, heads = heads, dim_head = dim_head, attend_self = attend_self, edge_dim = edge_dim, use_null_kv = use_null_kv, rel_dist_num_fourier_features = rel_dist_num_fourier_features, fourier_encode_dist =fourier_encode_dist, splits = splits, global_feats_dim = global_feats_dim, linear_proj_keys = linear_proj_keys, tie_key_values = tie_key_values)
        # 初始化预处理层
        self.prenorm = NormSE3(fiber, gated_scale = norm_gated_scale)
        # 初始化残差连接
        self.residual = ResidualSE3()

    def forward(self, features, edge_info, rel_dist, basis, global_feats = None, pos_emb = None, mask = None):
        res = features
        # 对输入特征进行预处理
        outputs = self.prenorm(features)
        # 使用注意力机制处理特征
        outputs = self.attn(outputs, edge_info, rel_dist, basis, global_feats, pos_emb, mask)
        # 返回残差连接结果
        return self.residual(outputs, res)

# 定义 Swish_ 类
class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

# 如果 nn 模块中有 SiLU 函数，则使用 nn.SiLU，否则使用自定义的 Swish_ 类
SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_

# 定义 HtypesNorm 类
class HtypesNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8, scale_init = 1e-2, bias_init = 1e-2):
        super().__init__()
        self.eps = eps
        # 初始化缩放参数和偏置参数
        scale = torch.empty(1, 1, 1, dim, 1).fill_(scale_init)
        bias = torch.empty(1, 1, 1, dim, 1).fill_(bias_init)
        self.scale = nn.Parameter(scale)
        self.bias = nn.Parameter(bias)

    def forward(self, coors):
        # 计算输入张量的范数
        norm = coors.norm(dim = -1, keepdim = True)
        # 对输入张量进行归一化处理
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * (norm * self.scale + self.bias)

# 定义 EGNN 类
class EGNN(nn.Module):
    def __init__(
        self,
        fiber,
        hidden_dim = 32,
        edge_dim = 0,
        init_eps = 1e-3,
        coor_weights_clamp_value = None
    ):
        super().__init__()
        self.fiber = fiber
        node_dim = fiber[0]

        htypes = list(filter(lambda t: t.degrees != 0, fiber))
        num_htypes = len(htypes)
        htype_dims = sum([fiberel.dim for fiberel in htypes])

        edge_input_dim = node_dim * 2 + htype_dims + edge_dim + 1

        # 初始化节点归一化层
        self.node_norm = nn.LayerNorm(node_dim)

        # 初始化边 MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            SiLU(),
            nn.Linear(edge_input_dim * 2, hidden_dim),
            SiLU()
        )

        self.htype_norms = nn.ModuleDict({})
        self.htype_gating = nn.ModuleDict({})

        for degree, dim in fiber:
            if degree == 0:
                continue
            # 初始化 HtypesNorm 和线性层
            self.htype_norms[str(degree)] = HtypesNorm(dim)
            self.htype_gating[str(degree)] = nn.Linear(node_dim, dim)

        # 初始化 Htypes MLP
        self.htypes_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            SiLU(),
            nn.Linear(hidden_dim * 4, htype_dims)
        )

        # 初始化节点 MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim * 2),
            SiLU(),
            nn.Linear(node_dim * 2, node_dim)
        )

        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # 初始化线性层的权重
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(
        self,
        features,
        edge_info,
        rel_dist,
        mask = None,
        **kwargs
        ):
            # 解包边信息
            neighbor_indices, neighbor_masks, edges = edge_info

            # 使用邻居掩码
            mask = neighbor_masks

            # 类型 0 特征

            # 获取节点特征
            nodes = features['0']
            # 重新排列节点特征
            nodes = rearrange(nodes, '... () -> ...')

            # 更高级别类型（htype）

            # 过滤出非 '0' 类型的特征
            htypes = list(filter(lambda t: t[0] != '0', features.items()))
            # 获取每个类型的度数
            htype_degrees = list(map(lambda t: t[0], htypes))
            # 获取每个类型的维度
            htype_dims = list(map(lambda t: t[1].shape[-2], htypes))

            # 准备更高级别类型

            rel_htypes = []
            rel_htypes_dists = []

            for degree, htype in htypes:
                # 计算相对类型
                rel_htype = rearrange(htype, 'b i d m -> b i () d m') - rearrange(htype, 'b j d m -> b () j d m')
                rel_htype_dist = rel_htype.norm(dim = -1)

                rel_htypes.append(rel_htype)
                rel_htypes_dists.append(rel_htype_dist)

            # 为边 MLP 准备边

            nodes_i = rearrange(nodes, 'b i d -> b i () d')
            nodes_j = batched_index_select(nodes, neighbor_indices, dim = 1)
            neighbor_higher_type_dists = map(lambda t: batched_index_select(t, neighbor_indices, dim = 2), rel_htypes_dists)
            coor_rel_dist = rearrange(rel_dist, 'b i j -> b i j ()')

            edge_mlp_inputs = broadcat((nodes_i, nodes_j, *neighbor_higher_type_dists, coor_rel_dist), dim = -1)

            if exists(edges):
                edge_mlp_inputs = torch.cat((edge_mlp_inputs, edges), dim = -1)

            # 获取中间表示

            m_ij = self.edge_mlp(edge_mlp_inputs)

            # 转换为坐标

            htype_weights = self.htypes_mlp(m_ij)

            if exists(self.coor_weights_clamp_value):
                clamp_value = self.coor_weights_clamp_value
                htype_weights.clamp_(min = -clamp_value, max = clamp_value)

            split_htype_weights = htype_weights.split(htype_dims, dim = -1)

            htype_updates = []

            if exists(mask):
                htype_mask = rearrange(mask, 'b i j -> b i j ()')
                htype_weights = htype_weights.masked_fill(~htype_mask, 0.)

            for degree, rel_htype, htype_weight in zip(htype_degrees, rel_htypes, split_htype_weights):
                normed_rel_htype = self.htype_norms[str(degree)](rel_htype)
                normed_rel_htype = batched_index_select(normed_rel_htype, neighbor_indices, dim = 2)

                htype_update = einsum('b i j d m, b i j d -> b i d m', normed_rel_htype, htype_weight)
                htype_updates.append(htype_update)

            # 转换为节点

            if exists(mask):
                m_ij_mask = rearrange(mask, '... -> ... ()')
                m_ij = m_ij.masked_fill(~m_ij_mask, 0.)

            m_i = m_ij.sum(dim = -2)

            normed_nodes = self.node_norm(nodes)
            node_mlp_input = torch.cat((normed_nodes, m_i), dim = -1)
            node_out = self.node_mlp(node_mlp_input) + nodes

            # 更新节点

            features['0'] = rearrange(node_out, '... -> ... ()')

            # 更新更高级别类型

            update_htype_dicts = dict(zip(htype_degrees, htype_updates))

            for degree, update_htype in update_htype_dicts.items():
                features[degree] = features[degree] + update_htype

            for degree in htype_degrees:
                gating = self.htype_gating[str(degree)](node_out).sigmoid()
                features[degree] = features[degree] * rearrange(gating, '... -> ... ()')

            return features
# 定义一个 EGnnNetwork 类，继承自 nn.Module 类
class EGnnNetwork(nn.Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        fiber,
        depth,
        edge_dim = 0,
        hidden_dim = 32,
        coor_weights_clamp_value = None,
        feedforward = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 将参数赋值给对象属性
        self.fiber = fiber
        self.layers = nn.ModuleList([])
        # 循环创建指定数量的 EGNN 和 FeedForwardBlockSE3 对象，并添加到 layers 中
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                EGNN(fiber = fiber, edge_dim = edge_dim, hidden_dim = hidden_dim, coor_weights_clamp_value = coor_weights_clamp_value),
                FeedForwardBlockSE3(fiber) if feedforward else None
            ]))

    # 前向传播函数，接收多个参数
    def forward(
        self,
        features,
        edge_info,
        rel_dist,
        basis,
        global_feats = None,
        pos_emb = None,
        mask = None,
        **kwargs
    ):
        # 解包 edge_info 参数
        neighbor_indices, neighbor_masks, edges = edge_info
        # 获取设备信息
        device = neighbor_indices.device

        # 修改邻居信息以包含自身（因为 SE3 变换器依赖于去除对自身的注意力，但这不适用于 EGNN）

        # 创建包含自身索引的张量
        self_indices = torch.arange(neighbor_indices.shape[1], device = device)
        self_indices = rearrange(self_indices, 'i -> () i ()')
        neighbor_indices = broadcat((self_indices, neighbor_indices), dim = -1)

        # 对邻居掩码进行填充
        neighbor_masks = F.pad(neighbor_masks, (1, 0), value = True)
        rel_dist = F.pad(rel_dist, (1, 0), value = 0.)

        # 如果存在边信息，则对边信息进行填充
        if exists(edges):
            edges = F.pad(edges, (0, 0, 1, 0), value = 0.)  # 暂时将令牌到自身的边设置为 0

        edge_info = (neighbor_indices, neighbor_masks, edges)

        # 遍历每一层
        for egnn, ff in self.layers:
            # 调用 EGNN 对象进行特征变换
            features = egnn(
                features,
                edge_info = edge_info,
                rel_dist = rel_dist,
                basis = basis,
                global_feats = global_feats,
                pos_emb = pos_emb,
                mask = mask,
                **kwargs
            )

            # 如果存在 FeedForwardBlockSE3 对象，则调用进行特征变换
            if exists(ff):
                features = ff(features)

        return features

# 主类
class SE3Transformer(nn.Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 24,
        depth = 2,
        input_degrees = 1,
        num_degrees = None,
        output_degrees = 1,
        valid_radius = 1e5,
        reduce_dim_out = False,
        num_tokens = None,
        num_positions = None,
        num_edge_tokens = None,
        edge_dim = None,
        reversible = False,
        attend_self = True,
        use_null_kv = False,
        differentiable_coors = False,
        fourier_encode_dist = False,
        rel_dist_num_fourier_features = 4,
        num_neighbors = float('inf'),
        attend_sparse_neighbors = False,
        num_adj_degrees = None,
        adj_dim = 0,
        max_sparse_neighbors = float('inf'),
        dim_in = None,
        dim_out = None,
        norm_out = False,
        num_conv_layers = 0,
        causal = False,
        splits = 4,
        global_feats_dim = None,
        linear_proj_keys = False,
        one_headed_key_values = False,
        tie_key_values = False,
        rotary_position = False,
        rotary_rel_dist = False,
        norm_gated_scale = False,
        use_egnn = False,
        egnn_hidden_dim = 32,
        egnn_weights_clamp_value = None,
        egnn_feedforward = False,
        hidden_fiber_dict = None,
        out_fiber_dict = None
    # 前向传播函数，接收多个参数
    def forward(
        self,
        feats,
        coors,
        mask = None,
        adj_mat = None,
        edges = None,
        return_type = None,
        return_pooled = False,
        neighbor_mask = None,
        global_feats = None
```