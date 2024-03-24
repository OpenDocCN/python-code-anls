# `.\lucidrains\equiformer-pytorch\equiformer_pytorch\equiformer_pytorch.py`

```
from math import sqrt
from functools import partial
from itertools import product
from collections import namedtuple

from beartype.typing import Optional, Union, Tuple, Dict
from beartype import beartype

import torch
from torch import nn, is_tensor, Tensor
import torch.nn.functional as F

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from opt_einsum import contract as opt_einsum

from equiformer_pytorch.basis import (
    get_basis,
    get_D_to_from_z_axis
)

from equiformer_pytorch.reversible import (
    SequentialSequence,
    ReversibleSequence
)

from equiformer_pytorch.utils import (
    exists,
    default,
    masked_mean,
    to_order,
    cast_tuple,
    safe_cat,
    fast_split,
    slice_for_centering_y_to_x,
    pad_for_centering_y_to_x
)

from einx import get_at

from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

# constants

# 定义一个命名元组，用于返回多个类型
Return = namedtuple('Return', ['type0', 'type1'])

# 定义一个命名元组，用于存储边的信息
EdgeInfo = namedtuple('EdgeInfo', ['neighbor_indices', 'neighbor_mask', 'edges'])

# helpers

# 定义一个函数，将一个张量打包成指定模式
def pack_one(t, pattern):
    return pack([t], pattern)

# 定义一个函数，将一个打包的张量解包成指定模式
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# fiber functions

# 定义一个函数，计算两个fiber的笛卡尔积
@beartype
def fiber_product(
    fiber_in: Tuple[int, ...],
    fiber_out: Tuple[int, ...]
):
    fiber_in, fiber_out = tuple(map(lambda t: [(degree, dim) for degree, dim in enumerate(t)], (fiber_in, fiber_out)))
    return product(fiber_in, fiber_out)

# 定义一个函数，计算两个fiber的交集
@beartype
def fiber_and(
    fiber_in: Tuple[int, ...],
    fiber_out: Tuple[int, ...]
):
    fiber_in = [(degree, dim) for degree, dim in enumerate(fiber_in)]
    fiber_out_degrees = set(range(len(fiber_out))

    out = []
    for degree, dim in fiber_in:
        if degree not in fiber_out_degrees:
            continue

        dim_out = fiber_out[degree]
        out.append((degree, dim, dim_out))

    return out

# helper functions

# 将一个数字分成指定组数的函数
def split_num_into_groups(num, groups):
    num_per_group = (num + groups - 1) // groups
    remainder = num % groups

    if remainder == 0:
        return (num_per_group,) * groups

    return (*((num_per_group,) * remainder), *((((num_per_group - 1),) * (groups - remainder))))

# 获取张量的设备和数据类型函数
def get_tensor_device_and_dtype(features):
    _, first_tensor = next(iter(features.items()))
    return first_tensor.device, first_tensor.dtype

# 计算残差的函数
def residual_fn(x, residual):
    out = {}

    for degree, tensor in x.items():
        out[degree] = tensor

        if degree not in residual:
            continue

        if not any(t.requires_grad for t in (out[degree], residual[degree])):
            out[degree] += residual[degree]
        else:
            out[degree] = out[degree] + residual[degree]

    return out

# 在元组中设置指定索引的值函数
def tuple_set_at_index(tup, index, value):
    l = list(tup)
    l[index] = value
    return tuple(l)

# 获取特征形状的函数
def feature_shapes(feature):
    return tuple(v.shape for v in feature.values())

# 获取特征fiber的函数
def feature_fiber(feature):
    return tuple(v.shape[-2] for v in feature.values())

# 计算两个张量之间的距离函数
def cdist(a, b, dim = -1, eps = 1e-5):
    a = a.expand_as(b)
    a, _ = pack_one(a, '* c')
    b, ps = pack_one(b, '* c')

    dist = F.pairwise_distance(a, b, p = 2)
    dist = unpack_one(dist, ps, '*')
    return dist

# classes

# 定义一个带残差的模块类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        y = self.fn(x, **kwargs)
        if not y.requires_grad and not x.requires_grad:
            return x.add_(y)
        return x + y

# 定义一个LayerNorm类
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 定义一个线性层类
class Linear(nn.Module):
    @beartype
    def __init__(
        self,
        fiber_in: Tuple[int, ...],
        fiber_out: Tuple[int, ...
    ):  
        # 调用父类的构造函数
        super().__init__()
        # 初始化权重列表和度数列表
        self.weights = nn.ParameterList([])
        self.degrees = []

        # 遍历输入和输出的纤维组合
        for (degree, dim_in, dim_out) in fiber_and(fiber_in, fiber_out):
            # 将随机初始化的权重添加到权重列表中
            self.weights.append(nn.Parameter(torch.randn(dim_in, dim_out) / sqrt(dim_in)))
            # 将度数添加到度数列表中
            self.degrees.append(degree)

    def init_zero_(self):
        # 将所有权重初始化为零
        for weight in self.weights:
            weight.data.zero_()

    def forward(self, x):
        # 初始化输出字典
        out = {}

        # 遍历度数和权重，进行张量乘法操作
        for degree, weight in zip(self.degrees, self.weights):
            out[degree] = einsum(x[degree], weight, '... d m, d e -> ... e m')

        return out
class Norm(nn.Module):
    @beartype
    def __init__(
        self,
        fiber: Tuple[int, ...],
        eps = 1e-12,
    ):
        """
        deviates from the paper slightly, will use rmsnorm throughout (no mean centering or bias, even for type0 fatures)
        this has been proven at scale for a number of models, including T5 and alphacode
        """

        super().__init__()
        # 设置 eps 参数
        self.eps = eps
        # 初始化 transforms 为一个空的 nn.ParameterList
        self.transforms = nn.ParameterList([])

        # 遍历 fiber 中的每个维度
        for degree, dim in enumerate(fiber):
            # 将每个维度的参数初始化为 1，并添加到 transforms 中
            self.transforms.append(nn.Parameter(torch.ones(dim, 1)))

    def forward(self, features):
        # 初始化输出字典
        output = {}

        # 遍历 transforms 和 features 中的每个元素
        for scale, (degree, t) in zip(self.transforms, features.items()):
            # 获取输入张量的维度
            dim = t.shape[-2]

            # 计算 L2 范数
            l2normed = t.norm(dim = -1, keepdim = True)
            # 计算 RMS 范数
            rms = l2normed.norm(dim = -2, keepdim = True) * (dim ** -0.5)

            # 将处理后的张量存入输出字典
            output[degree] = t / rms.clamp(min = self.eps) * scale

        return output

class Gate(nn.Module):
    @beartype
    def __init__(
        self,
        fiber: Tuple[int, ...]
    ):
        super().__init__()

        # 获取 type0_dim 和 dim_gate
        type0_dim = fiber[0]
        dim_gate = sum(fiber[1:])

        # 确保 type0_dim 大于 dim_gate
        assert type0_dim > dim_gate, 'sum of channels from rest of the degrees must be less than the channels in type 0, as they would be used up for gating and subtracted out'

        # 初始化 Gate 类的属性
        self.fiber = fiber
        self.num_degrees = len(fiber)
        self.type0_dim_split = [*fiber[1:], type0_dim - dim_gate]

    def forward(self, x):
        # 初始化输出字典
        output = {}

        # 获取 type0_tensor
        type0_tensor = x[0]
        # 将 type0_tensor 拆分为 gates 和 type0_tensor
        *gates, type0_tensor = type0_tensor.split(self.type0_dim_split, dim = -2)

        # 对 type 0 使用 silu 激活函数
        output = {0: F.silu(type0_tensor)}

        # 对高阶类型使用 sigmoid gate
        for degree, gate in zip(range(1, self.num_degrees), gates):
            output[degree] = x[degree] * gate.sigmoid()

        return output

class DTP(nn.Module):
    """ 'Tensor Product' - in the equivariant sense """

    @beartype
    def __init__(
        self,
        fiber_in: Tuple[int, ...],
        fiber_out: Tuple[int, ...],
        self_interaction = True,
        project_xi_xj = True,   # whether to project xi and xj and then sum, as in paper
        project_out = True,     # whether to do a project out after the "tensor product"
        pool = True,
        edge_dim = 0,
        radial_hidden_dim = 16
    ):
        super().__init__()
        # 初始化 DTP 类的属性
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction
        self.pool = pool

        self.project_xi_xj = project_xi_xj
        if project_xi_xj:
            # 初始化 Linear 层
            self.to_xi = Linear(fiber_in, fiber_in)
            self.to_xj = Linear(fiber_in, fiber_in)

        self.kernel_unary = nn.ModuleDict()

        # 遍历输出 fiber 中的每个维度和输入 fiber 中的每个维度
        for degree_out, dim_out in enumerate(self.fiber_out):
            num_degrees_in = len(self.fiber_in)
            # 将输出维度拆分为输入维度的组合
            split_dim_out = split_num_into_groups(dim_out, num_degrees_in)

            # 遍历每个输入维度和输出维度的组合
            for degree_in, (dim_in, dim_out_from_degree_in) in enumerate(zip(self.fiber_in, split_dim_out)):
                degree_min = min(degree_out, degree_in)

                # 初始化 Radial 层
                self.kernel_unary[f'({degree_in},{degree_out})'] = Radial(degree_in, dim_in, degree_out, dim_out_from_degree_in, radial_hidden_dim = radial_hidden_dim, edge_dim = edge_dim)

        # 是否进行单个 token 的自交互
        if self_interaction:
            self.self_interact = Linear(fiber_in, fiber_out)

        self.project_out = project_out
        if project_out:
            # 初始化 Linear 层
            self.to_out = Linear(fiber_out, fiber_out)

    @beartype
    def forward(
        self,
        inp,
        basis,
        D,
        edge_info: EdgeInfo,
        rel_dist = None,
        ):
            # 解包边信息
            neighbor_indices, neighbor_masks, edges = edge_info

            # 初始化变量
            kernels = {}
            outputs = {}

            # neighbors

            # 如果需要将输入投影到 xi 和 xj
            if self.project_xi_xj:
                source, target = self.to_xi(inp), self.to_xj(inp)
            else:
                source, target = inp, inp

            # 遍历输入度类型到输出度类型的每种排列
            for degree_out, _ in enumerate(self.fiber_out):
                output = None
                m_out = to_order(degree_out)

                for degree_in, _ in enumerate(self.fiber_in):
                    etype = f'({degree_in},{degree_out})'

                    m_in = to_order(degree_in)
                    m_min = min(m_in, m_out)

                    degree_min = min(degree_in, degree_out)

                    # 获取源和目标（邻居）表示
                    xi, xj = source[degree_in], target[degree_in]

                    x = get_at('b [i] d m, b j k -> b j k d m', xj, neighbor_indices)

                    # 如果需要将 xi 和 xj 投影
                    if self.project_xi_xj:
                        xi = rearrange(xi, 'b i d m -> b i 1 d m')
                        x = x + xi

                    # 乘以 D(R) - 旋转到 z 轴
                    if degree_in > 0:
                        Di = D[degree_in]
                        x = einsum(Di, x, '... mi1 mi2, ... li mi1 -> ... li mi2')

                    # 如果 degree_in != degree_out，则移除一些 0s
                    maybe_input_slice = slice_for_centering_y_to_x(m_in, m_min)
                    maybe_output_pad = pad_for_centering_y_to_x(m_out, m_min)
                    x = x[..., maybe_input_slice]

                    # 在序列维度上按块处理输入、边和基础
                    kernel_fn = self.kernel_unary[etype]
                    edge_features = safe_cat(edges, rel_dist, dim=-1)
                    B = basis.get(etype, None)
                    R = kernel_fn(edge_features)

                    # 如果没有基础
                    if not exists(B):
                        output_chunk = einsum(R, x, '... lo li, ... li mi -> ... lo mi')
                    else:
                        y = x.clone()
                        x = repeat(x, '... mi -> ... mi mf r', mf=(B.shape[-1] + 1) // 2, r=2)
                        x, x_to_flip = x.unbind(dim=-1)
                        x_flipped = torch.flip(x_to_flip, dims=(-2,))
                        x = torch.stack((x, x_flipped), dim=-1)
                        x = rearrange(x, '... mf r -> ... (mf r)', r=2)
                        x = x[..., :-1]
                        output_chunk = opt_einsum('... o i, m f, ... i m f -> ... o m', R, B, x)

                    # 如果 degree_out < degree_in
                    output_chunk = F.pad(output_chunk, (maybe_output_pad, maybe_output_pad), value=0.)
                    output = safe_cat(output, output_chunk, dim=-2)

                # 乘以 D(R^-1) - 从 z 轴旋转回来
                if degree_out > 0:
                    Do = D[degree_out]
                    output = einsum(output, Do, '... lo mo1, ... mo2 mo1 -> ... lo mo2')

                # 沿 j（邻居）维度池化或不池化
                if self.pool:
                    output = masked_mean(output, neighbor_masks, dim=2)

                outputs[degree_out] = output

            # 如果不需要自相互作用且不需要输出投影，则返回输出
            if not self.self_interaction and not self.project_out:
                return outputs

            # 如果需要输出投影
            if self.project_out:
                outputs = self.to_out(outputs)

            self_interact_out = self.self_interact(inp)

            # 如果需要池化
            if self.pool:
                return residual_fn(outputs, self_interact_out)

            self_interact_out = {k: rearrange(v, '... d m -> ... 1 d m') for k, v in self_interact_out.items()}
            outputs = {degree: torch.cat(tensors, dim=-3) for degree, tensors in enumerate(zip(self_interact_out.values(), outputs.values()))}
            return outputs
# 定义一个名为 Radial 的类，继承自 nn.Module
class Radial(nn.Module):
    # 初始化函数，接受输入特征的度数、通道数、输出特征的度数、通道数、边维度和径向隐藏维度等参数
    def __init__(
        self,
        degree_in,
        nc_in,
        degree_out,
        nc_out,
        edge_dim = 0,
        radial_hidden_dim = 64
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化类的属性
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        # 将输出特征的度数转换为对应的顺序
        self.d_out = to_order(degree_out)
        self.edge_dim = edge_dim

        # 设置中间维度为径向隐藏维度
        mid_dim = radial_hidden_dim
        edge_dim = default(edge_dim, 0)

        # 定义径向网络的结构
        self.rp = nn.Sequential(
            nn.Linear(edge_dim + 1, mid_dim),
            nn.SiLU(),
            LayerNorm(mid_dim),
            nn.Linear(mid_dim, mid_dim),
            nn.SiLU(),
            LayerNorm(mid_dim),
            nn.Linear(mid_dim, nc_in * nc_out),
            Rearrange('... (lo li) -> ... lo li', li = nc_in, lo = nc_out)
        )

    # 前向传播函数
    def forward(self, feat):
        # 返回径向网络的前向传播结果
        return self.rp(feat)

# 定义名为 FeedForward 的类，继承自 nn.Module
class FeedForward(nn.Module):
    # 初始化函数，接受输入特征的维度、输出特征的维度、倍数、是否包含类型归一化和是否初始化输出为零等参数
    @beartype
    def __init__(
        self,
        fiber: Tuple[int, ...],
        fiber_out: Optional[Tuple[int, ...]] = None,
        mult = 4,
        include_htype_norms = True,
        init_out_zero = True
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化类的属性
        self.fiber = fiber

        # 计算隐藏层特征的维度
        fiber_hidden = tuple(dim * mult for dim in fiber)

        project_in_fiber = fiber
        project_in_fiber_hidden = tuple_set_at_index(fiber_hidden, 0, sum(fiber_hidden))

        # 根据是否包含类型归一化来调整输入特征的维度
        self.include_htype_norms = include_htype_norms
        if include_htype_norms:
            project_in_fiber = tuple_set_at_index(project_in_fiber, 0, sum(fiber))

        fiber_out = default(fiber_out, fiber)

        # 定义前向传播的结构
        self.prenorm     = Norm(fiber)
        self.project_in  = Linear(project_in_fiber, project_in_fiber_hidden)
        self.gate        = Gate(project_in_fiber_hidden)
        self.project_out = Linear(fiber_hidden, fiber_out)

        # 如果初始化输出为零，则将输出初始化为零
        if init_out_zero:
            self.project_out.init_zero_()

    # 前向传播函数
    def forward(self, features):
        # 对输入特征进行预归一化
        outputs = self.prenorm(features)

        # 如果包含类型归一化，则对类型进行归一化
        if self.include_htype_norms:
            type0, *htypes = [*outputs.values()]
            htypes = map(lambda t: t.norm(dim = -1, keepdim = True), htypes)
            type0 = torch.cat((type0, *htypes), dim = -2)
            outputs[0] = type0

        # 对特征进行投影
        outputs = self.project_in(outputs)
        outputs = self.gate(outputs)
        outputs = self.project_out(outputs)
        return outputs

# 定义全局线性注意力类
class LinearAttention(nn.Module):
    # 初始化函数，接受特征维度、头维度和头数等参数
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化头数
        self.heads = heads
        dim_inner = dim_head * heads
        # 线性变换得到查询、键、值
        self.to_qkv = nn.Linear(dim, dim_inner * 3)

    # 前向传播函数
    def forward(self, x, mask = None):
        # 判断输入是否包含度数维度
        has_degree_m_dim = x.ndim == 4

        if has_degree_m_dim:
            x = rearrange(x, '... 1 -> ...')

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b 1 n 1')
            k = k.masked_fill(~mask, -torch.finfo(q.dtype).max)
            v = v.masked_fill(~mask, 0.)

        k = k.softmax(dim = -2)
        q = q.softmax(dim = -1)

        kv = einsum(k, v, 'b h n d, b h n e -> b h d e')
        out = einsum(kv, q, 'b h d e, b h n d -> b h n e')
        out = rearrange(out, 'b h n d -> b n (h d)')

        if has_degree_m_dim:
            out = rearrange(out, '... -> ... 1')

        return out

# 定义 L2 距离注意力类
class L2DistAttention(nn.Module):
    @beartype
    # 初始化函数，定义了模型的各种参数和层
    def __init__(
        self,
        fiber: Tuple[int, ...],  # 输入特征的维度
        dim_head: Union[int, Tuple[int, ...]] = 64,  # 头的维度
        heads: Union[int, Tuple[int, ...]] = 8,  # 头的数量
        attend_self = False,  # 是否自注意力
        edge_dim = None,  # 边的维度
        single_headed_kv = False,  # 是否单头键值对
        radial_hidden_dim = 64,  # 径向隐藏维度
        splits = 4,  # 分割数
        linear_attn_dim_head = 8,  # 线性注意力头的维度
        num_linear_attn_heads = 0,  # 线性注意力头的数量
        init_out_zero = True,  # 输出是否初始化为零
        gate_attn_head_outputs = True  # 是否对注意力头输出进行门控
    ):
        super().__init__()  # 调用父类的初始化函数
        num_degrees = len(fiber)  # 输入特征的维度数

        dim_head = cast_tuple(dim_head, num_degrees)  # 将头的维度转换为元组
        assert len(dim_head) == num_degrees  # 确保头的维度数与输入特征的维度数相同

        heads = cast_tuple(heads, num_degrees)  # 将头的数量转换为元组
        assert len(heads) == num_degrees  # 确保头的数量与输入特征的维度数相同

        hidden_fiber = tuple(dim * head for dim, head in zip(dim_head, heads))  # 计算隐藏层的维度

        self.single_headed_kv = single_headed_kv  # 是否单头键值对
        self.attend_self = attend_self  # 是否自注意力

        kv_hidden_fiber = hidden_fiber if not single_headed_kv else dim_head  # 键值对隐藏层的维度
        kv_hidden_fiber = tuple(dim * 2 for dim in kv_hidden_fiber)  # 键值对隐藏层的维度

        self.scale = tuple(dim ** -0.5 for dim in dim_head)  # 缩放因子
        self.heads = heads  # 头的数量

        self.prenorm = Norm(fiber)  # 规范化层

        self.to_q = Linear(fiber, hidden_fiber)  # 查询层
        self.to_kv = DTP(fiber, kv_hidden_fiber, radial_hidden_dim = radial_hidden_dim, edge_dim = edge_dim, pool = False, self_interaction = attend_self)  # 键值对层

        # 线性注意力头

        self.has_linear_attn = num_linear_attn_heads > 0  # 是否有线性注意力头

        if self.has_linear_attn:
            degree_zero_dim = fiber[0]  # 输入特征的第一个维度
            self.linear_attn = TaylorSeriesLinearAttn(degree_zero_dim, dim_head = linear_attn_dim_head, heads = num_linear_attn_heads, combine_heads = False, gate_value_heads = True)  # 线性注意力层
            hidden_fiber = tuple_set_at_index(hidden_fiber, 0, hidden_fiber[0] + linear_attn_dim_head * num_linear_attn_heads)  # 更新隐藏层的维度

        # 对所有度的输出进行门控，以允许不关注任何内容

        self.attn_head_gates = None  # 注意力头的门控

        if gate_attn_head_outputs:
            self.attn_head_gates = nn.Sequential(
                Rearrange('... d 1 -> ... d'),
                nn.Linear(fiber[0], sum(heads)),
                nn.Sigmoid(),
                Rearrange('... n h -> ... h n 1 1')
            )  # 门控层

        # 合并头

        self.to_out = Linear(hidden_fiber, fiber)  # 输出层

        if init_out_zero:
            self.to_out.init_zero_()  # 初始化输出为零

    @beartype
    def forward(
        self,
        features,  # 特征
        edge_info: EdgeInfo,  # 边信息
        rel_dist,  # 相对距离
        basis,  # 基础
        D,  # D
        mask = None  # 掩码
        ):
            # 获取单头键值对应的标志
            one_head_kv = self.single_headed_kv

            # 获取特征的设备和数据类型
            device, dtype = get_tensor_device_and_dtype(features)
            # 获取邻居索引、邻居掩码和边信息
            neighbor_indices, neighbor_mask, edges = edge_info

            # 如果邻居掩码存在
            if exists(neighbor_mask):
                # 重新排列邻居掩码的维度
                neighbor_mask = rearrange(neighbor_mask, 'b i j -> b 1 i j')

                # 如果需要考虑自身
                if self.attend_self:
                    # 在邻居掩码上进行填充
                    neighbor_mask = F.pad(neighbor_mask, (1, 0), value = True)

            # 对特征进行预处理
            features = self.prenorm(features)

            # 生成查询、键、值
            queries = self.to_q(features)

            keyvalues   = self.to_kv(
                features,
                edge_info = edge_info,
                rel_dist = rel_dist,
                basis = basis,
                D = D
            )

            # 创建门
            gates = (None,) * len(self.heads)

            # 如果存在注意力头门控
            if exists(self.attn_head_gates):
                # 对特征的第一个元素应用注意力头门控，并按头数分割
                gates = self.attn_head_gates(features[0]).split(self.heads, dim = -4)

            # 单头与多头的区别
            kv_einsum_eq = 'b h i j d m' if not one_head_kv else 'b i j d m'

            outputs = {}

            # 遍历特征的键，门，头数和缩放因子
            for degree, gate, h, scale in zip(features.keys(), gates, self.heads, self.scale):
                # 判断是否为零度
                is_degree_zero = degree == 0

                q, kv = map(lambda t: t[degree], (queries, keyvalues))

                q = rearrange(q, 'b i (h d) m -> b h i d m', h = h)

                # 如果不是单头键值
                if not one_head_kv:
                    kv = rearrange(kv, f'b i j (h d) m -> b h i j d m', h = h)

                k, v = kv.chunk(2, dim = -2)

                # 如果是单头键值
                if one_head_kv:
                    k = repeat(k, 'b i j d m -> b h i j d m', h = h)

                q = repeat(q, 'b h i d m -> b h i j d m', j = k.shape[-3])

                # 如果是零度
                if is_degree_zero:
                    q, k = map(lambda t: rearrange(t, '... 1 -> ...'), (q, k))

                sim = -cdist(q, k) * scale

                # 如果不是零度
                if not is_degree_zero:
                    sim = sim.sum(dim = -1)
                    sim = sim.masked_fill(~neighbor_mask, -torch.finfo(sim.dtype).max)

                attn = sim.softmax(dim = -1)
                out = einsum(attn, v, f'b h i j, {kv_einsum_eq} -> b h i d m')

                # 如果门存在
                if exists(gate):
                    out = out * gate

                outputs[degree] = rearrange(out, 'b h n d m -> b n (h d) m')

            # 如果具有线性注意力
            if self.has_linear_attn:
                linear_attn_input = rearrange(features[0], '... 1 -> ...')
                lin_attn_out = self.linear_attn(linear_attn_input, mask = mask)
                lin_attn_out = rearrange(lin_attn_out, '... -> ... 1')
                outputs[0] = torch.cat((outputs[0], lin_attn_out), dim = -2)

            # 返回输出
            return self.to_out(outputs)
# 定义一个多层感知机注意力模型的类
class MLPAttention(nn.Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        fiber: Tuple[int, ...],  # 输入特征的维度
        dim_head: Union[int, Tuple[int, ...]] = 64,  # 注意力头的维度
        heads: Union[int, Tuple[int, ...]] = 8,  # 注意力头的数量
        attend_self = False,  # 是否自注意力
        edge_dim = None,  # 边的维度
        splits = 4,  # 分割数
        single_headed_kv = False,  # 是否单头键值对
        attn_leakyrelu_slope = 0.1,  # 注意力LeakyReLU斜率
        attn_hidden_dim_mult = 4,  # 注意力隐藏层维度倍数
        radial_hidden_dim = 16,  # 径向隐藏层维度
        linear_attn_dim_head = 8,  # 线性注意力头维度
        num_linear_attn_heads = 0,  # 线性注意力头数量
        init_out_zero = True,  # 输出初始化为零
        gate_attn_head_outputs = True,  # 是否门控注意力头输出
        **kwargs
    ):
        super().__init__()
        num_degrees = len(fiber)

        dim_head = cast_tuple(dim_head, num_degrees)  # 将dim_head转换为元组
        assert len(dim_head) == num_degrees

        heads = cast_tuple(heads, num_degrees)  # 将heads转换为元组
        assert len(heads) == num_degrees

        hidden_fiber = tuple(dim * head for dim, head in zip(dim_head, heads))  # 计算隐藏层的维度

        self.single_headed_kv = single_headed_kv  # 是否单头键值对
        value_hidden_fiber = hidden_fiber if not single_headed_kv else dim_head  # 值的隐藏层维度

        self.attend_self = attend_self  # 是否自注意力

        self.scale = tuple(dim ** -0.5 for dim in dim_head)  # 缩放因子
        self.heads = heads  # 注意力头数量

        self.prenorm = Norm(fiber)  # 规范化层

        # type 0需要更大的维度，用于
        # (1) 在值分支上对htypes进行门控
        # (2) 注意力logits，初始维度等于头的数量

        type0_dim = value_hidden_fiber[0]  # 类型0的维度
        htype_dims = sum(value_hidden_fiber[1:])  # htype的维度

        value_gate_fiber = tuple_set_at_index(value_hidden_fiber, 0, type0_dim + htype_dims)  # 值门控的维度

        attn_hidden_dims = tuple(head * attn_hidden_dim_mult for head in heads)  # 注意力隐藏层的维度

        intermediate_fiber = tuple_set_at_index(value_hidden_fiber, 0, sum(attn_hidden_dims) + type0_dim + htype_dims)  # 中间层的维度
        self.intermediate_type0_split = [*attn_hidden_dims, type0_dim + htype_dims]  # 类型0的分割

        # 主分支张量乘积

        self.to_attn_and_v = DTP(fiber, intermediate_fiber, radial_hidden_dim = radial_hidden_dim, edge_dim = edge_dim, pool = False, self_interaction = attend_self)  # 注意力和值的张量乘积

        # 注意力分支的非线性投影到注意力logits

        self.to_attn_logits = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(attn_leakyrelu_slope),  # LeakyReLU激活函数
                nn.Linear(attn_hidden_dim, h, bias = False)  # 线性层
            ) for attn_hidden_dim, h in zip(attn_hidden_dims, self.heads)
        ])

        # 值分支的非线性变换
        # todo - 这里需要一个DTP吗？

        self.to_values = nn.Sequential(
            Gate(value_gate_fiber),  # 门控层
            Linear(value_hidden_fiber, value_hidden_fiber)  # 线性层
        )

        # 线性注意力头

        self.has_linear_attn = num_linear_attn_heads > 0  # 是否有线性注意力头

        if self.has_linear_attn:
            degree_zero_dim = fiber[0]
            self.linear_attn = TaylorSeriesLinearAttn(degree_zero_dim, dim_head = linear_attn_dim_head, heads = num_linear_attn_heads, combine_heads = False)  # 线性注意力

            hidden_fiber = tuple_set_at_index(hidden_fiber, 0, hidden_fiber[0] + linear_attn_dim_head * num_linear_attn_heads)  # 更新隐藏层的维度

        # 门控所有度输出的头
        # 允许不关注任何内容

        self.attn_head_gates = None

        if gate_attn_head_outputs:
            self.attn_head_gates = nn.Sequential(
                Rearrange('... d 1 -> ... d'),  # 重新排列维度
                nn.Linear(fiber[0], sum(heads)),  # 线性层
                nn.Sigmoid(),  # Sigmoid激活函数
                Rearrange('... h -> ... h 1 1')  # 重新排列维度
            )

        # 合并头和投影输出

        self.to_out = Linear(hidden_fiber, fiber)  # 输出层

        if init_out_zero:
            self.to_out.init_zero_()  # 初始化输出为零

    @beartype
    def forward(
        self,
        features,
        edge_info: EdgeInfo,
        rel_dist,
        basis,
        D,
        mask = None
        ):
            # 获取单头键值对
            one_headed_kv = self.single_headed_kv

            # 解包边信息
            _, neighbor_mask, _ = edge_info

            # 如果邻居掩码存在
            if exists(neighbor_mask):
                # 如果需要考虑自身，则在左侧填充一个位置
                if self.attend_self:
                    neighbor_mask = F.pad(neighbor_mask, (1, 0), value = True)

                # 重新排列邻居掩码的维度
                neighbor_mask = rearrange(neighbor_mask, '... -> ... 1')

            # 对特征进行预处理
            features = self.prenorm(features)

            # 获取注意力和值的中间结果
            intermediate = self.to_attn_and_v(
                features,
                edge_info = edge_info,
                rel_dist = rel_dist,
                basis = basis,
                D = D
            )

            # 拆分注意力分支和值分支
            *attn_branch_type0, value_branch_type0 = intermediate[0].split(self.intermediate_type0_split, dim = -2)

            # 将值分支替换回中间结果
            intermediate[0] = value_branch_type0

            # 创建门控
            gates = (None,) * len(self.heads)

            # 如果存在注意力头门控
            if exists(self.attn_head_gates):
                gates = self.attn_head_gates(features[0]).split(self.heads, dim = -3)

            # 处理注意力分支
            attentions = []

            for fn, attn_intermediate, scale in zip(self.to_attn_logits, attn_branch_type0, self.scale):
                attn_intermediate = rearrange(attn_intermediate, '... 1 -> ...')
                attn_logits = fn(attn_intermediate)
                attn_logits = attn_logits * scale

                # 如果邻居掩码存在，则进行掩码处理
                if exists(neighbor_mask):
                    attn_logits = attn_logits.masked_fill(~neighbor_mask, -torch.finfo(attn_logits.dtype).max)

                # 计算注意力权重
                attn = attn_logits.softmax(dim = -2) # (batch, source, target, heads)
                attentions.append(attn)

            # 处理值分支
            values = self.to_values(intermediate)

            # 使用注意力矩阵聚合值
            outputs = {}

            value_einsum_eq = 'b i j h d m' if not one_headed_kv else 'b i j d m'

            for degree, (attn, value, gate, h) in enumerate(zip(attentions, values.values(), gates, self.heads)):
                if not one_headed_kv:
                    value = rearrange(value, 'b i j (h d) m -> b i j h d m', h = h)

                out = einsum(attn, value, f'b i j h, {value_einsum_eq} -> b i h d m')

                if exists(gate):
                    out = out * gate

                out = rearrange(out, 'b i h d m -> b i (h d) m')
                outputs[degree] = out

            # 线性注意力
            if self.has_linear_attn:
                linear_attn_input = rearrange(features[0], '... 1 -> ...')
                lin_attn_out = self.linear_attn(linear_attn_input, mask = mask)
                lin_attn_out = rearrange(lin_attn_out, '... -> ... 1')

                outputs[0] = torch.cat((outputs[0], lin_attn_out), dim = -2)

            # 合并头部输出
            return self.to_out(outputs)
# 主类定义

class Equiformer(nn.Module):
    # 初始化函数，使用装饰器进行参数类型检查
    @beartype
    def __init__(
        self,
        *,
        dim: Union[int, Tuple[int, ...]],  # 维度参数，可以是整数或元组
        dim_in: Optional[Union[int, Tuple[int, ...]]] = None,  # 输入维度参数，可选
        num_degrees = 2,  # 角度数量，默认为2
        input_degrees = 1,  # 输入角度数量，默认为1
        heads: Union[int, Tuple[int, ...]] = 8,  # 头数，可以是整数或元组，默认为8
        dim_head: Union[int, Tuple[int, ...]] = 24,  # 头维度，可以是整数或元组，默认为24
        depth = 2,  # 深度，默认为2
        valid_radius = 1e5,  # 有效半径，默认为1e5
        num_neighbors = float('inf'),  # 邻居数量，默认为无穷大
        reduce_dim_out = False,  # 是否减少输出维度，默认为False
        radial_hidden_dim = 64,  # 径向隐藏维度，默认为64
        num_tokens = None,  # 令牌数量，默认为None
        num_positions = None,  # 位置数量，默认为None
        num_edge_tokens = None,  # 边令牌数量，默认为None
        edge_dim = None,  # 边维度，默认为None
        attend_self = True,  # 是否自注意，默认为True
        splits = 4,  # 分割数，默认为4
        linear_out = True,  # 是否线性输出，默认为True
        embedding_grad_frac = 0.5,  # 嵌入梯度比例，默认为0.5
        single_headed_kv = False,  # 是否对点积注意力进行单头键/值操作，以节省内存和计算资源，默认为False
        ff_include_htype_norms = False,  # 是否在类型0投影中还涉及所有更高类型的规范化，在前馈第一次投影中。这允许所有更高类型受其他类型规范化的门控
        l2_dist_attention = True,  # 是否使用L2距离注意力，默认为True。将其设置为False以使用论文中提出的MLP注意力，但是点积注意力与-cdist相似性仍然要好得多，而且我甚至还没有将距离（旋转嵌入）旋转到类型0特征中
        reversible = False,  # 打开可逆网络，以在不增加深度内存成本的情况下扩展深度
        attend_sparse_neighbors = False,  # 能够接受邻接矩阵，默认为False
        gate_attn_head_outputs = True,  # 对每个注意力头输出进行门控，以允许不关注任何内容
        num_adj_degrees_embed = None,  # 邻接度嵌入数量，默认为None
        adj_dim = 0,  # 邻接维度，默认为0
        max_sparse_neighbors = float('inf'),  # 最大稀疏邻居数量，默认为无穷大
        **kwargs  # 其他关键字参数
    # 初始化函数，继承父类的初始化方法
    def __init__(
        self,
        embedding_grad_frac,
        dim,
        num_degrees,
        dim_in,
        input_degrees,
        num_tokens,
        num_positions,
        edge_dim,
        num_edge_tokens,
        attend_sparse_neighbors,
        max_sparse_neighbors,
        num_adj_degrees_embed,
        adj_dim,
        valid_radius,
        num_neighbors,
        radial_hidden_dim,
        depth,
        heads,
        dim_head,
        attend_self,
        single_headed_kv,
        l2_dist_attention,
        gate_attn_head_outputs,
        reversible,
        ff_include_htype_norms,
        linear_out,
        reduce_dim_out,
        **kwargs
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 用于更稳定的训练的嵌入梯度比例
        self.embedding_grad_frac = embedding_grad_frac

        # 为所有类型决定隐藏维度
        self.dim = cast_tuple(dim, num_degrees)
        assert len(self.dim) == num_degrees

        self.num_degrees = len(self.dim)

        # 为所有类型决定输入维度
        dim_in = default(dim_in, (self.dim[0],))
        self.dim_in = cast_tuple(dim_in, input_degrees)
        assert len(self.dim_in) == input_degrees

        self.input_degrees = len(self.dim_in)

        # token 嵌入
        type0_feat_dim = self.dim_in[0]
        self.type0_feat_dim = type0_feat_dim
        self.token_emb = nn.Embedding(num_tokens, type0_feat_dim) if exists(num_tokens) else None

        # 位置嵌入
        self.num_positions = num_positions
        self.pos_emb = nn.Embedding(num_positions, type0_feat_dim) if exists(num_positions) else None

        # 初始化嵌入
        if exists(self.token_emb):
            nn.init.normal_(self.token_emb.weight, std=1e-2)

        if exists(self.pos_emb):
            nn.init.normal_(self.pos_emb.weight, std=1e-2)

        # 边
        assert not (exists(num_edge_tokens) and not exists(edge_dim)), 'edge dimension (edge_dim) must be supplied if equiformer is to have edge tokens'
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None
        self.has_edges = exists(edge_dim) and edge_dim > 0

        # 稀疏邻居，从邻接矩阵或传入的边派生
        self.attend_sparse_neighbors = attend_sparse_neighbors
        self.max_sparse_neighbors = max_sparse_neighbors

        # 邻接邻居派生和嵌入
        assert not exists(num_adj_degrees_embed) or num_adj_degrees_embed >= 1, 'number of adjacent degrees to embed must be 1 or greater'
        self.num_adj_degrees_embed = num_adj_degrees_embed
        self.adj_emb = nn.Embedding(num_adj_degrees_embed + 1, adj_dim) if exists(num_adj_degrees_embed) and adj_dim > 0 else None
        edge_dim = (edge_dim if self.has_edges else 0) + (adj_dim if exists(self.adj_emb) else 0)

        # 邻居超参数
        self.valid_radius = valid_radius
        self.num_neighbors = num_neighbors

        # 主网络
        self.tp_in = DTP(
            self.dim_in,
            self.dim,
            edge_dim=edge_dim,
            radial_hidden_dim=radial_hidden_dim
        )

        # 主干
        self.layers = []

        attention_klass = L2DistAttention if l2_dist_attention else MLPAttention

        for ind in range(depth):
            self.layers.append((
                attention_klass(
                    self.dim,
                    heads=heads,
                    dim_head=dim_head,
                    attend_self=attend_self,
                    edge_dim=edge_dim,
                    single_headed_kv=single_headed_kv,
                    radial_hidden_dim=radial_hidden_dim,
                    gate_attn_head_outputs=gate_attn_head_outputs,
                    **kwargs
                ),
                FeedForward(self.dim, include_htype_norms=ff_include_htype_norms)
            ))

        SequenceKlass = ReversibleSequence if reversible else SequentialSequence

        self.layers = SequenceKlass(self.layers)

        # 输出
        self.norm = Norm(self.dim)

        proj_out_klass = Linear if linear_out else FeedForward

        self.ff_out = proj_out_klass(self.dim, (1,) * self.num_degrees) if reduce_dim_out else None

        # 基础现在是常数
        # pytorch 目前没有 BufferDict，用 Python 属性来实现一个解决方案
        self.basis = get_basis(self.num_degrees - 1)

    @property
    def basis(self):
        out = dict()
        for k in self.basis_keys:
            out[k] = getattr(self, f'basis:{k}')
        return out

    @basis.setter
    # 定义一个方法，用于设置基础信息
    def basis(self, basis):
        # 将传入的基础信息的键存储到对象的属性中
        self.basis_keys = basis.keys()

        # 遍历基础信息的键值对
        for k, v in basis.items():
            # 将每个键值对注册为缓冲区
            self.register_buffer(f'basis:{k}', v)

    # 定义一个属性，用于获取模型参数所在的设备
    @property
    def device(self):
        # 返回第一个参数的设备信息
        return next(self.parameters()).device

    # 定义一个前向传播方法，接受输入数据、坐标、掩码、邻接矩阵、边信息和是否返回池化结果等参数
    @beartype
    def forward(
        self,
        inputs: Union[Tensor, Dict[int, Tensor]],
        coors: Tensor,
        mask = None,
        adj_mat = None,
        edges = None,
        return_pooled = False
```