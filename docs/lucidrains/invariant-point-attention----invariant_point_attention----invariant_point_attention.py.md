# `.\lucidrains\invariant-point-attention\invariant_point_attention\invariant_point_attention.py`

```
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

# helpers

# 检查值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回给定类型的最大负值
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

@contextmanager
def disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value

# classes

class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 4,
        pairwise_repr_dim = None,
        require_pairwise_repr = True,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr

        # num attention contributions

        num_attn_logits = 3 if require_pairwise_repr else 2

        # qkv projection for scalar attention (normal)

        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5

        self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias = False)

        # qkv projection for point attention (coordinate and orientation aware)

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weights = nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2)) ** -0.5

        self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias = False)
        self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias = False)
        self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias = False)

        # pairwise representation projection to attention bias

        pairwise_repr_dim = default(pairwise_repr_dim, dim) if require_pairwise_repr else 0

        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits ** -0.5

            self.to_pairwise_attn_bias = nn.Sequential(
                nn.Linear(pairwise_repr_dim, heads),
                Rearrange('b ... h -> (b h) ...')
            )

        # combine out - scalar dim + pairwise dim + point dim * (3 for coordinates in R3 and then 1 for norm)

        self.to_out = nn.Linear(heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim * (3 + 1)), dim)

    def forward(
        self,
        single_repr,
        pairwise_repr = None,
        *,
        rotations,
        translations,
        mask = None
    ):
        pass

# one transformer block based on IPA

def FeedForward(dim, mult = 1., num_layers = 2, act = nn.ReLU):
    layers = []
    dim_hidden = dim * mult

    for ind in range(num_layers):
        is_first = ind == 0
        is_last  = ind == (num_layers - 1)
        dim_in   = dim if is_first else dim_hidden
        dim_out  = dim if is_last else dim_hidden

        layers.append(nn.Linear(dim_in, dim_out))

        if is_last:
            continue

        layers.append(act())

    return nn.Sequential(*layers)

class IPABlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult = 1,
        ff_num_layers = 3,          # in the paper, they used 3 layer transition (feedforward) block
        post_norm = True,           # in the paper, they used post-layernorm - offering pre-norm as well
        post_attn_dropout = 0.,
        post_ff_dropout = 0.,
        **kwargs
    ):
        pass
    # 初始化函数，继承父类的初始化方法
    def __init__(
        self,
        post_norm: bool
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置是否在后处理时进行归一化
        self.post_norm = post_norm

        # 初始化注意力层的归一化层
        self.attn_norm = nn.LayerNorm(dim)
        # 创建不变点注意力层对象
        self.attn = InvariantPointAttention(dim = dim, **kwargs)
        # 初始化注意力层后的丢弃层
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        # 初始化前馈神经网络的归一化层
        self.ff_norm = nn.LayerNorm(dim)
        # 创建前馈神经网络对象
        self.ff = FeedForward(dim, mult = ff_mult, num_layers = ff_num_layers)
        # 初始化前馈神经网络后的丢弃层
        self.post_ff_dropout = nn.Dropout(post_ff_dropout)

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 获取是否在后处理时进行归一化的标志
        post_norm = self.post_norm

        # 如果不进行后处理归一化，则直接使用输入作为注意力层的输入，否则对输入进行归一化
        attn_input = x if post_norm else self.attn_norm(x)
        # 经过注意力层的计算，并加上残差连接
        x = self.attn(attn_input, **kwargs) + x
        # 经过注意力层后的丢弃操作
        x = self.post_attn_dropout(x)
        # 如果不进行后处理归一化，则对输出进行归一化，否则直接输出
        x = self.attn_norm(x) if post_norm else x

        # 如果不进行后处理归一化，则直接使用输入作为前馈神经网络的输入，否则对输入进行归一化
        ff_input = x if post_norm else self.ff_norm(x)
        # 经过前馈神经网络的计算，并加上残差连接
        x = self.ff(ff_input) + x
        # 经过前馈神经网络后的丢弃操作
        x = self.post_ff_dropout(x)
        # 如果不进行后处理归一化，则对输出进行归一化，否则直接输出
        x = self.ff_norm(x) if post_norm else x
        # 返回最终输出
        return x
# 添加一个 IPA Transformer - 迭代更新旋转和平移

# 这部分与 AF2 不太准确，因为 AF2 在每一层都应用了一个 FAPE 辅助损失，以及在旋转上应用了一个停止梯度
# 这只是一个尝试，看看是否可以演变成更普遍可用的东西

class IPATransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens = None,
        predict_points = False,
        detach_rotations = True,
        **kwargs
    ):
        super().__init__()

        # 使用来自 pytorch3d 的四元数函数

        try:
            from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix
            self.quaternion_to_matrix = quaternion_to_matrix
            self.quaternion_multiply = quaternion_multiply
        except (ImportError, ModuleNotFoundError) as err:
            print('unable to import pytorch3d - please install with `conda install pytorch3d -c pytorch3d`')
            raise err

        # 嵌入

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        # 层

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                IPABlock(dim = dim, **kwargs),
                nn.Linear(dim, 6)
            ]))

        # 是否分离旋转以保持训练稳定性

        self.detach_rotations = detach_rotations

        # 输出

        self.predict_points = predict_points

        if predict_points:
            self.to_points = nn.Linear(dim, 3)

    def forward(
        self,
        single_repr,
        *,
        translations = None,
        quaternions = None,
        pairwise_repr = None,
        mask = None
    ):
        x, device, quaternion_multiply, quaternion_to_matrix = single_repr, single_repr.device, self.quaternion_multiply, self.quaternion_to_matrix
        b, n, *_ = x.shape

        if exists(self.token_emb):
            x = self.token_emb(x)

        # 如果没有传入初始四元数，从单位矩阵开始

        if not exists(quaternions):
            quaternions = torch.tensor([1., 0., 0., 0.], device = device) # 初始旋转
            quaternions = repeat(quaternions, 'd -> b n d', b = b, n = n)

        # 如果没有传入平移，从零开始

        if not exists(translations):
            translations = torch.zeros((b, n, 3), device = device)

        # 遍历层并应用不变点注意力和前馈

        for block, to_update in self.layers:
            rotations = quaternion_to_matrix(quaternions)

            if self.detach_rotations:
                rotations = rotations.detach()

            x = block(
                x,
                pairwise_repr = pairwise_repr,
                rotations = rotations,
                translations = translations
            )

            # 更新四元数和平移

            quaternion_update, translation_update = to_update(x).chunk(2, dim = -1)
            quaternion_update = F.pad(quaternion_update, (1, 0), value = 1.)
            quaternion_update = quaternion_update / torch.linalg.norm(quaternion_update, dim=-1, keepdim=True)
            quaternions = quaternion_multiply(quaternions, quaternion_update)
            translations = translations + einsum('b n c, b n c r -> b n r', translation_update, rotations)

        if not self.predict_points:
            return x, translations, quaternions

        points_local = self.to_points(x)
        rotations = quaternion_to_matrix(quaternions)
        points_global = einsum('b n c, b n c d -> b n d', points_local, rotations) + translations
        return points_global
```