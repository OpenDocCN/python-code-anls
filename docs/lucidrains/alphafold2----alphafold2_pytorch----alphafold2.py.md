# `.\lucidrains\alphafold2\alphafold2_pytorch\alphafold2.py`

```py
import torch
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from inspect import isfunction
from functools import partial
from dataclasses import dataclass
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from alphafold2_pytorch.utils import *
import alphafold2_pytorch.constants as constants
from alphafold2_pytorch.mlm import MLM

# structure module

from invariant_point_attention import IPABlock
from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix

# constants

@dataclass
class Recyclables:
    coords: torch.Tensor
    single_msa_repr_row: torch.Tensor
    pairwise_repr: torch.Tensor

@dataclass
class ReturnValues:
    distance: torch.Tensor = None
    theta: torch.Tensor = None
    phi: torch.Tensor = None
    omega: torch.Tensor = None
    msa_mlm_loss: torch.Tensor = None
    recyclables: Recyclables = None

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# helper classes

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return self.val

# feed forward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
        init_zero_(self.net[-1])

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        gating = True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(self, x, mask = None, attn_bias = None, context = None, context_mask = None, tie_dim = None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # query / key similarities

        if exists(tie_dim):
            # as in the paper, for the extra MSAs
            # they average the queries along the rows of the MSAs
            # they named this particular module MSAColumnGlobalAttention

            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r = tie_dim), (q, k))
            q = q.mean(dim = 1)

            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)

        if exists(attn_bias):
            dots = dots + attn_bias

        # masking

        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2], device = device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating

        gates = self.gating(x)
        out = out * gates.sigmoid()

        # combine to out

        out = self.to_out(out)
        return out

class AxialAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        row_attn = True,
        col_attn = True,
        accept_edges = False,
        global_query_attn = False,
        **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn

        self.norm = nn.LayerNorm(dim)

        self.attn = Attention(dim = dim, heads = heads, **kwargs)

        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None

    def forward(self, x, edges = None, mask = None):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'

        b, h, w, d = x.shape

        x = self.norm(x)

        # axial attention

        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'

        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'

        x = rearrange(x, input_fold_eq)

        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = axial_dim)

        tie_dim = axial_dim if self.global_query_attn else None

        out = self.attn(x, mask = mask, attn_bias = attn_bias, tie_dim = tie_dim)
        out = rearrange(out, output_fold_eq, h = h, w = w)

        return out

class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        mix = 'ingoing'
    # 初始化函数，继承父类的初始化方法
    def __init__(
        super().__init__()
        # 断言混合参数只能是'ingoing'或'outgoing'
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        # 如果隐藏维度未指定，则默认为输入维度
        hidden_dim = default(hidden_dim, dim)
        # 对输入进行层归一化
        self.norm = nn.LayerNorm(dim)

        # 左投影层
        self.left_proj = nn.Linear(dim, hidden_dim)
        # 右投影层
        self.right_proj = nn.Linear(dim, hidden_dim)

        # 左门控层
        self.left_gate = nn.Linear(dim, hidden_dim)
        # 右门控层
        self.right_gate = nn.Linear(dim, hidden_dim)
        # 输出门控层
        self.out_gate = nn.Linear(dim, hidden_dim)

        # 初始化所有门控为恒等变换
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        # 根据混合类型设置混合的乘积表达式
        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        # 输出层归一化
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        # 输出层线性变换
        self.to_out = nn.Linear(hidden_dim, dim)

    # 前向传播函数
    def forward(self, x, mask = None):
        # 断言特征图必须是对称的
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        # 如果存在掩码，则重排掩码的维度
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        # 对输入进行归一化
        x = self.norm(x)

        # 左投影
        left = self.left_proj(x)
        # 右投影
        right = self.right_proj(x)

        # 如果存在掩码，则将投影结果与掩码相乘
        if exists(mask):
            left = left * mask
            right = right * mask

        # 计算左门控
        left_gate = self.left_gate(x).sigmoid()
        # 计算右门控
        right_gate = self.right_gate(x).sigmoid()
        # 计算输出门控
        out_gate = self.out_gate(x).sigmoid()

        # 左投影结果与左门控相乘
        left = left * left_gate
        # 右投影结果与右门控相乘
        right = right * right_gate

        # 执行乘积操作
        out = einsum(self.mix_einsum_eq, left, right)

        # 输出结果归一化
        out = self.to_out_norm(out)
        # 输出结果与输出门控相乘
        out = out * out_gate
        # 返回输出结果
        return self.to_out(out)
# 定义 OuterMean 类，用于计算两个输入的外积均值
class OuterMean(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim = None,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        hidden_dim = default(hidden_dim, dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d')

        if exists(mask):
            # 如果存在 mask，则进行 masked mean 操作，用于处理 MSA 中的填充
            mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(mask, 'b m j -> b m () j ()')
            outer = outer.masked_fill(~mask, 0.)
            outer = outer.mean(dim = 1) / (mask.sum(dim = 1) + self.eps)
        else:
            outer = outer.mean(dim = 1)

        return self.proj_out(outer)

# 定义 PairwiseAttentionBlock 类，用于计算两个输入的注意力
class PairwiseAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.,
        global_column_attn = False
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim)

        self.triangle_attention_outgoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.triangle_attention_ingoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True, global_query_attn = global_column_attn)
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim = dim, mix = 'outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim = dim, mix = 'ingoing')

    def forward(
        self,
        x,
        mask = None,
        msa_repr = None,
        msa_mask = None
    ):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr, mask = msa_mask)

        x = self.triangle_multiply_outgoing(x, mask = mask) + x
        x = self.triangle_multiply_ingoing(x, mask = mask) + x
        x = self.triangle_attention_outgoing(x, edges = x, mask = mask) + x
        x = self.triangle_attention_ingoing(x, edges = x, mask = mask) + x
        return x

# 定义 MsaAttentionBlock 类，用于计��� MSA 的注意力
class MsaAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.
    ):
        super().__init__()
        self.row_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.col_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True)

    def forward(
        self,
        x,
        mask = None,
        pairwise_repr = None
    ):
        x = self.row_attn(x, mask = mask, edges = pairwise_repr) + x
        x = self.col_attn(x, mask = mask) + x
        return x

# 定义 EvoformerBlock 类，包含 PairwiseAttentionBlock、FeedForward 和 MsaAttentionBlock
class EvoformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        global_column_attn = False
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            PairwiseAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, global_column_attn = global_column_attn),
            FeedForward(dim = dim, dropout = ff_dropout),
            MsaAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
            FeedForward(dim = dim, dropout = ff_dropout),
        ])

    def forward(self, inputs):
        x, m, mask, msa_mask = inputs
        attn, ff, msa_attn, msa_ff = self.layer

        # msa attention and transition
        m = msa_attn(m, mask = msa_mask, pairwise_repr = x)
        m = msa_ff(m) + m

        # pairwise attention and transition
        x = attn(x, mask = mask, msa_repr = m, msa_mask = msa_mask)
        x = ff(x) + x

        return x, m, mask, msa_mask

# 定义 Evoformer 类，包含多个 EvoformerBlock
class Evoformer(nn.Module):
    def __init__(
        self,
        *,
        depth,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([EvoformerBlock(**kwargs) for _ in range(depth)])

    def forward(
        self,
        x,
        m,
        mask = None,
        msa_mask = None
    ):
        inp = (x, m, mask, msa_mask)
        x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
        return x, m

# 定义 Alphafold2 类，包含各种模型参数和结构相关的参数
class Alphafold2(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_seq_len = 2048,
        depth = 6,
        heads = 8,
        dim_head = 64,
        max_rel_dist = 32,
        num_tokens = constants.NUM_AMINO_ACIDS,
        num_embedds = constants.NUM_EMBEDDS_TR,
        max_num_msas = constants.MAX_NUM_MSA,
        max_num_templates = constants.MAX_NUM_TEMPLATES,
        extra_msa_evoformer_layers = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        templates_dim = 32,
        templates_embed_layers = 4,
        templates_angles_feats_dim = 55,
        predict_angles = False,
        symmetrize_omega = False,
        predict_coords = False,                # structure module related keyword arguments below
        structure_module_depth = 4,
        structure_module_heads = 1,
        structure_module_dim_head = 4,
        disable_token_embed = False,
        mlm_mask_prob = 0.15,
        mlm_random_replace_token_prob = 0.1,
        mlm_keep_token_same_prob = 0.1,
        mlm_exclude_token_ids = (0,),
        recycling_distance_buckets = 32
    ):
        # 初始化函数，继承父类的初始化方法
        super().__init__()
        # 设置维度
        self.dim = dim

        # token embedding

        # 创建一个词嵌入层，用于将词索引映射为向量表示，如果禁用了词嵌入，则使用常数0
        self.token_emb = nn.Embedding(num_tokens + 1, dim) if not disable_token_embed else Always(0)
        # 线性层，用于将维度转换为双倍
        self.to_pairwise_repr = nn.Linear(dim, dim * 2)
        # 是否禁用了词嵌入
        self.disable_token_embed = disable_token_embed

        # positional embedding

        # 设置最大相对距离
        self.max_rel_dist = max_rel_dist
        # 创建一个位置嵌入层，用于将位置索引映射为向量表示
        self.pos_emb = nn.Embedding(max_rel_dist * 2 + 1, dim)

        # extra msa embedding

        # 创建一个额外的多序列比对嵌入模块
        self.extra_msa_evoformer = Evoformer(
            dim = dim,
            depth = extra_msa_evoformer_layers,
            seq_len = max_seq_len,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            global_column_attn = True
        )

        # template embedding

        # 线性层，用于将模板维度转换为指定维度
        self.to_template_embed = nn.Linear(templates_dim, dim)
        self.templates_embed_layers = templates_embed_layers

        # 模板对注意力块
        self.template_pairwise_embedder = PairwiseAttentionBlock(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            seq_len = max_seq_len
        )

        # 模板点注意力
        self.template_pointwise_attn = Attention(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            dropout = attn_dropout
        )

        # 模板角度MLP
        self.template_angle_mlp = nn.Sequential(
            nn.Linear(templates_angles_feats_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # projection for angles, if needed

        # 是否需要预测角度
        self.predict_angles = predict_angles
        self.symmetrize_omega = symmetrize_omega

        if predict_angles:
            # 线性层，用于将维度转换为角度桶的数量
            self.to_prob_theta = nn.Linear(dim, constants.THETA_BUCKETS)
            self.to_prob_phi   = nn.Linear(dim, constants.PHI_BUCKETS)
            self.to_prob_omega = nn.Linear(dim, constants.OMEGA_BUCKETS)

        # custom embedding projection

        # 自定义嵌入投影
        self.embedd_project = nn.Linear(num_embedds, dim)

        # main trunk modules

        # 主干模块
        self.net = Evoformer(
            dim = dim,
            depth = depth,
            seq_len = max_seq_len,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # MSA SSL MLM

        # 多序列比对自监督MLM
        self.mlm = MLM(
            dim = dim,
            num_tokens = num_tokens,
            mask_id = num_tokens, # 最后一个嵌入的标记用于掩码
            mask_prob = mlm_mask_prob,
            keep_token_same_prob = mlm_keep_token_same_prob,
            random_replace_token_prob = mlm_random_replace_token_prob,
            exclude_token_ids = mlm_exclude_token_ids
        )

        # calculate distogram logits

        # 计算距离图的logits
        self.to_distogram_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, constants.DISTOGRAM_BUCKETS)
        )

        # to coordinate output

        # 是否预测坐标
        self.predict_coords = predict_coords
        self.structure_module_depth = structure_module_depth

        self.msa_to_single_repr_dim = nn.Linear(dim, dim)
        self.trunk_to_pairwise_repr_dim = nn.Linear(dim, dim)

        with torch_default_dtype(torch.float32):
            # IPA块
            self.ipa_block = IPABlock(
                dim = dim,
                heads = structure_module_heads,
            )

            self.to_quaternion_update = nn.Linear(dim, 6)

        init_zero_(self.ipa_block.attn.to_out)

        self.to_points = nn.Linear(dim, 3)

        # aux confidence measure

        # 辅助置信度测量
        self.lddt_linear = nn.Linear(dim, 1)

        # recycling params

        # 回收参数
        self.recycling_msa_norm = nn.LayerNorm(dim)
        self.recycling_pairwise_norm = nn.LayerNorm(dim)
        self.recycling_distance_embed = nn.Embedding(recycling_distance_buckets, dim)
        self.recycling_distance_buckets = recycling_distance_buckets

    def forward(
        self,
        seq,
        msa = None,
        mask = None,
        msa_mask = None,
        extra_msa = None,
        extra_msa_mask = None,
        seq_index = None,
        seq_embed = None,
        msa_embed = None,
        templates_feats = None,
        templates_mask = None,
        templates_angles = None,
        embedds = None,
        recyclables = None,
        return_trunk = False,
        return_confidence = False,
        return_recyclables = False,
        return_aux_logits = False
```