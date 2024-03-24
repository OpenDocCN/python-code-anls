# `.\lucidrains\tf-bind-transformer\tf_bind_transformer\tf_bind_transformer.py`

```
# 导入必要的库
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from functools import wraps

# 导入 einops 库中的函数
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# 导入 contextlib 库中的 contextmanager 函数
from contextlib import contextmanager

# 导入自定义的 Enformer 模型和相关函数
from enformer_pytorch import Enformer
from enformer_pytorch.modeling_enformer import poisson_loss, pearson_corr_coef
from enformer_pytorch.finetune import freeze_batchnorms_, freeze_all_but_layernorms_, unfreeze_last_n_layers_, unfreeze_all_layers_

# 导入 logavgexp 库中的函数
from logavgexp_pytorch import logavgexp

# 导入自定义的缓存函数和一些工具函数
from tf_bind_transformer.cache_utils import cache_fn
from tf_bind_transformer.protein_utils import get_protein_embedder
from tf_bind_transformer.context_utils import get_text_repr, get_contextual_dim

# 导入自定义的注意力机制相关类
from tf_bind_transformer.attention import FeedForward, JointCrossAttentionBlock, CrossAttention, SelfAttentionBlock

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回函数本身
def identity(fn, *args, **kwargs):
    return fn

# 空上下文管理器
@contextmanager
def null_context():
    yield

# 张量操作函数

# 对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 根据概率生成掩码
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# 对输入进行傅立叶编码
def fourier_encode(x, dims, theta = 20000):
    device, dtype = x.device, x.dtype
    emb = math.log(theta) / (dims // 2)
    emb = torch.exp(torch.arange(dims // 2, device = device) * -emb)
    emb = rearrange(x, 'n -> n 1') * rearrange(emb, 'd -> 1 d')
    emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
    return emb

# 计算相关系数损失
def corr_coef_loss(pred, target):
    return 1 - pearson_corr_coef(pred, target).mean()

# 缓存 Enformer 前向传播结果的装饰器

def cache_enformer_forward(fn):
    cached_forward = cache_fn(fn, clear = True, path = 'genetic')

    @wraps(fn)
    def inner(seqs, *args, **kwargs):
        if seqs.ndim == 3:
            seqs = seqs.argmax(dim = -1)

        seq_list = seqs.unbind(dim = 0)
        seq_cache_keys = [''.join(list(map(str, one_seq.tolist()))) for one_seq in seq_list]
        outputs = [cached_forward(one_seq, *args, __cache_key = seq_cache_key, **kwargs) for one_seq, seq_cache_key in zip(seq_list, seq_cache_keys)]
        return torch.stack(outputs)

    return inner

# 模型

# FiLM 模块
class FiLM(nn.Module):
    def __init__(
        self,
        dim,
        conditioned_dim
    ):
        super().__init__()
        self.to_gamma = nn.Linear(dim, conditioned_dim)
        self.to_bias = nn.Linear(dim, conditioned_dim)

    def forward(self, x, condition, mask = None):
        gamma = self.to_gamma(condition)
        bias = self.to_bias(condition)

        x = x * rearrange(gamma, 'b d -> b 1 d')
        x = x + rearrange(bias, 'b d -> b 1 d')
        return x

# SqueezeExcitation 模块
class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        dim,
        conditioned_dim,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.to_gate = nn.Linear(dim + conditioned_dim, conditioned_dim)

    def forward(self, x, condition, mask = None):
        if exists(mask):
            numer = x.masked_fill(mask[..., None], 0.).sum(dim = 1)
            denom = mask.sum(dim = 1)[..., None].clamp(min = self.eps)
            mean_x = numer / denom
        else:
            mean_x = x.mean(dim = 1)

        condition = torch.cat((condition, mean_x), dim = -1)
        gate = self.to_gate(condition)

        x = x * rearrange(gate, 'b d -> b 1 d').sigmoid()
        return x

# 用于计算辅助损失的 ReadValueMLP 类
class ReadValueMLP(nn.Module):
    def __init__(
        self,
        dim,
        *,
        fourier_dims = 256,
        norm_factor_fourier = 50,
        norm_factor_linear = 8000,
        eps = 1e-20
    # 初始化函数，设置模型参数
    def __init__(
        self,
        eps,
        fourier_dims,
        norm_factor_fourier,
        norm_factor_linear
    ):
        # 调用父类初始化函数
        super().__init__()
        # 设置模型参数
        self.eps = eps
        self.fourier_dims = fourier_dims
        self.norm_factor_fourier = norm_factor_fourier
        self.norm_factor_linear = norm_factor_linear

        # 定义 logits 的归一化层
        self.logits_norm = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),  # 对 logits 进行平均池化
            nn.LayerNorm(dim)  # 对结果进行 LayerNorm
        )

        # 定义 MLP 网络
        self.mlp = nn.Sequential(
            nn.Linear(dim + fourier_dims + 2, dim * 2),  # 线性层
            nn.GELU(),  # GELU 激活函数
            nn.Linear(dim * 2, 1),  # 线性层
            Rearrange('... 1 -> ...')  # 重新排列维度
        )

    # 前向传播函数
    def forward(self, logits, peaks_nr, read_value):
        # 对 logits 进行归一化
        logits = self.logits_norm(logits)

        # 对 peaks_nr 进行对数变换
        peaks_nr_log_space = torch.log(peaks_nr + self.eps)

        # 重新排列 peaks_nr 的维度
        peaks_nr = rearrange(peaks_nr, '... -> (...)')
        # 对 peaks_nr 进行傅立叶编码
        peaks_nr_encoded = fourier_encode(peaks_nr / self.norm_factor_fourier, self.fourier_dims)
        # 对 peaks_nr 进行归一化
        peaks_nr_normed = rearrange(peaks_nr, '... -> ... 1') / self.norm_factor_linear

        # 将 peaks_nr_normed、peaks_nr_log_space、peaks_nr_encoded 拼接在一起
        peaks_nr_encoded_with_self = torch.cat((peaks_nr_normed, peaks_nr_log_space, peaks_nr_encoded), dim = -1)

        # 将 logits 和 peaks_nr_encoded_with_self 拼接在一起
        logits_with_peaks = torch.cat((logits, peaks_nr_encoded_with_self), dim = -1)

        # 通过 MLP 网络得到预测值
        pred = self.mlp(logits_with_peaks)
        # 重新排列 read_value 的维度
        read_value = rearrange(read_value, '... -> (...)')

        # 返回 Smooth L1 损失
        return F.smooth_l1_loss(pred, read_value)
# 定义一个名为 HypergridLinear 的类，继承自 nn.Module
class HypergridLinear(nn.Module):
    # 初始化函数，接受输入维度 dim、输出维度 dim_out 和上下文维度 context_dim
    def __init__(
        self,
        dim,
        dim_out,
        *,
        context_dim
    ):
        super().__init__()
        # 定义权重参数，使用随机初始化
        self.weights = nn.Parameter(torch.randn(dim, dim_out))
        # 定义上下文投影层，使用线性变换
        self.contextual_projection = nn.Linear(context_dim, dim * dim_out)

    # 前向传播函数，接受输入 x 和上下文 context
    def forward(self, x, context):
        # 推导上下文门控，参考超网格论文
        gating = self.contextual_projection(context).sigmoid()
        gating = rearrange(gating, 'b (i o) -> b i o', i = int(math.sqrt(gating.shape[-1])))
        
        # 门控交互投影与上下文
        to_logits_w = rearrange(self.weights, 'i o -> 1 i o') * gating
        return einsum('b n d, b d e -> b n e', x, to_logits_w)

# 定义一个名为 FILIP 的类，继承自 nn.Module
class FILIP(nn.Module):
    # 初始化函数，接受输入维度 dim、上下文维度 context_dim、头数 heads、头维度 dim_head、dropout 概率
    def __init__(
        self,
        dim,
        context_dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        inner_latent_dim = heads * dim_head

        # 定义转换到潜在空间的权重和偏置
        self.to_latent_w = nn.Parameter(torch.randn(dim, inner_latent_dim))
        self.to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

        self.pre_attn_dropout = dropout

        # 定义空上下文和上下文到潜在空间的权重和偏置
        self.null_context = nn.Parameter(torch.randn(heads, dim_head))
        self.context_to_latent_w = nn.Parameter(torch.randn(context_dim, inner_latent_dim))
        self.context_to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

    # 前向传播函数，接受输入 x、上下文 context 和上下文掩码 context_mask
    def forward(
        self,
        x,
        context,
        context_mask = None
    ):
        b, heads, device = x.shape[0], self.heads, x.device

        x = einsum('b n d, d e -> b n e', x, self.to_latent_w)
        x = x + self.to_latent_b

        x = rearrange(x, 'b n (h d) -> b h n d', h = heads)

        context = einsum('b n d, d e -> b n e', context, self.context_to_latent_w)
        context = context + self.context_to_latent_b

        context = rearrange(context, 'b n (h d) -> b h n d', h = heads)

        context, x = map(l2norm, (context, x))

        # DNA 和蛋白质序列之间的细粒度交互，参考 FILIP 论文
        if x.shape[0] == 1:
            x = rearrange(x, '1 ... -> ...')
            einsum_eq = 'h i d, b h j d -> b h i j'
        else:
            einsum_eq = 'b h i d, b h j d -> b h i j'

        # 如果上下文掩码不存在，则创建一个全为 True 的掩码
        if not exists(context_mask):
            context_mask = torch.ones((b, context.shape[-1]), device = device).bool()

        # 根据 dropout 概率生成掩码
        if self.training:
            keep_mask = prob_mask_like(context_mask, 1 - self.pre_attn_dropout)
            context_mask = context_mask & keep_mask

        # 添加空上下文并修改掩码
        context_mask = F.pad(context_mask, (1, 0), value = True)
        context_mask = rearrange(context_mask, 'b j -> b 1 1 j')

        null_context = repeat(self.null_context, 'h d -> b h 1 d', b = b)
        context = torch.cat((null_context, context), dim = -2)

        # 可微分最大化，参考 FILIP 论文
        interactions = einsum(einsum_eq, x, context)
        interactions = logavgexp(interactions, mask = context_mask, dim = -1, temp = 0.05)
        interactions = rearrange(interactions, 'b h i -> b i h')
        return interactions

# 定义一个名为 AdapterModel 的类，继承自 nn.Module
class AdapterModel(nn.Module):
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        *,
        enformer,  # enformer 模型
        latent_dim = 64,  # 潜在维度，默认为 64
        latent_heads = 32,  # 潜在头数，默认为 32
        aa_embed_dim = None,  # 氨基酸嵌入维度，默认为 None
        aa_embed_encoder = 'esm',  # 氨基酸嵌入编码器，默认为 'esm'
        contextual_embed_dim = None,  # 上下文嵌入维度，默认为 None
        use_aa_embeds = False,  # 是否使用氨基酸嵌入，默认为 False
        use_free_text_context = False,  # 是否使用自由文本上下文，默认为 False
        free_text_context_encoder = 'pubmed',  # 自由文本上下文编码器，默认为 'pubmed'
        free_text_embed_method = 'cls',  # 自由文本嵌入方法，默认为 'cls'
        dropout = 0.,  # 丢弃率，默认为 0
        binary_target = False,  # 是否为二进制目标，默认为 False
        target_mse_loss = False,  # 是否使用均方误差损失，默认为 False
        aux_read_value_loss = False,  # 是否使用辅助读值损失，默认为 False
        read_value_aux_loss_weight = 0.05,  # 读值辅助损失权重，默认为 0.05
        joint_cross_attn_depth = 1,  # 联合交叉注意力深度，默认为 1
        genome_self_attn_depth = 0,  # 基因组自注意力深度，默认为 0
        fourier_dims = 256,  # 傅立叶维度，默认为 256
        condition_squeeze_excite = False,  # 是否条件挤压激活，默认为 False
        condition_film = False,  # 是否条件 FILM，默认为 False
        condition_hypergrid = True,  # 是否条件超网格，默认为 True
        use_corr_coef_loss = False,  # 是否使用相关系数损失，默认为 False
        finetune_output_heads = None,  # 微调输出头，默认为 None
        **kwargs  # 其他参数
        ):
            # 调用父类的构造函数
            super().__init__()
            # 断言 enformer 是 Enformer 的实例
            assert isinstance(enformer, Enformer), 'enformer must be an instance of Enformer'
            # 设置 self.enformer 为传入的 enformer
            self.enformer = enformer
            # 计算 enformer_dim 为 enformer.dim 的两倍
            enformer_dim = enformer.dim * 2

            # 如果 finetune_output_heads 存在，则为 enformer 添加头部
            if exists(finetune_output_heads):
                self.enformer.add_heads(**finetune_output_heads)

            # 初始化 norm_seq_embed 为 LayerNorm 层，输入维度为 enformer_dim
            self.norm_seq_embed = nn.LayerNorm(enformer_dim)

            # 上下文嵌入相关变量

            # 断言 free_text_embed_method 只能是 'cls' 或 'mean_pool'
            assert free_text_embed_method in {'cls', 'mean_pool'}, 'must be either cls or mean_pool'
            # 设置 self.free_text_embed_method 为传入的 free_text_embed_method
            self.free_text_embed_method = free_text_embed_method
            # 设置 self.use_free_text_context 为传入的 use_free_text_context

            if use_free_text_context:
                # 如果使用自由文本上下文，则计算上下文嵌入维度
                contextual_embed_dim = get_contextual_dim(free_text_context_encoder)
            else:
                # 否则，断言必须给出上下文嵌入维度
                assert exists(contextual_embed_dim), 'contextual embedding dimension must be given if not using transformer encoder'

            # 蛋白质嵌入相关变量

            # 设置 self.use_aa_embeds 为传入的 use_aa_embeds
            self.use_aa_embeds = use_aa_embeds
            # 获取蛋白质嵌入器的配置
            self.aa_embed_config = get_protein_embedder(aa_embed_encoder)
            # 获取蛋白质嵌入函数
            self.get_aa_embed = self.aa_embed_config['fn']

            if use_aa_embeds:
                # 如果使用蛋白质嵌入，则设置 aa_embed_dim 为蛋白质嵌入维度
                aa_embed_dim = self.aa_embed_config['dim']
            else:
                # 否则，断言必须设置 AA 嵌入维度
                assert exists(aa_embed_dim), 'AA embedding dimensions must be set if not using ESM'

            # 条件

            self.cond_genetic = None
            self.cond_protein = None

            if condition_squeeze_excite or condition_film:
                # 根据条件选择 SqueezeExcitation 或 FiLM 类
                condition_klass = SqueezeExcitation if condition_squeeze_excite else FiLM

                # 如果需要条件激活，则为 genetic 和 protein 设置条件
                self.cond_genetic  = condition_klass(contextual_embed_dim, enformer_dim)
                self.cond_protein  = condition_klass(contextual_embed_dim, aa_embed_dim)

            # 基因组自注意力

            # 初始化 genome_self_attns 为空的 ModuleList

            for _ in range(genome_self_attn_depth):
                # 循环创建 SelfAttentionBlock，并添加到 genome_self_attns 中
                attn = SelfAttentionBlock(
                    dim = enformer_dim,
                    dropout = dropout
                )
                self.genome_self_attns.append(attn)

            # 联合注意力

            # 初始化 joint_cross_attns 为空的 ModuleList

            for _ in range(joint_cross_attn_depth):
                # 循环创建 JointCrossAttentionBlock，并添加到 joint_cross_attns 中
                attn = JointCrossAttentionBlock(
                    dim = enformer_dim,
                    context_dim = aa_embed_dim,
                    dropout = dropout
                )

                self.joint_cross_attns.append(attn)

            # 潜变量

            # 初始化 filip 为 FILIP 模块
            self.filip = FILIP(
                dim = enformer_dim,
                context_dim = aa_embed_dim,
                dim_head = latent_dim,
                heads = latent_heads,
                dropout = dropout
            )

            # 超网格条件

            if condition_hypergrid:
                # 如果需要超网格条件，则初始化 linear_with_hypergrid 为 HypergridLinear
                self.linear_with_hypergrid = HypergridLinear(latent_heads, latent_heads, context_dim = contextual_embed_dim)
            else:
                # 否则，初始化 linear_to_logits 为 Linear 层
                self.linear_to_logits = nn.Linear(latent_heads, latent_heads)

            # 到预测

            # 设置 binary_target 和 aux_read_value_loss 为传入的值
            self.binary_target = binary_target
            self.aux_read_value_loss = aux_read_value_loss
            self.read_value_aux_loss_weight = read_value_aux_loss_weight

            if binary_target:
                # 如果是二进制目标，则设置损失函数为二进制交叉熵或均方误差
                self.loss_fn = F.binary_cross_entropy_with_logits if not target_mse_loss else F.mse_loss

                # 设置 to_pred 为 Sequential 模块，用于预测
                self.to_pred = nn.Sequential(
                    Reduce('... n d -> ... d', 'mean'),
                    nn.LayerNorm(latent_heads),
                    nn.Linear(latent_heads, 1),
                    Rearrange('... 1 -> ...')
                )

                # 设置 to_read_value_aux_loss 为 ReadValueMLP 模块
                self.to_read_value_aux_loss = ReadValueMLP(
                    dim = latent_heads,
                    fourier_dims = fourier_dims
                )

            else:
                # 如果不是二进制目标，则设置损失函数为泊松损失或相关系数损失
                self.loss_fn = poisson_loss if not use_corr_coef_loss else corr_coef_loss

                # 设置 to_pred 为 Sequential 模块，用于预测
                self.to_pred = nn.Sequential(
                    nn.Linear(latent_heads, 1),
                    Rearrange('... 1 -> ...'),
                    nn.Softplus()
                )
    # 合并主要损失和辅助损失，如果不需要辅助损失则返回主要损失
    def combine_losses(self, loss, aux_loss):
        if not self.aux_read_value_loss:
            return loss

        return loss + self.read_value_aux_loss_weight * aux_loss

    # 前向传播函数，用于处理 Enformer 模型的头部
    def forward_enformer_head(
        self,
        seq_embed,
        *,
        head,
        target = None,
        return_corr_coef = False
    ):
        # 检查是否开启二进制目标训练，如果是则无法在轨道上微调
        assert not self.binary_target, 'cannot finetune on tracks if binary_target training is turned on'

        # 解冻 Enformer 模型的所有层
        unfreeze_all_layers_(self.enformer._heads)

        # 检查指定的头部是否存在于 Enformer 模型中
        assert head in self.enformer._heads, f'{head} head not found in enformer'

        # 使用指定的头部对序列嵌入进行预测
        pred = self.enformer._heads[head](seq_embed)

        # 如果没有提供目标数据，则直接返回预测结果
        if not exists(target):
            return pred

        # 检查预测结果和目标数据的维度是否匹配
        assert pred.shape[-1] == target.shape[-1], f'{head} head on enformer produced {pred.shape[-1]} tracks, but the supplied target only has {target.shape[-1]}'

        # 如果提供了目标数据并且需要返回相关系数，则计算并返回相关系数
        if exists(target) and return_corr_coef:
            return pearson_corr_coef(pred, target)

        # 计算并返回损失函数的结果
        return self.loss_fn(pred, target)

    # 前向传播函数，用于处理多个输入和参数的情况
    def forward(
        self,
        seq,
        *,
        aa = None,
        aa_embed = None,
        contextual_embed = None,
        contextual_free_text = None,
        aa_mask = None,
        target = None,
        read_value = None,
        peaks_nr = None,
        return_corr_coef = False,
        finetune_enformer = False,
        finetune_enformer_ln_only = False,
        unfreeze_enformer_last_n_layers = 0,
        head = None
```