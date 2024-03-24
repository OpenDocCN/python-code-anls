# `.\lucidrains\enformer-pytorch\enformer_pytorch\modeling_enformer.py`

```
# 导入所需的库
import math
from pathlib import Path

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from enformer_pytorch.data import str_to_one_hot, seq_indices_to_one_hot

from enformer_pytorch.config_enformer import EnformerConfig

from transformers import PreTrainedModel

# 定义常量
SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896

# 从 TensorFlow 中加载 gamma 位置
# 解决 TensorFlow 和 PyTorch 之间 xlogy 结果的差异
# 解决方案来自 @johahi
DIR = Path(__file__).parents[0]
TF_GAMMAS = torch.load(str(DIR / "precomputed"/ "tf_gammas.pt")

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回始终为指定值的函数
def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

# 对字典中的值应用函数
def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

# 在指数范围内生成整数序列
def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

# 计算对数，避免值过小
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 可能用于同步批归一化，在分布式训练中
def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

# 损失函数和指标

# Poisson 损失函数
def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

# 计算 Pearson 相关系数
def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

# 相对位置编码函数

# 获取指数衰减的位置特征
def get_positional_features_exponential(positions, features, seq_len, min_half_life = 3., dtype = torch.float):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device = positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)

# 获取中心掩码位置特征
def get_positional_features_central_mask(positions, features, seq_len, dtype = torch.float):
    center_widths = 2 ** torch.arange(1, features + 1, device = positions.device).to(dtype)
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).to(dtype)

# Gamma 分布概率密度函数
def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)

# 获取 Gamma 分布位置特征
def get_positional_features_gamma(positions, features, seq_len, stddev = None, start_mean = None, eps = 1e-8, dtype = torch.float):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device = positions.device)

    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2

    probabilities = gamma_pdf(positions.to(dtype).abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim = -1, keepdim = True)
    return outputs

# 获取位置嵌入
def get_positional_embed(seq_len, feature_size, device, use_tf_gamma, dtype = torch.float):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    assert not use_tf_gamma or seq_len == 1536, 'if using tf gamma, only sequence length of 1536 allowed for now'
    # 定义特征函数列表，包括指数特征、中心掩码特征和伽马特征（如果不使用 TensorFlow 伽马则使用 TF_GAMMAS）
    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma if not use_tf_gamma else always(TF_GAMMAS.to(device))
    ]

    # 计算特征组件的数量
    num_components = len(feature_functions) * 2

    # 检查特征大小是否能被组件数量整除
    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    # 计算每个类别的基础数量
    num_basis_per_class = feature_size // num_components

    # 初始化嵌入列表
    embeddings = []
    # 遍历特征函数列表，生成嵌入特征并添加到嵌入列表中
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len, dtype = dtype))

    # 在最后一个维度上连接所有嵌入特征
    embeddings = torch.cat(embeddings, dim = -1)
    # 在最后一个维度上连接嵌入特征和距离的符号乘积
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    # 将嵌入特征转换为指定数据类型并返回
    return embeddings.to(dtype)
def relative_shift(x):
    # 创建一个与 x 的最后一个维度大小相同的全零张量
    to_pad = torch.zeros_like(x[..., :1])
    # 在 x 的最后一个维度上连接全零张量，实现相对位移
    x = torch.cat((to_pad, x), dim=-1)
    # 获取 x 的形状信息
    _, h, t1, t2 = x.shape
    # 重新调整 x 的形状
    x = x.reshape(-1, h, t2, t1)
    # 从 x 中删除第一个元素
    x = x[:, :, 1:, :]
    # 重新调整 x 的形状
    x = x.reshape(-1, h, t1, t2 - 1)
    # 返回 x 的前一半元素
    return x[..., :((t2 + 1) // 2)]

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # 返回残差连接结果
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        # GELU 激活函数
        return torch.sigmoid(1.702 * x) * x

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        # 定义池化函数
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p=pool_size)

        # 定义注意力机制中的卷积层
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

        # 初始化卷积层的权重
        nn.init.dirac_(self.to_attn_logits.weight)

        # 对卷积层的权重进行缩放
        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            # 对输入进行填充
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)

        # 对输入进行池化操作
        x = self.pool_fn(x)
        # 计算注意力权重
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        # 计算加权和
        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]

def ConvBlock(dim, dim_out=None, kernel_size=1, is_distributed=None):
    batchnorm_klass = MaybeSyncBatchnorm(is_distributed=is_distributed)

    return nn.Sequential(
        batchnorm_klass(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=kernel_size // 2)
    )

# attention classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.,
        pos_dropout=0.,
        use_tf_gamma=False
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        # 线性变换得到查询、键、值
        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        # 输��层的线性变换
        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # 相对位置编码
        self.num_rel_pos_features = num_rel_pos_features
        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropout
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # 是否使用 tf gamma
        self.use_tf_gamma = use_tf_gamma
    # 定义前向传播函数，接受输入张量 x
    def forward(self, x):
        # 获取输入张量 x 的维度信息
        n, h, device = x.shape[-2], self.heads, x.device

        # 将输入张量 x 分别转换为查询（q）、键（k）、值（v）张量
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # 将查询（q）、键（k）、值（v）张量重排维度，以适应多头注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 对查询张量（q）进行缩放
        q = q * self.scale

        # 计算内容注意力得分
        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        # 获取位置嵌入向量
        positions = get_positional_embed(n, self.num_rel_pos_features, device, use_tf_gamma = self.use_tf_gamma, dtype = self.to_rel_k.weight.dtype)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        # 重排位置嵌入向量的维度，以适应多头注意力机制
        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h = h)
        # 计算相对位置注意力得分
        rel_logits = einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, rel_k)
        # 对相对位置注意力得分进行相对偏移
        rel_logits = relative_shift(rel_logits)

        # 组合内容注意力得分和相对位置注意力得分
        logits = content_logits + rel_logits
        # 对注意力得分进行 softmax 操作
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # 根据注意力权重计算输出张量
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 返回输出张量
        return self.to_out(out)
# 主类 Enformer 继承自 PreTrainedModel
class Enformer(PreTrainedModel):
    # 设置配置类和基础模型前缀
    config_class = EnformerConfig
    base_model_prefix = "enformer"

    # 从超参数创建 Enformer 实例的静态方法
    @staticmethod
    def from_hparams(**kwargs):
        return Enformer(EnformerConfig(**kwargs))

    # 初始化方法，接受配置参数
    def __init__(self, config):
        super().__init__(config)
        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2

        # 创建 stem 模块
        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding=7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size=2)
        )

        # 创建卷积 tower
        filter_list = exponential_linspace_int(half_dim, config.dim, num=(config.num_downsamples - 1), divisible_by=config.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size=2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        # 是否使用 tensorflow gamma 位置
        use_tf_gamma = config.use_tf_gamma
        self.use_tf_gamma = use_tf_gamma

        # transformer 模块
        transformer = []
        for _ in range(config.depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    Attention(
                        config.dim,
                        heads=config.heads,
                        dim_key=config.attn_dim_key,
                        dim_value=config.dim // config.heads,
                        dropout=config.attn_dropout,
                        pos_dropout=config.pos_dropout,
                        num_rel_pos_features=config.dim // config.heads,
                        use_tf_gamma=use_tf_gamma
                    ),
                    nn.Dropout(config.dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(*transformer)

        # 目标裁剪
        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

        # 最终的 pointwise 模块
        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(config.dropout_rate / 8),
            GELU()
        )

        # 创建 trunk 顺序模块
        self._trunk = nn.Sequential(
            Rearrange('b n d -> b d n'),
            self.stem,
            self.conv_tower,
            Rearrange('b d n -> b n d'),
            self.transformer,
            self.crop_final,
            self.final_pointwise
        )

        # 为人类和老鼠创建最终头部
        self.add_heads(**config.output_heads)

        # 在 transformer trunk 上使用检查点
        self.use_checkpointing = config.use_checkpointing

    # 添加头部方法
    def add_heads(self, **kwargs):
        self.output_heads = kwargs

        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim * 2, features),
            nn.Softplus()
        ), kwargs))

    # 设置目标长度的方法
    def set_target_length(self, target_length):
        crop_module = self._trunk[-2]
        crop_module.target_length = target_length

    # trunk 属性
    @property
    def trunk(self):
        return self._trunk

    @property
    # 返回当前对象的头部属性
    def heads(self):
        return self._heads

    # 对输入进行处理，返回经过处理后的结果
    def trunk_checkpointed(self, x):
        # 重新排列输入的数据维度
        x = rearrange(x, 'b n d -> b d n')
        # 对输入数据进行处理
        x = self.stem(x)
        x = self.conv_tower(x)
        x = rearrange(x, 'b d n -> b n d')
        # 使用序列化函数对输入数据进行处理
        x = checkpoint_sequential(self.transformer, len(self.transformer), x)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x

    # 对输入数据进行前向传播处理
    def forward(
        self,
        x,
        target = None,
        return_corr_coef = False,
        return_embeddings = False,
        return_only_embeddings = False,
        head = None,
        target_length = None
    ):
        # 如果输入数据是列表，则将其转换为独热编码
        if isinstance(x, list):
            x = str_to_one_hot(x)

        # 如果输入数据是 torch.Tensor 类型且数据类型为 long，则将其转换为独热编码
        elif type(x) == torch.Tensor and x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)
        # 将数据移动到指定设备上
        x.to(self.device)

        # 判断是否存在批次维度
        no_batch = x.ndim == 2

        # 如果没有批次维度，则重新排列数据维度
        if no_batch:
            x = rearrange(x, '... -> () ...')

        # 如果存在目标长度，则设置目标长度
        if exists(target_length):
            self.set_target_length(target_length)

        # 根据是否使用检查点技术选择相应的处理函数
        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        x = trunk_fn(x)

        # 如果没有批次维度，则重新排列数据维度
        if no_batch:
            x = rearrange(x, '() ... -> ...')

        # 如果只返回嵌入向量，则直接返回处理后的结果
        if return_only_embeddings:
            return x

        # 对处理后的结果进行映射处理
        out = map_values(lambda fn: fn(x), self._heads)

        # 如果指定了头部，则返回指定头部的结果
        if exists(head):
            assert head in self._heads, f'head {head} not found'
            out = out[head]

        # 如果存在目标数据，则计算损失
        if exists(target):
            assert exists(head), 'head must be passed in if one were to calculate loss directly with targets'

            # 如果需要返回相关系数，则返回相关系数
            if return_corr_coef:
                return pearson_corr_coef(out, target)

            # 返回泊松损失
            return poisson_loss(out, target)

        # 如果需要返回嵌入向量，则返回嵌入向量和处理后的结果
        if return_embeddings:
            return out, x

        # 返回处理后的结果
        return out
# 从预训练模型加载模型
def from_pretrained(name, use_tf_gamma = None, **kwargs):
    # 从预训练模型名称加载 Enformer 模型
    enformer = Enformer.from_pretrained(name, **kwargs)

    # 如果模型名称为 'EleutherAI/enformer-official-rough'
    if name == 'EleutherAI/enformer-official-rough':
        # 如果 use_tf_gamma 为 None，则设置为 True
        use_tf_gamma = default(use_tf_gamma, True)

        # 遍历 Enformer 模型的所有模块
        for module in enformer.modules():
            # 如果模块是 Attention 类型
            if isinstance(module, Attention):
                # 设置模块的 use_tf_gamma 属性为 use_tf_gamma
                module.use_tf_gamma = use_tf_gamma

    # 返回加载的 Enformer 模型
    return enformer
```