# `.\lucidrains\enformer-pytorch\enformer_pytorch\finetune.py`

```
# 导入 torch 库
import torch
# 导入类型提示 Optional
from typing import Optional

# 从 copy 模块中导入 deepcopy 函数
from copy import deepcopy
# 从 contextlib 模块中导入 contextmanager 装饰器
from contextlib import contextmanager
# 从 torch.nn.functional 模块中导入 F 别名
import torch.nn.functional as F
# 从 torch 模块中导入 nn、einsum
from torch import nn, einsum

# 从 einops 模块中导入 rearrange、repeat
from einops import rearrange, repeat
# 从 einops.layers.torch 模块中导入 Rearrange 类
from einops.layers.torch import Rearrange
# 从 enformer_pytorch.modeling_enformer 模块中导入 Enformer、poisson_loss 函数
from enformer_pytorch.modeling_enformer import Enformer, poisson_loss

# 从 discrete_key_value_bottleneck_pytorch 模块中导入 DiscreteKeyValueBottleneck 类

# 定义 exists 函数，判断变量是否存在
def exists(val):
    return val is not None

# 定义 default 函数，如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义 null_context 上下文管理器
@contextmanager
def null_context():
    yield

# 定义 better sequential 函数，返回过滤掉不存在的模块的 nn.Sequential 对象
def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# 控制层的冻结

# 设置模块的 requires_grad 属性
def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

# 冻结所有层
def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

# 解冻所有层
def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

# 冻结批归一化层
def freeze_batchnorms_(model):
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]

    for bn in bns:
        bn.eval()
        bn.track_running_stats = False
        set_module_requires_grad_(bn, False)

# 冻结除了层归一化层之外的所有层
def freeze_all_but_layernorms_(model):
    for m in model.modules():
        set_module_requires_grad_(m, isinstance(m, nn.LayerNorm))

# 冻结除了最后 N 层之外的所有层
def freeze_all_but_last_n_layers_(enformer, n):
    assert isinstance(enformer, Enformer)
    freeze_all_layers_(enformer)

    transformer_blocks = enformer.transformer

    for module in transformer_blocks[-n:]:
        set_module_requires_grad_(module, True)

# 获取 Enformer 的嵌入

def get_enformer_embeddings(
    model,
    seq,
    freeze = False,
    train_layernorms_only = False,
    train_last_n_layers_only = None,
    enformer_kwargs: dict = {}
):
    freeze_batchnorms_(model)

    if train_layernorms_only:
        assert not freeze, 'you set the intent to train the layernorms of the enformer, yet also indicated you wanted to freeze the entire model'
        freeze_all_but_layernorms_(model)

    if exists(train_last_n_layers_only):
        assert not freeze, 'you set the intent to train last N layers of enformer, but also indicated you wanted to freeze the entire network'
        freeze_all_but_last_n_layers_(model, train_last_n_layers_only)

    enformer_context = null_context() if not freeze else torch.no_grad()

    with enformer_context:
        embeddings = model(seq, return_only_embeddings = True, **enformer_kwargs)

        if freeze:
            embeddings.detach_()

    return embeddings

# 微调包装类

# 额外头部投影，类似于人类和老鼠轨迹的训练方式

class HeadAdapterWrapper(nn.Module):
    def __init__(
        self,
        *,
        enformer,
        num_tracks,
        post_transformer_embed = False, # 是否从变换器后面的嵌入中获取嵌入，而不是在最终的逐点卷积之后获取 - 这将添加另一个层归一化
        discrete_key_value_bottleneck = False,
        bottleneck_num_memories = 256,
        bottleneck_num_codebooks = 4,
        bottleneck_decay = 0.9,
        transformer_embed_fn: nn.Module = nn.Identity(),
        output_activation: Optional[nn.Module] = nn.Softplus(),
        auto_set_target_length = True
        ):
        # 调用父类的构造函数
        super().__init__()
        # 断言 enformer 是 Enformer 类的实例
        assert isinstance(enformer, Enformer)
        # 计算 enformer_hidden_dim，如果 post_transformer_embed 为 False，则乘以 2
        enformer_hidden_dim = enformer.dim * (2 if not post_transformer_embed else 1)

        # 设置离散键值瓶颈的标志
        self.discrete_key_value_bottleneck = discrete_key_value_bottleneck

        # 如果启用了离散键值瓶颈
        if discrete_key_value_bottleneck:
            # 创建 DiscreteKeyValueBottleneck 对象
            enformer = DiscreteKeyValueBottleneck(
                encoder = enformer,
                dim = enformer_hidden_dim,
                num_memory_codebooks = bottleneck_num_codebooks,
                num_memories = bottleneck_num_memories,
                dim_memory = enformer_hidden_dim // bottleneck_num_codebooks,
                decay = bottleneck_decay,
            )

        # 设置 post_transformer_embed 标志
        self.post_transformer_embed = post_transformer_embed

        # 设置 enformer 属性
        self.enformer = enformer

        # 设置 auto_set_target_length 标志
        self.auto_set_target_length = auto_set_target_length

        # 如果启用了 post_transformer_embed
        if post_transformer_embed:
            # 深拷贝 enformer 对象
            self.enformer = deepcopy(enformer)
            # 将 enformer 的最后一层设置为 nn.Identity()
            self.enformer._trunk[-1] = nn.Identity()
            # 将 enformer 的 final_pointwise 层设置为 nn.Identity()
            self.enformer.final_pointwise = nn.Identity()

        # 设置 post_embed_transform 属性
        self.post_embed_transform = Sequential(
            transformer_embed_fn,
            nn.LayerNorm(enformer_hidden_dim) if post_transformer_embed else None
        )

        # 设置 to_tracks 属性
        self.to_tracks = Sequential(
            nn.Linear(enformer_hidden_dim, num_tracks),
            output_activation
        )

    # 定义前向传播函数
    def forward(
        self,
        seq,
        *,
        target = None,
        freeze_enformer = False,
        finetune_enformer_ln_only = False,
        finetune_last_n_layers_only = None
    ):
        # 初始化 enformer_kwargs 字典
        enformer_kwargs = dict()

        # 如果存在目标数据并且 auto_set_target_length 为 True
        if exists(target) and self.auto_set_target_length:
            # 设置 enformer_kwargs 中的 target_length 键值对
            enformer_kwargs = dict(target_length = target.shape[-2])

        # 如果启用了离散键值瓶颈
        if self.discrete_key_value_bottleneck:
            # 获取 enformer 的 embeddings
            embeddings = self.enformer(seq, return_only_embeddings = True, **enformer_kwargs)
        else:
            # 获取 enformer 的 embeddings
            embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer, train_layernorms_only = finetune_enformer_ln_only, train_last_n_layers_only = finetune_last_n_layers_only, enformer_kwargs = enformer_kwargs)

        # 将 embeddings 转换为预测结果
        preds = self.to_tracks(embeddings)

        # 如果不存在目标数据，则返回预测结果
        if not exists(target):
            return preds

        # 计算 Poisson 损失并返回结果
        return poisson_loss(preds, target)
# 定义一个包装器，允许为每个轨道提供上下文维度
# 上下文嵌入将投影到头线性投影（超网络）的权重和偏置中

class ContextAdapterWrapper(nn.Module):
    def __init__(
        self,
        *,
        enformer,  # Enformer 模型
        context_dim,  # 上下文维度
        discrete_key_value_bottleneck = False,  # 是否使用离散键值瓶颈
        bottleneck_num_memories = 256,  # 瓶颈内存数量
        bottleneck_num_codebooks = 4,  # 瓶颈码书数量
        bottleneck_decay = 0.9,  # 瓶颈衰减率
        auto_set_target_length = True,  # 是否自动设置目标长度
        output_activation: Optional[nn.Module] = nn.Softplus()  # 输出激活函数，默认为 Softplus
    ):
        super().__init__()
        assert isinstance(enformer, Enformer)
        enformer_hidden_dim = enformer.dim * 2

        self.discrete_key_value_bottleneck = discrete_key_value_bottleneck

        if discrete_key_value_bottleneck:
            enformer = DiscreteKeyValueBottleneck(
                encoder = enformer,
                dim = enformer_hidden_dim,
                num_memory_codebooks = bottleneck_num_codebooks,
                num_memories = bottleneck_num_memories,
                dim_memory = enformer_hidden_dim // bottleneck_num_codebooks,
                decay = bottleneck_decay,
            )

        self.enformer = enformer

        self.auto_set_target_length = auto_set_target_length

        self.to_context_weights = nn.Parameter(torch.randn(context_dim, enformer_hidden_dim))  # 上下文权重参数
        self.to_context_bias = nn.Parameter(torch.randn(context_dim))  # 上下文偏置参数

        self.activation = default(output_activation, nn.Identity())  # 激活函数

    def forward(
        self,
        seq,  # 输入序列
        *,
        context,  # 上下文
        target = None,  # 目标
        freeze_enformer = False,  # 是否冻结 Enformer
        finetune_enformer_ln_only = False,  # 是否仅微调 Enformer 层归一化
        finetune_last_n_layers_only = None  # 仅微调最后 n 层
    ):
        enformer_kwargs = dict()

        if exists(target) and self.auto_set_target_length:
            enformer_kwargs = dict(target_length = target.shape[-2])

        if self.discrete_key_value_bottleneck:
            embeddings = self.enformer(seq, return_only_embeddings = True, **enformer_kwargs)
        else:
            embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer, train_layernorms_only = finetune_enformer_ln_only, train_last_n_layers_only = finetune_last_n_layers_only, enformer_kwargs = enformer_kwargs)

        weights = einsum('t d, d e -> t e', context, self.to_context_weights)  # 计算权重
        bias = einsum('t d, d -> t', context, self.to_context_bias)  # 计算偏置

        pred = einsum('b n d, t d -> b n t', embeddings, weights) + bias  # 预测结果

        pred = self.activation(pred)  # 应用激活函数

        if not exists(target):
            return pred

        return poisson_loss(pred, target)  # 返回 Poisson 损失

# 包装器，执行上下文的注意力聚合，上下文可以是一个标记列表（批次 x 序列 x 维度）

class ContextAttentionAdapterWrapper(nn.Module):
    def __init__(
        self,
        *,
        enformer,  # Enformer 模型
        context_dim,  # 上下文维度
        heads = 8,  # 头数
        dim_head = 64,  # 每个头的维度
        discrete_key_value_bottleneck = False,  # 是否使用离散键值瓶颈
        bottleneck_num_memories = 256,  # 瓶颈内存数量
        bottleneck_num_codebooks = 4,  # 瓶颈码书数量
        bottleneck_decay = 0.9,  # 瓶颈衰减率
        auto_set_target_length = True,  # 是否自动设置目标长度
        output_activation: Optional[nn.Module] = nn.Softplus()  # 输出激活函数，默认为 Softplus
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言 enformer 是 Enformer 类的实例
        assert isinstance(enformer, Enformer)
        # 计算 enformer 隐藏维度
        enformer_hidden_dim = enformer.dim * 2

        # 设置离散键值瓶颈
        self.discrete_key_value_bottleneck = discrete_key_value_bottleneck

        # 如果启用了离散键值瓶颈
        if discrete_key_value_bottleneck:
            # 创建 DiscreteKeyValueBottleneck 对象
            enformer = DiscreteKeyValueBottleneck(
                encoder = enformer,
                dim = enformer_hidden_dim,
                num_memory_codebooks = bottleneck_num_codebooks,
                num_memories = bottleneck_num_memories,
                dim_memory = enformer_hidden_dim // bottleneck_num_codebooks,
                decay = bottleneck_decay,
            )

        # 设置 enformer
        self.enformer = enformer

        # 设置是否自动设置目标长度
        self.auto_set_target_length = auto_set_target_length

        # 对查询进行归一化
        self.query_norm = nn.LayerNorm(enformer_hidden_dim)
        # 对键值进行归一化
        self.key_values_norm = nn.LayerNorm(context_dim)

        # 设置缩放因子和头数
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head
        # 线性变换生成查询
        self.to_queries = nn.Linear(enformer_hidden_dim, inner_dim, bias = False)

        # 初始化空键和空值
        self.null_key = nn.Parameter(torch.randn(inner_dim))
        self.null_value = nn.Parameter(torch.randn(inner_dim))

        # 线性变换生成键值
        self.to_key_values = nn.Linear(context_dim, inner_dim * 2, bias = False)
        # 线性变换生成输出
        self.to_out = nn.Linear(inner_dim, enformer_hidden_dim)

        # 线性变换生成预测结果
        self.to_pred  = Sequential(
            nn.Linear(enformer_hidden_dim, 1),
            Rearrange('b c ... 1 -> b ... c'),
            output_activation
        )

    # 前向传播函数
    def forward(
        self,
        seq,
        *,
        context,
        context_mask = None,
        target = None,
        freeze_enformer = False,
        finetune_enformer_ln_only = False,
        finetune_last_n_layers_only = None
        ):
        """
        b - batch
        n - sequence length
        c - number of contexts (tracks)
        d - dimension
        i - sequence length (query embeddings)
        j - sequence length (keys / values contexts)
        h - attention heads
        """

        # 设置变量 h 为 self.heads

        enformer_kwargs = dict()

        # 如果 target 存在且 self.auto_set_target_length 为真，则设置 enformer_kwargs 的 target_length 为 target 的倒数第二维度长度
        if exists(target) and self.auto_set_target_length:
            enformer_kwargs = dict(target_length = target.shape[-2])

        # 如果 self.discrete_key_value_bottleneck 为真，则调用 self.enformer 方法获取 embeddings
        # 否则调用 get_enformer_embeddings 方法获取 embeddings
        if self.discrete_key_value_bottleneck:
            embeddings = self.enformer(seq, return_only_embeddings = True, **enformer_kwargs)
        else:
            embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer, train_layernorms_only = finetune_enformer_ln_only, train_last_n_layers_only = finetune_last_n_layers_only, enformer_kwargs = enformer_kwargs)

        # 从 genetic 到 context 执行交叉注意力

        # 如果 context 的维度为 2，则将其重排为 'b d -> b 1 d'
        if context.ndim == 2:
            context = rearrange(context, 'b d -> b 1 d')

        # 获取查询 q，键 k 和值 v
        q = self.to_queries(self.query_norm(embeddings))
        k, v = self.to_key_values(self.key_values_norm(context)).chunk(2, dim = -1)

        # 创建 null_k 和 null_v，并将其重复到与 k 和 v 相同的维度
        null_k, null_v = map(lambda t: repeat(t, 'd -> b 1 d', b = context.shape[0]), (self.null_key, self.null_value))

        # 将 null_k 和 k 连接在一起，将 null_v 和 v 连接在一起
        k = torch.cat((null_k, k), dim = 1)
        v = torch.cat((null_v, v), dim = 1)

        # 分离头部
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, c h j d -> b c h i j', q, k) * self.scale

        # 掩码
        if exists(context_mask):
            context_mask = F.pad(context_mask, (1, 0), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 1 j')
            sim = sim.masked_fill(~context_mask, -torch.finfo(sim.dtype).max)

        # 注意力
        attn = sim.softmax(dim = -1)

        # 聚合
        out = einsum('b c h i j, c h j d -> b c h i d', attn, v)
        out = rearrange(out, 'b c h n d -> b c n (h d)', h = h)

        # 合并头部
        branch_out = self.to_out(out)

        # 残差连接
        embeddings = embeddings + branch_out

        # 转换为预测
        pred = self.to_pred(embeddings)

        # 如果 target 不存在，则返回 pred，否则返回 poisson_loss(pred, target)
        if not exists(target):
            return pred

        return poisson_loss(pred, target)
```