# `.\lucidrains\protein-bert-pytorch\protein_bert_pytorch\protein_bert_pytorch.py`

```
# 导入 math、torch 库以及 torch.nn.functional 模块中的 F 函数
import math
import torch
import torch.nn.functional as F
# 从 torch 模块中导入 nn、einsum 函数
from torch import nn, einsum
# 从 einops.layers.torch 模块中导入 Rearrange、Reduce 类
from einops.layers.torch import Rearrange, Reduce
# 从 einops 模块中导入 rearrange、repeat 函数

# helpers

# 判断变量是否存在的辅助函数
def exists(val):
    return val is not None

# 返回给定张量类型的最小负值的辅助函数
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# helper classes

# 残差连接类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# 全局线性自注意力类
class GlobalLinearSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head,
        heads
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, feats, mask = None):
        h = self.heads
        q, k, v = self.to_qkv(feats).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n ()')
            k = k.masked_fill(~mask, -torch.finfo(k.dtype).max)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        if exists(mask):
            v = v.masked_fill(~mask, 0.)

        context = einsum('b h n d, b h n e -> b h d e', k, v)
        out = einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 交叉注意力类
class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_keys,
        dim_out,
        heads,
        dim_head = 64,
        qk_activation = nn.Tanh()
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.qk_activation = qk_activation

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_keys, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim_out)

        self.null_key = nn.Parameter(torch.randn(dim_head))
        self.null_value = nn.Parameter(torch.randn(dim_head))

    def forward(self, x, context, mask = None, context_mask = None):
        b, h, device = x.shape[0], self.heads, x.device

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        null_k, null_v = map(lambda t: repeat(t, 'd -> b h () d', b = b, h = h), (self.null_key, self.null_value))
        k = torch.cat((null_k, k), dim = -2)
        v = torch.cat((null_v, v), dim = -2)

        q, k = map(lambda t: self.qk_activation(t), (q, k))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask) or exists(context_mask):
            i, j = sim.shape[-2:]

            if not exists(mask):
                mask = torch.ones(b, i, dtype = torch.bool, device = device)

            if exists(context_mask):
                context_mask = F.pad(context_mask, (1, 0), value = True)
            else:
                context_mask = torch.ones(b, j, dtype = torch.bool, device = device)

            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            sim.masked_fill_(~mask, max_neg_value(sim))

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Layer(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        dim,
        dim_global,
        narrow_conv_kernel = 9,
        wide_conv_kernel = 9,
        wide_conv_dilation = 5,
        attn_heads = 8,
        attn_dim_head = 64,
        attn_qk_activation = nn.Tanh(),
        local_to_global_attn = False,
        local_self_attn = False,
        glu_conv = False
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 如果启用局部自注意力机制，则创建全局线性自注意力对象
        self.seq_self_attn = GlobalLinearSelfAttention(dim = dim, dim_head = attn_dim_head, heads = attn_heads) if local_self_attn else None

        # 如果启用门控线性单元，则设置卷积倍数为2，否则为1
        conv_mult = 2 if glu_conv else 1

        # 创建窄卷积层
        self.narrow_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, narrow_conv_kernel, padding = narrow_conv_kernel // 2),
            nn.GELU() if not glu_conv else nn.GLU(dim = 1)
        )

        # 计算宽卷积的填充大小
        wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

        # 创建宽卷积层
        self.wide_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, wide_conv_kernel, dilation = wide_conv_dilation, padding = wide_conv_padding),
            nn.GELU() if not glu_conv else nn.GLU(dim = 1)
        )

        # 设置是否进行局部到全局的注意力计算
        self.local_to_global_attn = local_to_global_attn

        # 根据是否进行局部到全局的注意力计算，创建相应的全局信息提取层
        if local_to_global_attn:
            self.extract_global_info = CrossAttention(
                dim = dim,
                dim_keys = dim_global,
                dim_out = dim,
                heads = attn_heads,
                dim_head = attn_dim_head
            )
        else:
            self.extract_global_info = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.Linear(dim_global, dim),
                nn.GELU(),
                Rearrange('b d -> b () d')
            )

        # 创建局部层归一化层
        self.local_norm = nn.LayerNorm(dim)

        # 创建局部前馈网络
        self.local_feedforward = nn.Sequential(
            Residual(nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
            )),
            nn.LayerNorm(dim)
        )

        # 创建全局关注局部的交叉注意力层
        self.global_attend_local = CrossAttention(dim = dim_global, dim_out = dim_global, dim_keys = dim, heads = attn_heads, dim_head = attn_dim_head, qk_activation = attn_qk_activation)

        # 创建全局密集层
        self.global_dense = nn.Sequential(
            nn.Linear(dim_global, dim_global),
            nn.GELU()
        )

        # 创建全局层归一化层
        self.global_norm = nn.LayerNorm(dim_global)

        # 创建全局前馈网络
        self.global_feedforward = nn.Sequential(
            Residual(nn.Sequential(
                nn.Linear(dim_global, dim_global),
                nn.GELU()
            )),
            nn.LayerNorm(dim_global),
        )

    # 前向传播函数
    def forward(self, tokens, annotation, mask = None):
        # 如果启用局部到全局的注意力计算，则提取全局信息
        if self.local_to_global_attn:
            global_info = self.extract_global_info(tokens, annotation, mask = mask)
        else:
            global_info = self.extract_global_info(annotation)

        # 处理局部（蛋白质序列）

        # 如果存在局部自注意力机制，则计算全局线性注意力
        global_linear_attn = self.seq_self_attn(tokens) if exists(self.seq_self_attn) else 0

        # 重排输入以适应卷积层的输入格式
        conv_input = rearrange(tokens, 'b n d -> b d n')

        # 如果存在掩码，则根据掩码进行填充
        if exists(mask):
            conv_input_mask = rearrange(mask, 'b n -> b () n')
            conv_input = conv_input.masked_fill(~conv_input_mask, 0.)

        # 进行窄卷积和宽卷积操作
        narrow_out = self.narrow_conv(conv_input)
        narrow_out = rearrange(narrow_out, 'b d n -> b n d')
        wide_out = self.wide_conv(conv_input)
        wide_out = rearrange(wide_out, 'b d n -> b n d')

        # 更新 tokens
        tokens = tokens + narrow_out + wide_out + global_info + global_linear_attn
        tokens = self.local_norm(tokens)

        # 应用局部前馈网络
        tokens = self.local_feedforward(tokens)

        # 处理全局（注释）

        # 全局关注局部的交叉注意力
        annotation = self.global_attend_local(annotation, tokens, context_mask = mask)
        annotation = self.global_dense(annotation)
        annotation = self.global_norm(annotation)
        annotation = self.global_feedforward(annotation)

        return tokens, annotation
# 主模型类定义
class ProteinBERT(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        num_tokens = 26,  # 标记的数量
        num_annotation = 8943,  # 注释的数量
        dim = 512,  # 维度
        dim_global = 256,  # 全局维度
        depth = 6,  # 深度
        narrow_conv_kernel = 9,  # 窄卷积核大小
        wide_conv_kernel = 9,  # 宽卷积核大小
        wide_conv_dilation = 5,  # 宽卷积膨胀率
        attn_heads = 8,  # 注意力头数
        attn_dim_head = 64,  # 注意力头维度
        attn_qk_activation = nn.Tanh(),  # 注意力激活函数
        local_to_global_attn = False,  # 是否使用局部到全局注意力
        local_self_attn = False,  # 是否使用局部自注意力
        num_global_tokens = 1,  # 全局标记数量
        glu_conv = False  # 是否使用门控线性单元卷积
    ):
        super().__init__()
        self.num_tokens = num_tokens  # 设置标记数量
        self.token_emb = nn.Embedding(num_tokens, dim)  # 标记嵌入层

        self.num_global_tokens = num_global_tokens  # 设置全局标记数量
        self.to_global_emb = nn.Linear(num_annotation, num_global_tokens * dim_global)  # 全局嵌入层

        # 创建多层神经网络
        self.layers = nn.ModuleList([Layer(dim = dim, dim_global = dim_global, narrow_conv_kernel = narrow_conv_kernel, wide_conv_dilation = wide_conv_dilation, wide_conv_kernel = wide_conv_kernel, attn_qk_activation = attn_qk_activation, local_to_global_attn = local_to_global_attn, local_self_attn = local_self_attn, glu_conv = glu_conv) for layer in range(depth)])

        self.to_token_logits = nn.Linear(dim, num_tokens)  # 标记的逻辑回归层

        self.to_annotation_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),  # 减少维度
            nn.Linear(dim_global, num_annotation)  # 全局注释的逻辑回归层
        )

    # 前向传播函数
    def forward(self, seq, annotation, mask = None):
        tokens = self.token_emb(seq)  # 标记嵌入

        annotation = self.to_global_emb(annotation)  # 全局嵌入
        annotation = rearrange(annotation, 'b (n d) -> b n d', n = self.num_global_tokens)  # 重新排列全局嵌入

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation, mask = mask)  # 多层神经网络的前向传播

        tokens = self.to_token_logits(tokens)  # 标记的逻辑回归
        annotation = self.to_annotation_logits(annotation)  # 全局注释的逻辑回归
        return tokens, annotation  # 返回标记和注释

# 预训练包装器类定义
class PretrainingWrapper(nn.Module):
    # 初始化函数
    def __init__(
        self,
        model,
        random_replace_token_prob = 0.05,  # 随机替换标记的概率
        remove_annotation_prob = 0.25,  # 移除注释的概率
        add_annotation_prob = 0.01,  # 添加注释的概率
        remove_all_annotations_prob = 0.5,  # 移除所有注释的概率
        seq_loss_weight = 1.,  # 序列损失权重
        annotation_loss_weight = 1.,  # 注释损失权重
        exclude_token_ids = (0, 1, 2)   # 要排除的标记ID（用于排除填充、开始和结束标记）
    ):
        super().__init__()
        assert isinstance(model, ProteinBERT), 'model must be an instance of ProteinBERT'  # 断言模型必须是ProteinBERT的实例

        self.model = model  # 设置模型

        self.random_replace_token_prob = random_replace_token_prob  # 设置随机替换标记的概率
        self.remove_annotation_prob = remove_annotation_prob  # 设置移除注释的概率
        self.add_annotation_prob = add_annotation_prob  # 设置添加注释的概率
        self.remove_all_annotations_prob = remove_all_annotations_prob  # 设置移除所有注释的概率

        self.seq_loss_weight = seq_loss_weight  # 设置序列损失权重
        self.annotation_loss_weight = annotation_loss_weight  # 设置注释损失权重

        self.exclude_token_ids = exclude_token_ids  # 设置要排除的标记ID
    # 定义一个前向传播函数，接受序列、注释和掩码作为输入
    def forward(self, seq, annotation, mask = None):
        # 获取批量大小和设备信息
        batch_size, device = seq.shape[0], seq.device

        # 复制输入序列和注释
        seq_labels = seq
        annotation_labels = annotation

        # 如果没有提供掩码，则创建一个全为 True 的掩码
        if not exists(mask):
            mask = torch.ones_like(seq).bool()

        # 准备用于对序列进行噪声处理的掩码

        excluded_tokens_mask = mask

        # 根据排除的标记 ID，生成排除标记的掩码
        for token_id in self.exclude_token_ids:
            excluded_tokens_mask = excluded_tokens_mask & (seq != token_id)

        # 根据给定的概率生成随机替换标记的掩码
        random_replace_token_prob_mask = get_mask_subset_with_prob(excluded_tokens_mask, self.random_replace_token_prob)

        # 准备用于对注释进行噪声处理的掩码

        batch_mask = torch.ones(batch_size, device = device, dtype = torch.bool)
        batch_mask = rearrange(batch_mask, 'b -> b ()')
        remove_annotation_from_batch_mask = get_mask_subset_with_prob(batch_mask, self.remove_all_annotations_prob)

        annotation_mask = annotation > 0
        remove_annotation_prob_mask = get_mask_subset_with_prob(annotation_mask, self.remove_annotation_prob)
        add_annotation_prob_mask = get_mask_subset_with_prob(~annotation_mask, self.add_annotation_prob)
        remove_annotation_mask = remove_annotation_from_batch_mask & remove_annotation_prob_mask

        # 生成随机标记

        random_tokens = torch.randint(0, self.model.num_tokens, seq.shape, device=seq.device)

        # 确保不会用排除的标记类型（填充、开始、结束）替换标记
        for token_id in self.exclude_token_ids:
            random_replace_token_prob_mask = random_replace_token_prob_mask & (random_tokens != token_id)

        # 对序列进行噪声处理

        noised_seq = torch.where(random_replace_token_prob_mask, random_tokens, seq)

        # 对注释进行噪声处理

        noised_annotation = annotation + add_annotation_prob_mask.type(annotation.dtype)
        noised_annotation = noised_annotation * remove_annotation_mask.type(annotation.dtype)

        # 使用模型进行去噪处理

        seq_logits, annotation_logits = self.model(noised_seq, noised_annotation, mask = mask)

        # 计算损失

        seq_logits = seq_logits[mask]
        seq_labels = seq_labels[mask]

        seq_loss = F.cross_entropy(seq_logits, seq_labels, reduction = 'sum')
        annotation_loss = F.binary_cross_entropy_with_logits(annotation_logits, annotation_labels, reduction = 'sum')

        # 返回序列损失加上注释损失的加权和
        return seq_loss * self.seq_loss_weight + annotation_loss * self.annotation_loss_weight
```