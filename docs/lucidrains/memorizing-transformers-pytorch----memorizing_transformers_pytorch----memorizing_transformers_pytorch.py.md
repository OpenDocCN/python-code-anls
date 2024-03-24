# `.\lucidrains\memorizing-transformers-pytorch\memorizing_transformers_pytorch\memorizing_transformers_pytorch.py`

```py
# 导入数学库
import math
# 从 functools 模块导入 partial 函数
from functools import partial
# 从 contextlib 模块导入 contextmanager 上下文管理器
from contextlib import contextmanager
# 从 pathlib 模块导入 Path 类
from pathlib import Path
# 从 filelock 模块导入 FileLock 类
from filelock import FileLock

# 导入 torch 库
import torch
# 从 torch 中导入 nn 模块和 F 模块
import torch.nn.functional as F
# 从 torch 中导入 nn 模块和 einsum 函数
from torch import nn, einsum

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 einops_exts 库中导入 repeat_many 函数
from einops_exts import repeat_many
# 从 einops.layers.torch 中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 从 memorizing_transformers_pytorch.knn_memory 模块中导入 KNNMemoryList 类和 DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY 常量

# 辅助函数

# 定义一个返回输入的函数
def identity(t):
    return t

# 判断输入是否存在的函数
def exists(val):
    return val is not None

# 返回输入列表中唯一元素的函数
def unique(arr):
    return list({el: True for el in arr}.keys())

# 返回输入值或默认值的函数
def default(val, d):
    return val if exists(val) else d

# 将输入值转换为元组的函数
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 对输入张量进行 L2 归一化的函数
def l2norm(t):
    return F.normalize(t, dim = -1)

# 辅助类

# 实现预层归一化残差连接的类
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        out = self.fn(self.norm(x), **kwargs)

        if not isinstance(out, tuple):
            return out + x

        head, *tail = out
        return (head + x, *tail)

# T5 相对位置偏置类

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return bias * self.scale

# 前馈网络类

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# 注意力机制类

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        xl_max_memories = 0.,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.xl_max_memories = xl_max_memories

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
    # 定义一个前向传播函数，接受输入 x，可选的 xl_memory 和 rel_pos_bias 参数
    def forward(self, x, *, xl_memory = None, rel_pos_bias = None):
        # 获取头数和设备信息
        h, device = self.heads, x.device
        # 将输入 x 分别转换为查询 q，键 k，值 v
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        # 重新排列查询 q 的维度
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        # 对查询 q 进行缩放
        q = q * self.scale

        # 如果存在 xl_memory，则将其拆分为键值对，并与当前的 k 和 v 连接起来
        if exists(xl_memory):
            k_xl_mem, v_xl_mem = xl_memory.unbind(dim = -2)
            k = torch.cat((k_xl_mem, k), dim = -2)
            v = torch.cat((v_xl_mem, v), dim = -2)

        # 计算查询和键之间的相似度
        sim = einsum('b h i d, b j d -> b h i j', q, k)
        i, j = sim.shape[-2:]

        # 如果存在相对位置偏置，则加到相似度上
        if exists(rel_pos_bias):
            sim = rel_pos_bias[..., -i:, -j:] + sim

        # 创建一个因果掩码，用于屏蔽未来信息
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 对相似度进行 softmax 操作
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # 根据注意力权重计算输出
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 创建新的 xl 记忆
        new_kv_memories = torch.stack((k, v), dim = -2).detach()

        # 如果设置了最大 xl 记忆数，则保留最新的 xl 记忆
        if self.xl_max_memories > 0:
            new_xl_kv_memories = new_kv_memories[:, -self.xl_max_memories:]
        else:
            new_xl_kv_memories = None

        # 返回输出和新的 xl 记忆
        return self.to_out(out), new_xl_kv_memories
# 定义一个近似最近邻注意力机制的类 KNNAttention
class KNNAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 输入特征的维度
        heads = 8,  # 多头注意力的头数
        dim_head = 64,  # 每个头的维度
        dropout = 0.,  # dropout 概率
        num_retrieved_memories = 32,  # 检索的记忆数量
        xl_max_memories = 0.,  # 最大记忆数量
        attn_scale_init = 20,  # 注意力缩放初始化值
        gate_output = False  # 是否使用输出门
    ):
        super().__init__()
        self.heads = heads  # 头数
        self.scale = nn.Parameter(torch.ones(heads, 1, 1) * math.log(attn_scale_init))  # 缩放参数

        inner_dim = heads * dim_head  # 内部维度
        self.xl_max_memories = xl_max_memories  # 最大记忆数量

        self.num_retrieved_memories = num_retrieved_memories  # 检索的记忆数量

        self.dropout = nn.Dropout(dropout)  # dropout 操作
        self.knn_mem_dropout = nn.Dropout(dropout)  # knn 记忆的 dropout 操作

        self.to_q = nn.Linear(dim, inner_dim, bias = False)  # 查询映射
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)  # 键值映射
        self.to_out = nn.Linear(inner_dim, dim, bias = False)  # 输出映射

        self.output_gate = nn.Parameter(torch.zeros(1)) if gate_output else None  # 输出门参数

    def forward(
        self,
        x,  # 输入张量
        *,
        knn_memory,  # KNN 记忆
        xl_memory = None,  # XL 记忆
        add_knn_memory = True,  # 是否添加 KNN 记忆
        rel_pos_bias = None  # 相对位置偏置
        ):
            # 解包 x 的形状，获取 batch size, 序列长度, 头数, 设备信息
            b, n, h, device = *x.shape[:2], self.heads
            # 将输入 x 分别转换为查询 q, 键 k, 值 v
            q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

            # 重排查询 q 的形状，以适应多头注意力计算
            q = rearrange(q, 'b n (h d) -> b h n d', h = h)

            # 根据论文，对键进行归一化以提高训练稳定性
            # 这里采用完全余弦相似度注意力 https://arxiv.org/abs/2010.04245
            q, k = map(l2norm, (q, k))

            # 处理 XL 内存
            if exists(xl_memory):
                k_xl_mem, v_xl_mem = xl_memory.unbind(dim = -2)
                k = torch.cat((k_xl_mem, k), dim = -2)
                v = torch.cat((v_xl_mem, v), dim = -2)

            # 计算局部注意力
            scale = self.scale.exp()

            sim = einsum('b h i d, b j d -> b h i j', q, k) * scale
            i, j = sim.shape[-2:]

            # 如果存在相对位置偏置，则加入到注意力矩阵中
            if exists(rel_pos_bias):
                sim = rel_pos_bias[..., -i:, -j:] + sim

            mask_value = -torch.finfo(sim.dtype).max

            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

            # 如果传入索引，则计算记忆中的 knn 注意力
            mem_kv, mem_mask = knn_memory.search(q, self.num_retrieved_memories)
            mem_k, mem_v = mem_kv.unbind(dim = -2)

            sim_mem = einsum('b h i d, b h i j d -> b h i j', q, mem_k) * scale
            sim_mem = sim_mem.masked_fill(~mem_mask, mask_value)

            # 计算新的 XL 记忆，以及要丢弃的记忆
            new_kv_memories = torch.stack((k, v), dim = -2).detach()

            if self.xl_max_memories > 0:
                new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories[:, :-self.xl_max_memories], new_kv_memories[:, -self.xl_max_memories:]
            else:
                new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories, None

            # 将要丢弃的记忆添加到 KNN 记忆中
            if add_knn_memory and new_kv_memories_discarded.numel() > 0:
                knn_memory.add(new_kv_memories_discarded)

            # 组合局部和远程注意力
            sim = torch.cat((sim_mem, sim), dim = -1)
            attn = sim.softmax(dim = -1)
            attn = self.dropout(attn)

            local_attn, mem_attn = attn[..., self.num_retrieved_memories:], attn[..., :self.num_retrieved_memories]
            local_out = einsum('b h i j, b j d -> b h i d', local_attn, v)
            mem_out = einsum('b h i j, b h i j d -> b h i d', mem_attn, mem_v)

            out = local_out + mem_out

            # 合并头部并进行投影
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)

            # 使用 flamingo 风格的输出门控制输出，以便将记忆化 transformer 门控到现有的 LLM 中
            if exists(self.output_gate):
                out = out * self.output_gate.tanh()

            return out, new_xl_kv_memories
# 主类
class MemorizingTransformer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        num_tokens,  # 标记数量
        dim,  # 维度
        depth,  # 深度
        dim_head = 64,  # 头维度
        heads = 8,  # 头数
        knn_attn_heads = None,  # KNN注意力头数
        attn_dropout = 0.,  # 注意力丢弃率
        ff_mult = 4,  # 前馈倍数
        ff_dropout = 0.,  # 前馈丢弃率
        memorizing_layers = None,  # 记忆层
        max_knn_memories = 250000,  # 最大KNN记忆
        num_retrieved_memories = 32,  # 检索的记忆数
        clear_memories_on_sos_token_id = None,  # SOS标记时清除记忆
        clear_memories_on_eos_token_id = None,  # EOS标记时清除记忆
        knn_memories_directory = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,  # KNN记忆目录
        shift_knn_memories_down = 0.,  # KNN记忆下移
        pad_id = 0,  # 填充标记
        xl_max_memories = 0,  # XL最大记忆
        xl_memory_layers = None,  # XL记忆层
        shift_xl_memories_down = 0.,  # XL记忆下移
        knn_memory_multiprocessing = False  # KNN记忆多进程
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)  # 标记嵌入
        self.pad_id = pad_id  # 填充标记

        block_wrapper = partial(PreNormResidual, dim)  # 块包装器
        valid_layers = set(range(1, depth + 1))  # 有效层范围

        memorizing_layers = default(memorizing_layers, (depth // 2,))  # 默认KNN注意力层为变压器中点
        memorizing_layers = cast_tuple(memorizing_layers)  # 转换为元组
        memorizing_layers = tuple(filter(lambda i: i in valid_layers, memorizing_layers))  # 过滤有效层

        self.dim_head = dim_head  # 头维度

        knn_attn_heads = default(knn_attn_heads, heads)  # 默认KNN注意力头数

        # XL记忆超参数
        if xl_max_memories > 0:
            xl_memory_layers = default(xl_memory_layers, tuple(range(1, depth + 1)))  # 默认XL记忆层为所有层
            xl_memory_layers = unique(xl_memory_layers)  # 唯一值
            self.xl_memory_layers = tuple(filter(lambda i: i in valid_layers, xl_memory_layers))  # 过滤有效层
            self.num_xl_memory_layers = len(self.xl_memory_layers)  # XL记忆层数
        else:
            self.xl_memory_layers = tuple()
            self.num_xl_memory_layers = 0

        # KNN记忆超参数
        self.max_knn_memories = max_knn_memories  # 最大KNN记忆
        self.knn_memories_directory = knn_memories_directory  # KNN记忆目录
        self.memorizing_layers = unique(memorizing_layers)  # 唯一值
        self.num_memory_layers = len(memorizing_layers)  # 记���层数

        self.clear_memories_on_sos_token_id = clear_memories_on_sos_token_id  # SOS标记时清除记忆
        self.clear_memories_on_eos_token_id = clear_memories_on_eos_token_id  # EOS标记时清除记忆

        # 相对位置偏置
        self.rel_pos_bias = T5RelativePositionBias(scale = dim_head ** 0.5, heads = heads)  # 相对位置偏置
        self.knn_rel_pos_bias = T5RelativePositionBias(scale = dim_head ** 0.5, heads = heads)  # KNN相对位置偏置

        # 层
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer_num = idx + 1

            use_xl_memories = layer_num in self.xl_memory_layers  # 使用XL记忆
            use_knn_attention = layer_num in memorizing_layers  # 使用KNN注意力
            xl_max_memories_layer = 0 if not use_xl_memories else xl_max_memories  # XL最大记忆层

            if use_knn_attention:
                attn = KNNAttention(dim = dim, dim_head = dim_head, heads = knn_attn_heads, dropout = attn_dropout, num_retrieved_memories = num_retrieved_memories, xl_max_memories = xl_max_memories_layer)  # KNN注意力
            else:
                attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, xl_max_memories = xl_max_memories_layer)  # 注意力

            self.layers.append(nn.ModuleList([
                block_wrapper(attn),
                block_wrapper(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))

        # 记忆层移动
        # 来自一篇鲜为人知的论文 https://arxiv.org/abs/2012.15688

        self.shift_knn_memories_down = shift_knn_memories_down  # KNN记忆下移
        self.shift_xl_memories_down = shift_xl_memories_down  # XL记忆下移

        # 转换为logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

        # KNN记忆初始化
        self.knn_mem_kwargs = dict(
            dim = self.dim_head,
            max_memories = self.max_knn_memories,
            multiprocessing = knn_memory_multiprocessing
        )
    # 创建 KNN 记忆体列表
    def create_knn_memories(
        self,
        *,
        batch_size
    ):
        # 调用 KNNMemoryList 类的 create_memories 方法创建记忆体
        return KNNMemoryList.create_memories(
            batch_size = batch_size,
            num_memory_layers = self.num_memory_layers,
            memories_directory = self.knn_memories_directory,
        )(**self.knn_mem_kwargs)

    # 上下文管理器，用于处理 KNN 记忆体
    @contextmanager
    def knn_memories_context(
        self,
        **kwargs
    ):
        # 获取 KNN 记忆体目录路径
        knn_dir = Path(self.knn_memories_directory)
        # 如果目录不存在则创建
        knn_dir.mkdir(exist_ok = True, parents = True)
        # 创建文件锁
        lock = FileLock(str(knn_dir / 'mutex'))

        # 使用文件锁
        with lock:
            # 创建 KNN 记忆体
            knn_memories = self.create_knn_memories(**kwargs)
            # 通过 yield 将 KNN 记忆体传递给调用者
            yield knn_memories
            # 清理 KNN 记忆体
            knn_memories.cleanup()

    # 清除记忆体中包含指定 token id 的批次行
    def clear_memory(self, x, token_id):
        """ clears the KNN memories based on if the batch row contains the specified token id """
        """ for auto-clearing KNN memories based on start and end of strings """

        # 判断是否需要清除记忆体
        clear_memory = (x == token_id).any(dim = -1)
        # 获取需要清除的批次索引
        batch_indices, _ = clear_memory.nonzero(as_tuple = True)
        batch_indices_to_clear = batch_indices.tolist()

        # 如果没有需要清除的批次索引，则直接返回
        if len(batch_indices_to_clear) == 0:
            return

        # 清除指定批次索引的记忆体
        knn_memories.clear_memory(batch_indices_to_clear)

    # 前向传播函数
    def forward(
        self,
        x,
        knn_memories,
        xl_memories = None,
        labels = None,
        add_knn_memory = True
        ):
            # 解构输入张量 x 的形状，获取批量大小、序列长度和设备信息
            batch_size, seq_len, *_, device = *x.shape, x.device
            # 使用 token_emb 对象对输入张量 x 进行 token 嵌入
            x = self.token_emb(x)

            # 验证 KNN memories 是否有足够的索引来匹配批量大小

            assert all([memory.num_indices == batch_size for memory in knn_memories]), f'you passed in an input with batch size {batch_size} but your memories were not instantiated with that number of KNN indices'

            # 如果传入了 KNN memories，并且研究人员希望在检测到 <sos> 标记时自动清除 memories
            # 执行适当的逻辑

            if exists(self.clear_memories_on_sos_token_id):
                self.clear_memory(x, self.clear_memories_on_sos_token_id)

            # 处理 XL memories

            xl_memories = default(xl_memories, (None,) * self.num_xl_memory_layers)
            assert len(xl_memories) == self.num_xl_memory_layers
            has_xl_memories = len(xl_memories) > 0

            # 将 memories 向下移动若干层，这是 Ernie-Doc 论文中展示的增强 memories 的鲜为人知的技术

            if len(knn_memories) > 0 and self.shift_knn_memories_down > 0:
                knn_memories = [*knn_memories[self.shift_knn_memories_down:], *knn_memories[:self.shift_knn_memories_down]]

            if len(xl_memories) > 0 and self.shift_xl_memories_down > 0:
                xl_memories = [*xl_memories[self.shift_xl_memories_down:], *xl_memories[:self.shift_xl_memories_down]]

            # 按照包含 KNNAttention 的升序层次顺序遍历 memories

            xl_memories_iter = iter(xl_memories)
            knn_memories_iter = iter(knn_memories)

            # 位置偏置

            max_context_len = max([seq_len, *map(lambda t: (t.shape[-3] if exists(t) else 0) + seq_len, xl_memories)])

            rel_pos_bias = self.rel_pos_bias(seq_len, max_context_len, device = device)
            knn_rel_pos_bias = self.knn_rel_pos_bias(seq_len, max_context_len, device = device)

            # 跟踪新的 XL memories

            new_xl_memories = [] if has_xl_memories else None

            # 遍历所有层

            for ind, (attn, ff) in enumerate(self.layers):
                layer_num = ind + 1

                is_memorizing_layer = layer_num in self.memorizing_layers
                is_xl_memory_layer = layer_num in self.xl_memory_layers

                attn_kwargs = dict(rel_pos_bias = rel_pos_bias if not is_memorizing_layer else knn_rel_pos_bias)

                if is_memorizing_layer:
                    attn_kwargs = {**attn_kwargs, 'knn_memory': next(knn_memories_iter), 'add_knn_memory': add_knn_memory}

                if is_xl_memory_layer:
                    attn_kwargs = {**attn_kwargs, 'xl_memory': next(xl_memories_iter)}

                # 注意力机制

                x, xl_mem = attn(x, **attn_kwargs)

                # 如果需要，添加新的 XL memories

                if exists(xl_mem):
                    new_xl_memories.append(xl_mem)

                # 前馈网络

                x = ff(x)

            # 转换为 logits

            logits = self.to_logits(x)

            # 在字符串结束标记时自动清除 KNN memories

            if exists(self.clear_memories_on_eos_token_id):
                self.clear_memory(x, self.clear_memories_on_eos_token_id)

            # 对于训练

            if not exists(labels):
                if exists(new_xl_memories):
                    return logits, new_xl_memories

                return logits

            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = self.pad_id)

            if exists(new_xl_memories):
                return loss, new_xl_memories

            return loss
```