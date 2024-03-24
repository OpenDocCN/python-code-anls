# `.\lucidrains\n-grammer-pytorch\n_grammer_pytorch\n_grammer_pytorch.py`

```py
# 基于 jax 代码的实现
# https://github.com/tensorflow/lingvo/blob/master/lingvo/jax/layers/ngrammer.py

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import sympy

# 辅助函数

def exists(val):
    return val is not None

def sum_squares(t, dim = -1):
    return (t ** 2).sum(dim = dim)

# 与 bigram 相关的函数

def multi_way_hash_ids(x, a, b, prime, buckets):
    return ((x * a + b) % prime) % buckets

def get_bigram_ids(ids, vocab_size, segment_pos = None):
    # ids 的形状为 (batch, seq, heads)

    ids = ids.long()
    ids_0 = F.pad(ids, (0, 0, 0, 1))
    ids_1 = F.pad(ids, (0, 0, 1, 0))

    if exists(segment_pos):
        segment_pos = rearrange(segment_pos, 'b n -> b n 1')
        mask = (segment_pos == 0).long()
        mask = 1 - mask
        mask = F.pad(mask, (0, 0, 0, 1))
        ids_1 *= mask

    ngram_ids = ids_0 + ids_1 * vocab_size
    ngram_ids = ngram_ids[:, :-1]
    return ngram_ids

# 与优化器相关的函数

def get_ngrammer_parameters(module):
    params = set()
    for m in module.modules():
        if isinstance(m, Ngrammer):
            params.update(m.parameters())
    rest = set(module.parameters()) - params
    return list(params), list(rest)

def get_ngrammer_param_groups(module, ngrammer_learning_rate = 1e-2):
    ngrammer_params, rest = get_ngrammer_parameters(module)
    return [{'params': rest}, {'params': ngrammer_params, 'lr': ngrammer_learning_rate}]

# layernorm

class MultiheadLayerNorm(nn.Module):
    def __init__(self, dim, heads = 1, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(heads, dim))
        self.b = nn.Parameter(torch.zeros(heads, dim))

    def forward(self, x):
        std = torch.var(x, dim = -1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

# 类

class VectorQuantization(nn.Module):
    def __init__(
        self,
        *,
        num_clusters,
        num_heads,
        dim_per_head,
        decay = 0.999,
        epsilon = 1e-6
    ):
        super().__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.num_clusters = num_clusters

        self.register_buffer('means', torch.randn(num_heads, num_clusters, dim_per_head))

    def forward(
        self,
        x,
        mask = None
    ):
        h, dim_head, num_clusters, eps, decay, means = self.num_heads, self.dim_per_head, self.num_clusters, self.epsilon, self.decay, self.means
        assert x.shape[-1] == (h * dim_head), f'input embedding feature dimension must be {h * dim_head}'

        # 将输入中的头部分离出来

        x = rearrange(x, 'b n (h d) -> b n h d', h = h)

        # 获取输入嵌入与均值之间的距离

        dists = (
            rearrange(sum_squares(x), 'b n h -> b n h 1')
            - 2 * einsum('b n h d, h k d -> b n h k', x, means)
            + rearrange(sum_squares(means), 'h k -> 1 1 h k')
        )

        # 获取簇 id

        cluster_ids = dists.argmin(dim = -1)

        if self.training:
            # 获取 one hot 编码，用于计算每个均值的匹配数

            nearest_one_hot = F.one_hot(cluster_ids, num_classes = num_clusters)
            per_cluster_count = nearest_one_hot.sum(dim = (0, 1))

            # 每个最近质心的输入之和。

            sum_x = einsum('b n h k, b n h d -> h k d', nearest_one_hot.float(), x)

            # 计算新的均值

            new_means = sum_x / (eps + rearrange(per_cluster_count, '... -> ... 1'))

            # 指数移动平均

            updated_means = (1. - decay) * new_means + decay * means

            self.means.data.copy_(updated_means)

        return cluster_ids

class Ngrammer(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        unigram_vocab_size,  # 单字词汇表大小
        dim_per_head,  # 每个头的维度
        num_heads = 1,  # 头的数量，默认为1
        ngram_emb_dim = 8,  # n-gram嵌入维度，默认为8
        ngram_vocab_size = 768 * 256,  # n-gram词汇表大小，默认为768 * 256
        concat_ngrams = True  # 是否连接n-gram，默认为True
    ):
        super().__init__()
        # 断言，确保当连接n-gram时，每个头的维度不能小于n-gram嵌入维度
        assert not (concat_ngrams and dim_per_head <= ngram_emb_dim), 'unigram head dimension cannot be smaller than ngram embedding dimension when concatting'
        # 断言，确保当不连接n-gram时，每个头的维度必须等于n-gram嵌入维度
        assert not (not concat_ngrams and dim_per_head != ngram_emb_dim), 'unigram head dimension must be equal to ngram embedding dimension if not concatting'

        # 初始化模型参数
        self.num_heads = num_heads
        self.ngram_vocab_size = ngram_vocab_size
        self.unigram_vocab_size = unigram_vocab_size
        self.concat_ngrams = concat_ngrams

        # 初始化模型的嵌入层
        self.embeddings = nn.ModuleList([])

        # 初始化n-gram的LayerNorm
        self.ngram_layernorm = MultiheadLayerNorm(ngram_emb_dim, heads = num_heads)
        # 初始化嵌入的LayerNorm
        self.embeds_layernorm = MultiheadLayerNorm(dim_per_head, heads = num_heads)

        # 初始化n-gram的Embedding层
        self.ngram_embeds = nn.Embedding(ngram_vocab_size * num_heads, ngram_emb_dim)

        # 生成质数列表，用于多头哈希计算
        primes = list(sympy.primerange(ngram_vocab_size + 1, 2 * ngram_vocab_size))[:num_heads]
        self.register_buffer('primes', torch.tensor(primes), persistent = False)

    # 前向传播函数
    def forward(
        self,
        embeds,  # 嵌入
        cluster_ids,  # 聚类ID
        mask = None,  # 掩码，默认为None
        segment_pos = None  # 分段位置，默认为None
    ):
        # 获取模型参数
        num_heads, vocab_size, unigram_vocab_size, device = self.num_heads, self.ngram_vocab_size, self.unigram_vocab_size, embeds.device

        # 如果聚类ID的维度为2，则重复扩展为多头
        if cluster_ids.ndim == 2:
            cluster_ids = repeat(cluster_ids, '... -> ... h', h = num_heads)

        # 获取n-gram聚类ID
        ngram_cluster_ids = get_bigram_ids(cluster_ids, unigram_vocab_size, segment_pos)

        # 准备用于并行计算多头哈希ID的头范围
        head_range = torch.arange(num_heads, device = device)
        head_range = rearrange(head_range, 'h -> 1 1 h')
        primes = rearrange(self.primes, 'h -> 1 1 h')

        # 多头哈希ID计算
        ngram_ids = multi_way_hash_ids(ngram_cluster_ids, head_range + 1, head_range + 1, primes, vocab_size)

        # 根据头编号适当地移动词汇范围
        ngram_ids = ngram_ids + (vocab_size * head_range)

        # 一次性获取所有n-gram嵌入，并进行多头LayerNorm
        ngram_embeds = self.ngram_embeds(ngram_ids)
        normed_ngram_embeds = self.ngram_layernorm(ngram_embeds)

        # 多头LayerNorm输入
        embeds = rearrange(embeds, 'b n (h d) -> b n h d', h = num_heads)
        normed_embeds = self.embeds_layernorm(embeds)

        # 连接原始单字嵌入和bigram
        if self.concat_ngrams:
            input_sliced_dim = normed_embeds.shape[-1] - normed_ngram_embeds.shape[-1]
            out = torch.cat((
                normed_embeds[..., :input_sliced_dim],
                normed_ngram_embeds
            ), dim = -1)
        else:
            out = normed_embeds + normed_ngram_embeds

        # 展平
        out = rearrange(out, 'b n ... -> b n (...)')

        # 如果需要，进行掩码
        if exists(mask):
            out = out * rearrange(mask, 'b n -> b n 1').float()

        return out
# 主类定义

class VQNgrammer(nn.Module):
    def __init__(
        self,
        *,
        num_clusters,  # 聚类中心数量
        num_heads,  # 多头注意力机制中头的数量
        dim_per_head,  # 每个头的维度
        ngram_vocab_size = 768 * 256,  # N-gram词汇表大小，默认为768*256
        ngram_emb_dim = 8,  # N-gram嵌入维度，默认为8
        concat_ngrams = True,  # 是否连接N-gram
        decay = 0.999,  # 衰减率，默认为0.999
        epsilon = 1e-6  # 防止除零错误的小值，默认为1e-6
    ):
        super().__init__()
        assert ngram_vocab_size < (num_clusters ** 2), 'the ngram vocab size should be less than the number of clusters squared'

        # 初始化向量量化模块
        self.vq = VectorQuantization(
            num_clusters = num_clusters,
            num_heads = num_heads,
            dim_per_head = dim_per_head,
            decay = decay,
            epsilon = epsilon
        )

        # 初始化N-gram模块
        self.ngram = Ngrammer(
            unigram_vocab_size = num_clusters,
            ngram_vocab_size = ngram_vocab_size,
            ngram_emb_dim = ngram_emb_dim,
            concat_ngrams = concat_ngrams,
            num_heads = num_heads,
            dim_per_head = dim_per_head
        )

    def forward(
        self,
        x,
        mask = None,
        segment_pos = None
    ):

        # 使用向量量化模块对输入进行聚类
        cluster_ids = self.vq(x, mask = mask)

        # 使用N-gram模块处理输入数据
        out = self.ngram(
            x,
            cluster_ids = cluster_ids,
            mask = mask,
            segment_pos = segment_pos
        )

        return out
```