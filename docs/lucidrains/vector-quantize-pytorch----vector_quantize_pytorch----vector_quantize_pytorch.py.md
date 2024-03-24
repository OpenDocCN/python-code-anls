# `.\lucidrains\vector-quantize-pytorch\vector_quantize_pytorch\vector_quantize_pytorch.py`

```
# 导入必要的库
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.cuda.amp import autocast

# 导入 einops 库中的函数
from einops import rearrange, repeat, reduce, pack, unpack

# 导入 Callable 类型
from typing import Callable

# 检查变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 空函数
def noop(*args, **kwargs):
    pass

# 返回输入的函数
def identity(t):
    return t

# 对输入进行 L2 归一化
def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

# 计算输入张量 x 和 y 之间的欧氏距离
def cdist(x, y):
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min = 0).sqrt()

# 计算输入张量的自然对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 在原地更新指数移动平均值
def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith('mps:')

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))

# 将输入张量按照指定模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将输入张量按照指定模式解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 使用均匀分布初始化输入形状的张量
def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# Gumbel 分布采样
def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    reinmax = False,
    dim = -1,
    training = True
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (reinmax and not straight_through), 'reinmax can only be turned on if using straight through gumbel softmax'

    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot

    # 使用 ReinMax 提高二阶精度
    if reinmax:
        π0 = logits.softmax(dim = dim)
        π1 = (one_hot + (logits / temperature).softmax(dim = dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim = 1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim = dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot

# Laplace 平滑
def laplace_smoothing(x, n_categories, eps = 1e-5, dim = -1):
    denom = x.sum(dim = dim, keepdim = True)
    return (x + eps) / (denom + n_categories * eps)

# 从样本中随机抽取指定数量的向量
def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

# 批量从样本中随机抽取指定数量的向量
def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

# 在指定维度上填充形状
def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

# 多项式分布采样
def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)

# 收集所有进程的指定维度大小
def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    # 使用torch.stack将列表中的张量按照第0维度进行堆叠
    return torch.stack(all_sizes)
def all_gather_variably_sized(x, sizes, dim = 0):
    # 获取当前进程的排名
    rank = distributed.get_rank()
    # 初始化一个空列表用于存储所有进程的数据
    all_x = []

    # 遍历每个进程的数据大小
    for i, size in enumerate(sizes):
        # 如果当前进程是当前循环的进程，则直接使用原始数据x，否则创建一个新的空tensor
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        # 使用广播将数据传输到所有进程
        distributed.broadcast(t, src = i, async_op = True)
        # 将数据添加到列表中
        all_x.append(t)

    # 等待所有进程完成数据传输
    distributed.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    # 重新排列本地样本数据的维度
    local_samples = rearrange(local_samples, '1 ... -> ...')

    # 获取当前进程的排名
    rank = distributed.get_rank()
    # 获取所有进程的样本数量
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    # 如果当前进程是主进程
    if rank == 0:
        # 对所有进程的样本数量进行多项式采样
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        # 创建一个与所有进程样本数量相同的空tensor
        samples_per_rank = torch.empty_like(all_num_samples)

    # 使用广播将采样结果传输到所有进程
    distributed.broadcast(samples_per_rank, src = 0)
    # 将tensor转换为列表
    samples_per_rank = samples_per_rank.tolist()

    # 对本地样本进行采样
    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    # 将所有进程的样本数据按照不同大小进行收集
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    # 拼接所有进程的样本数据
    out = torch.cat(all_samples, dim = 0)

    return rearrange(out, '... -> 1 ...')

def batched_bincount(x, *, minlength):
    # 获取batch大小、数据类型和设备信息
    batch, dtype, device = x.shape[0], x.dtype, x.device
    # 初始化一个全零tensor用于存储结果
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    # 初始化一个全一tensor
    values = torch.ones_like(x)
    # 对目标tensor进行scatter_add操作
    target.scatter_add_(-1, x, values)
    return target

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    # 获取��本数据的维度、数据类型和设备信息
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    # 使用指定函数对样本数据进行采样得到初始均值
    means = sample_fn(samples, num_clusters)

    # 迭代更新均值
    for _ in range(num_iters):
        # 计算样本数据与均值之间的距离
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -cdist(samples, means)

        # 将样本分配到最近的均值点
        buckets = torch.argmax(dists, dim = -1)
        # 对分配结果进行统计
        bins = batched_bincount(buckets, minlength = num_clusters)
        # 对统计结果进行全局归约
        all_reduce_fn(bins)

        # 处理空簇
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        # 计算新的均值
        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        # 对新的均值进行归一化
        if use_cosine_sim:
            new_means = l2norm(new_means)

        # 更新均值
        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins

def batched_embedding(indices, embeds):
    # 获取batch大小和嵌入维度
    batch, dim = indices.shape[1], embeds.shape[-1]
    # 将索引数据扩展到与嵌入数据相同的维度
    indices = repeat(indices, 'h b n -> h b n d', d = dim)
    # 将嵌入数据扩展到与索引数据相同的维度
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    # 根据索引获取对应的嵌入数据
    return embeds.gather(2, indices)

# regularization losses

def orthogonal_loss_fn(t):
    # 计算正交损失
    # 参考论文中的公式(2)
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)

# distance types

class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True,
        affine_param = False,
        sync_affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9
    ):
        # 调用父类的构造函数
        super().__init__()
        # 设置输入变换函数为恒等映射
        self.transform_input = identity

        # 设置衰减率和指数移动平均更新标志
        self.decay = decay
        self.ema_update = ema_update

        # 根据是否使用 kmeans 初始化选择初始化函数
        init_fn = uniform_init if not kmeans_init else torch.zeros
        # 初始化嵌入矩阵
        embed = init_fn(num_codebooks, codebook_size, dim)

        # 设置码书大小和码书数量
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        # 设置 kmeans 迭代次数和阈值
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        # 确保 gumbel_sample 是可调用的
        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        # 检查是否在分布式环境中使用 kmeans 初始化
        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'

        # 根据是否使用分布式和同步 kmeans 选择采样函数
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        # 注册缓冲区
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        # 设置是否可学习码书
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        # 仿射相关参数

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        # 设置仿射参数的衰减率
        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        # 注册缓冲区
        self.register_buffer('batch_mean', None)
        self.register_buffer('batch_variance', None)

        self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
        self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        # 如果已经初始化，则直接返回
        if self.initted:
            return

        # 如果存在掩码，则重新排列数据
        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        # 使用 kmeans 初始化 embed 和 cluster_size
        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        # 更新 embed 和 cluster_size
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        # 获取旧值
        old_value = getattr(self, buffer_name)

        # 获取是否需要初始化的标志
        needs_init = getattr(self, buffer_name + "_needs_init", False)

        # 如果需要初始化，则更新标志
        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        # 如果旧值不存在或需要初始化，则注册新值
        if not exists(old_value) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        # 更新值
        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    # 更新仿射变换参数，根据输入数据和嵌入向量，可选地使用掩码
    def update_affine(self, data, embed, mask = None):
        # 断言仿射参数已存在
        assert self.affine_param

        # 创建一个偏函数，用于计算方差
        var_fn = partial(torch.var, unbiased = False)

        # 计算码书均值和方差
        embed = rearrange(embed, 'h ... d -> h (...) d')

        # 如果处于训练模式
        if self.training:
            # 使用指数衰减更新码书均值
            self.update_with_decay('codebook_mean', reduce(embed, 'h n d -> h 1 d', 'mean'), self.affine_param_codebook_decay)
            # 使用指数衰减更新码书方差
            self.update_with_decay('codebook_variance', reduce(embed, 'h n d -> h 1 d', var_fn), self.affine_param_codebook_decay)

        # 准备批量数据，取决于是否有掩码
        data = rearrange(data, 'h ... d -> h (...) d')

        # 如果存在掩码
        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        # 计算批量均值和方差
        if not self.sync_affine_param:
            # 如果不同步仿射参数，使用指数衰减更新批量均值和方差
            self.update_with_decay('batch_mean', reduce(data, 'h n d -> h 1 d', 'mean'), self.affine_param_batch_decay)
            self.update_with_decay('batch_variance', reduce(data, 'h n d -> h 1 d', var_fn), self.affine_param_batch_decay)
            return

        # 计算分布式均值和方差
        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

        # 计算向量数量，用作分母
        num_vectors = torch.tensor([num_vectors], device = device, dtype = dtype)
        distributed.all_reduce(num_vectors)

        # 计算分布式均值
        batch_sum = reduce(data, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(batch_sum)
        batch_mean = batch_sum / num_vectors

        self.update_with_decay('batch_mean', batch_mean, self.affine_param_batch_decay)

        # 计算分布式方差
        variance_numer = reduce((data - batch_mean) ** 2, 'h n d -> h 1 d', 'sum')
        distributed.all_reduce(variance_numer)
        batch_variance = variance_numer / num_vectors

        self.update_with_decay('batch_variance', batch_variance, self.affine_param_batch_decay)

    # 替换过期的码字
    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0)):
            if not torch.any(mask):
                continue

            # 从样本中采样新的码字
            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')
            
            # 替换过期的码字
            self.embed.data[ind][mask] = sampled

            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    # 过期码字
    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        # 检查哪些码字过期
        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    # 前向传播函数
    @autocast(enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
        ):
            # 检查输入张量的维度是否小于4
            needs_codebook_dim = x.ndim < 4
            # 如果sample_codebook_temp未指定，则使用默认值
            sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

            # 将输入张量转换为浮点型
            x = x.float()

            # 如果需要增加codebook的维度
            if needs_codebook_dim:
                x = rearrange(x, '... -> 1 ...')

            # 获取输入张量的数据类型
            dtype = x.dtype
            # 将输入张量打包成一维数组，并返回打包后的数组和打包参数ps
            flatten, ps = pack_one(x, 'h * d')

            # 如果存在mask，则重复mask以匹配flatten的形状
            if exists(mask):
                mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

            # 初始化嵌入层
            self.init_embed_(flatten, mask = mask)

            # 如果使用仿射参数
            if self.affine_param:
                # 更新仿射参数
                self.update_affine(flatten, self.embed, mask = mask)

            # 获取嵌入层，如果不可学习则使用detach
            embed = self.embed if self.learnable_codebook else self.embed.detach()

            # 如果使用仿射参数
            if self.affine_param:
                # 计算codebook的标准差和批次的标准差
                codebook_std = self.codebook_variance.clamp(min = 1e-5).sqrt()
                batch_std = self.batch_variance.clamp(min = 1e-5).sqrt()
                # 对嵌入层进行仿射变换
                embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

            # 计算输入张量和嵌入层之间的距离
            dist = -cdist(flatten, embed)

            # 使用Gumbel采样获取嵌入层索引和独热编码
            embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)

            # 解包嵌入层索引
            embed_ind = unpack_one(embed_ind, ps, 'h *')

            # 如果处于训练状态
            if self.training:
                # 解包独热编码
                unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
                # 量化操作
                quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
            else:
                # 批量嵌入操作
                quantize = batched_embedding(embed_ind, embed)

            # 如果处于训练状态且需要EMA更新且未冻结codebook
            if self.training and self.ema_update and not freeze_codebook:

                # 如果使用仿射参数
                if self.affine_param:
                    # 对输入张量进行仿射变换
                    flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

                # 如果存在mask，则将未被mask覆盖的部分置零
                if exists(mask):
                    embed_onehot[~mask] = 0.

                # 计算聚类大小
                cluster_size = embed_onehot.sum(dim = 1)

                # 全局归约操作
                self.all_reduce_fn(cluster_size)
                # EMA更新聚类大小
                ema_inplace(self.cluster_size.data, cluster_size, self.decay)

                # 计算嵌入层总和
                embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
                embed_sum = embed_sum.contiguous()
                # 全局归约操作
                self.all_reduce_fn(embed_sum)

                # EMA更新嵌入层平均值
                ema_inplace(self.embed_avg.data, embed_sum, self.decay)

                # 对聚类大小进行拉普拉斯平滑
                cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

                # 归一化嵌入层
                embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
                self.embed.data.copy_(embed_normalized)
                # 清除过时的code
                self.expire_codes_(x)

            # 如果需要增加codebook的维度
            if needs_codebook_dim:
                quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

            # 解包距离
            dist = unpack_one(dist, ps, 'h * d')

            # 返回量化结果、嵌入层索引和距离
            return quantize, embed_ind, dist
class CosineSimCodebook(nn.Module):
    # 定义一个继承自 nn.Module 的类 CosineSimCodebook
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True
    ):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数

        self.transform_input = l2norm
        # 设置 transform_input 为 l2norm 函数

        self.ema_update = ema_update
        self.decay = decay
        # 设置 ema_update 和 decay 的值

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)
        # 根据 kmeans_init 的值初始化 embed

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        # 设置 codebook_size 和 num_codebooks 的值

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)
        # 设置 kmeans_iters、eps、threshold_ema_dead_code 和 reset_cluster_size 的值

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp
        # 断言 gumbel_sample 是可调用的，设置 gumbel_sample 和 sample_codebook_temp 的值

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        # 根据 use_ddp 和 sync_kmeans 的值选择 sample_fn、kmeans_all_reduce_fn 和 all_reduce_fn 的函数

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())
        # 注册缓冲区 initted、cluster_size 和 embed_avg

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)
        # 设置 learnable_codebook 和 embed

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        # 定义一个忽略 Torch JIT 的函数 init_embed_
        if self.initted:
            return
        # 如果已经初始化过，则直接返回

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)
        # 如果 mask 存在，则重新排列数据

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )
        # 使用 kmeans 函数初始化 embed 和 cluster_size

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')
        # 计算 embed_sum

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
        # 复制数据到相应的缓���区

    def replace(self, batch_samples, batch_mask):
        # 定义一个替换函数 replace
        batch_samples = l2norm(batch_samples)
        # 对 batch_samples 进行 l2norm 处理

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0)):
            # 遍历 batch_samples 和 batch_mask
            if not torch.any(mask):
                continue
            # 如果 mask 中没有任何元素，则继续下一次循环

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')
            # 对样本进行采样和重新排列

            self.embed.data[ind][mask] = sampled
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size
            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            # 更新 embed、embed_avg 和 cluster_size

    def expire_codes_(self, batch_samples):
        # 定义一个过期代码的函数 expire_codes_
        if self.threshold_ema_dead_code == 0:
            return
        # 如果阈值为 0，则直接返回

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        # 计算过期代码

        if not torch.any(expired_codes):
            return
        # 如果没有过期代码，则直接返回

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)
        # 重新排列 batch_samples 并调用 replace 函数

    @autocast(enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
        # 定义前向传播函数 forward，接受多个参数
        ):
        # 检查输入张量的维度是否小于4，如果是则需要添加一个维度
        needs_codebook_dim = x.ndim < 4
        # 如果未指定sample_codebook_temp，则使用默认值
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        # 将输入张量转换为浮点型
        x = x.float()

        # 如果需要添加一个维度，则重新排列张量
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        # 获取输入张量的数据类型
        dtype = x.dtype

        # 将输入张量打包成一维数组，并返回打包后的数组和打包方案
        flatten, ps = pack_one(x, 'h * d')

        # 如果存在掩码，则重复掩码以匹配打包后的数组形状
        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        # 初始化嵌入层，传入打包后的数组和掩码
        self.init_embed_(flatten, mask = mask)

        # 如果学习可学习码书，则使用可学习码书，否则使用固定码书
        embed = self.embed if self.learnable_codebook else self.embed.detach()

        # 计算嵌入距离
        dist = einsum('h n d, h c d -> h n c', flatten, embed)

        # 使用Gumbel采样获取嵌入索引和独热编码
        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)
        # 解包嵌入索引
        embed_ind = unpack_one(embed_ind, ps, 'h *')

        # 如果处于训练状态
        if self.training:
            # 解包独热编码
            unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
            # 量化操作
            quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
        else:
            # 使用批量嵌入获取量化结果
            quantize = batched_embedding(embed_ind, embed)

        # 如果处于训练状态且需要EMA更新且未冻结码书
        if self.training and self.ema_update and not freeze_codebook:
            # 如果存在掩码，则将掩码外的元素置零
            if exists(mask):
                embed_onehot[~mask] = 0.

            # 计算码书中每个码字的数量
            bins = embed_onehot.sum(dim = 1)
            self.all_reduce_fn(bins)

            # 更新EMA
            ema_inplace(self.cluster_size.data, bins, self.decay)

            # 计算码书的均值
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            # 更新EMA
            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            # 对码书大小进行Laplace平滑
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

            # 归一化嵌入向量
            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            # 更新嵌入层参数
            self.embed.data.copy_(l2norm(embed_normalized))
            # 清除过时码字
            self.expire_codes_(x)

        # 如果需要添加一个维度，则重新排列量化结果和嵌入索引
        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        # 解包嵌入距离
        dist = unpack_one(dist, ps, 'h * d')
        # 返回量化结果、嵌入索引和嵌入距离
        return quantize, embed_ind, dist
# 主类

class VectorQuantize(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,  # 输入向量的维度
        codebook_size,  # 量化码书的大小
        codebook_dim = None,  # 量化码书的维度，默认为None
        heads = 1,  # 多头注意力机制中的头数，默认为1
        separate_codebook_per_head = False,  # 每个头是否有独立的码书，默认为False
        decay = 0.8,  # 指数移动平均的衰减率，默认为0.8
        eps = 1e-5,  # 用于数值稳定性的小值，默认为1e-5
        freeze_codebook = False,  # 是否冻结码书，默认为False
        kmeans_init = False,  # 是否使用K均值初始化码书，默认为False
        kmeans_iters = 10,  # K均值初始化码书的迭代次数，默认为10
        sync_kmeans = True,  # 是否同步K均值初始化码书，默认为True
        use_cosine_sim = False,  # 是否使用余弦相似度，默认为False
        threshold_ema_dead_code = 0,  # EMA更新码书时的阈值，默认为0
        channel_last = True,  # 是否使用通道最后的数据格式，默认为True
        accept_image_fmap = False,  # 是否接受图像特征图，默认为False
        commitment_weight = 1.,  # 量化损失的权重，默认为1.0
        commitment_use_cross_entropy_loss = False,  # 是否使用交叉熵损失，默认为False
        orthogonal_reg_weight = 0.,  # 正交正则化的权重，默认为0.0
        orthogonal_reg_active_codes_only = False,  # 是否只对激活码进行正交正则化，默认为False
        orthogonal_reg_max_codes = None,  # 正交正则化的最大码书数量，默认为None
        stochastic_sample_codes = False,  # 是否随机采样码书，默认为False
        sample_codebook_temp = 1.,  # 采样码书时的温度参数，默认为1.0
        straight_through = False,  # 是否使用直通梯度传播，默认为False
        reinmax = False,  # 是否使用reinmax来改进直通梯度传播，默认为False
        sync_codebook = None,  # 同步更新码书的规则，默认为None
        sync_affine_param = False,  # 是否同步更新仿射参数，默认为False
        ema_update = True,  # 是否使用EMA更新码书，默认为True
        learnable_codebook = False,  # 是否可学习码书，默认为False
        in_place_codebook_optimizer: Callable[..., Optimizer] = None,  # 用于更新可学习码书的优化器，默认为None
        affine_param = False,  # 是否使用仿射参数，默认为False
        affine_param_batch_decay = 0.99,  # 仿射参数的批次衰减率，默认为0.99
        affine_param_codebook_decay = 0.9,  # 仿射参数的码书衰减率，默认为0.9
        sync_update_v = 0.  # ���制同步更新规则中乐观与悲观更新的参数v，默认为0.0
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化模型的维度
        self.dim = dim
        # 初始化头数
        self.heads = heads
        # 是否为每个头单独使用码书
        self.separate_codebook_per_head = separate_codebook_per_head

        # 设置码书维度，默认为模型维度
        codebook_dim = default(codebook_dim, dim)
        # 计算码书输入维度
        codebook_input_dim = codebook_dim * heads

        # 判断是否需要投影
        requires_projection = codebook_input_dim != dim
        # 如果需要投影，则使用线性层进行投影，否则使用恒等映射
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        # 记录是否有投影
        self.has_projections = requires_projection

        # 设置 epsilon
        self.eps = eps
        # 设置码书权重
        self.commitment_weight = commitment_weight
        # 是否使用交叉熵损失作为码书的约束损失
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss

        # 是否可学习的码书
        self.learnable_codebook = learnable_codebook

        # 是否有码书正交损失
        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        # 检查是否EMA更新和可学习码书不兼容
        assert not (ema_update and learnable_codebook), 'learnable codebook not compatible with EMA update'

        # 检查同步更新参数的范围
        assert 0 <= sync_update_v <= 1.
        assert not (sync_update_v > 0. and not learnable_codebook), 'learnable codebook must be turned on'

        # 设置同步更新参数
        self.sync_update_v = sync_update_v

        # 根据是否使用余弦相似度选择码书类
        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        # 部分应用函数，用于生成 Gumbel 样本
        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic = stochastic_sample_codes,
            reinmax = reinmax,
            straight_through = straight_through
        )

        # 如果未提供同步码书，则根据分布式环境设置同步码书
        if not exists(sync_codebook):
            sync_codebook = distributed.is_initialized() and distributed.get_world_size() > 1

        # 设置码书参数
        codebook_kwargs = dict(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp = sample_codebook_temp,
            gumbel_sample = gumbel_sample_fn,
            ema_update = ema_update
        )

        # 如果使用仿射参数，则更新码书参数
        if affine_param:
            assert not use_cosine_sim, 'affine param is only compatible with euclidean codebook'
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param = True,
                sync_affine_param = sync_affine_param,
                affine_param_batch_decay = affine_param_batch_decay,
                affine_param_codebook_decay = affine_param_codebook_decay,
            )

        # 初始化码书对象
        self._codebook = codebook_class(**codebook_kwargs)

        # 如果存在码书优化器，则初始化
        self.in_place_codebook_optimizer = in_place_codebook_optimizer(self._codebook.parameters()) if exists(in_place_codebook_optimizer) else None

        # 设置码书大小
        self.codebook_size = codebook_size

        # 是否接受图像特征图
        self.accept_image_fmap = accept_image_fmap
        # 是否通道在最后
        self.channel_last = channel_last

    @property
    def codebook(self):
        # 获取码书
        codebook = self._codebook.embed

        # 如果每个头单独使用码书，则直接返回码书
        if self.separate_codebook_per_head:
            return codebook

        # 否则重新排列码书维度
        return rearrange(codebook, '1 ... -> ...')

    @codebook.setter
    def codebook(self, codes):
        # 如果不是每个头单独使用码书，则重新排列码书维度
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, '... -> 1 ...')

        # 将码书赋值给内部码书对象
        self._codebook.embed.copy_(codes)
    # 从给定的索引中获取对应的编码
    def get_codes_from_indices(self, indices):
        # 获取编码簿
        codebook = self.codebook
        # 判断是否为多头编码
        is_multiheaded = codebook.ndim > 2

        # 如果不是多头编码
        if not is_multiheaded:
            # 从编码簿中获取对应索引的编码
            codes = codebook[indices]
        else:
            # 打包索引
            indices, ps = pack_one(indices, 'b * h')
            # 重新排列索引
            indices = rearrange(indices, 'b n h -> b h n')

            # 重复索引
            indices = repeat(indices, 'b h n -> b h n d', d = codebook.shape[-1])
            # 重复编码簿
            codebook = repeat(codebook, 'h n d -> b h n d', b = indices.shape[0])

            # 从编码簿中收集编码
            codes = codebook.gather(2, indices)
            # 重新排列编码
            codes = rearrange(codes, 'b h n d -> b n (h d)')
            # 解包编码
            codes = unpack_one(codes, ps, 'b * d')

        # 如果不是通道在最后
        if not self.channel_last:
            # 重新排列编码
            codes = rearrange(codes, 'b ... d -> b d ...')

        # 返回编码
        return codes

    # 从给定的索引中获取输出
    def get_output_from_indices(self, indices):
        # 获取编码
        codes = self.get_codes_from_indices(indices)
        # 对编码进行投影
        return self.project_out(codes)

    # 前向传播函数
    def forward(
        self,
        x,
        indices = None,
        mask = None,
        sample_codebook_temp = None,
        freeze_codebook = False
```