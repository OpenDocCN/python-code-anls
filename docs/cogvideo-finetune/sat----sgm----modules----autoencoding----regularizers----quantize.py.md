# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\regularizers\quantize.py`

```py
# 导入 logging 模块以便进行日志记录
import logging
# 从 abc 模块导入 abstractmethod，用于定义抽象方法
from abc import abstractmethod
# 从 typing 模块导入多种类型提示
from typing import Dict, Iterator, Literal, Optional, Tuple, Union

# 导入 numpy 库并命名为 np
import numpy as np
# 导入 PyTorch 库及其子模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 从 einops 导入 rearrange 函数，用于重排张量
from einops import rearrange
# 从 torch 导入 einsum 函数，用于张量操作
from torch import einsum

# 从同一包中导入 AbstractRegularizer 类和 measure_perplexity 函数
from .base import AbstractRegularizer, measure_perplexity

# 创建一个 logger 实例，用于当前模块的日志记录
logpy = logging.getLogger(__name__)


# 定义一个抽象量化器类，继承自 AbstractRegularizer
class AbstractQuantizer(AbstractRegularizer):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 在初始化时定义这些属性
        # shape (N,) 表示该张量的形状为一维
        self.used: Optional[torch.Tensor]  # 定义已使用的张量，可能为 None
        self.re_embed: int  # 定义重嵌入的整数值
        self.unknown_index: Union[Literal["random"], int]  # 定义未知索引，可能为随机或整数

    # 将输入索引映射到已使用的索引
    def remap_to_used(self, inds: torch.Tensor) -> torch.Tensor:
        # 确保已定义 used 索引
        assert self.used is not None, "You need to define used indices for remap"
        ishape = inds.shape  # 获取输入索引的形状
        assert len(ishape) > 1  # 确保输入维度大于 1
        inds = inds.reshape(ishape[0], -1)  # 重塑输入索引为二维
        used = self.used.to(inds)  # 将 used 张量移动到与 inds 相同的设备
        match = (inds[:, :, None] == used[None, None, ...]).long()  # 计算索引匹配情况
        new = match.argmax(-1)  # 找到每个匹配的最大索引
        unknown = match.sum(2) < 1  # 标记未知索引
        # 如果未知索引为随机，则随机生成新的索引
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index  # 将未知索引设置为指定的未知索引
        return new.reshape(ishape)  # 返回重塑后的新索引

    # 将输入索引映射回所有索引
    def unmap_to_all(self, inds: torch.Tensor) -> torch.Tensor:
        # 确保已定义 used 索引
        assert self.used is not None, "You need to define used indices for remap"
        ishape = inds.shape  # 获取输入索引的形状
        assert len(ishape) > 1  # 确保输入维度大于 1
        inds = inds.reshape(ishape[0], -1)  # 重塑输入索引为二维
        used = self.used.to(inds)  # 将 used 张量移动到与 inds 相同的设备
        # 如果重嵌入数量大于已使用数量，则处理额外的令牌
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # 将超出范围的索引设置为零
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)  # 根据输入索引收集数据
        return back.reshape(ishape)  # 返回重塑后的数据

    # 定义抽象方法以获取编码表条目
    @abstractmethod
    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        raise NotImplementedError()  # 抛出未实现错误

    # 获取可训练参数的迭代器
    def get_trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self.parameters()  # 生成模型参数


# 定义 Gumbel 量化器类，继承自 AbstractQuantizer
class GumbelQuantizer(AbstractQuantizer):
    """
    credit to @karpathy:
    https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    # 初始化方法
    def __init__(
        self,
        num_hiddens: int,  # 隐藏层单元数
        embedding_dim: int,  # 嵌入维度
        n_embed: int,  # 嵌入数量
        straight_through: bool = True,  # 是否使用直通梯度
        kl_weight: float = 5e-4,  # KL 散度的权重
        temp_init: float = 1.0,  # 初始化温度
        remap: Optional[str] = None,  # 可选的重映射方式
        unknown_index: str = "random",  # 未知索引的默认值为随机
        loss_key: str = "loss/vq",  # 损失键的默认值
    # 定义一个返回 None 的方法
        ) -> None:
            # 调用父类的构造函数
            super().__init__()
    
            # 保存损失的关键字
            self.loss_key = loss_key
            # 设置嵌入维度
            self.embedding_dim = embedding_dim
            # 设置嵌入数量
            self.n_embed = n_embed
    
            # 设置是否使用直通估计
            self.straight_through = straight_through
            # 初始化温度参数
            self.temperature = temp_init
            # 设置 KL 散度权重
            self.kl_weight = kl_weight
    
            # 创建一个 2D 卷积层，输入通道为 num_hiddens，输出通道为 n_embed，卷积核大小为 1
            self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
            # 创建嵌入层，嵌入数量为 n_embed，嵌入维度为 embedding_dim
            self.embed = nn.Embedding(n_embed, embedding_dim)
    
            # 保存重映射文件路径
            self.remap = remap
            # 如果提供了重映射
            if self.remap is not None:
                # 从重映射文件中加载使用的索引，并将其注册为缓冲区
                self.register_buffer("used", torch.tensor(np.load(self.remap)))
                # 设置重嵌入数量为使用的索引的数量
                self.re_embed = self.used.shape[0]
            else:
                # 如果未提供重映射，则使用全部嵌入数量
                self.used = None
                self.re_embed = n_embed
            # 如果未知索引设置为 "extra"
            if unknown_index == "extra":
                # 将未知索引设置为重嵌入数量，并增加一个
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            else:
                # 断言未知索引必须为 "random"、"extra" 或整数
                assert unknown_index == "random" or isinstance(
                    unknown_index, int
                ), "unknown index needs to be 'random', 'extra' or any integer"
                # 设置未知索引
                self.unknown_index = unknown_index  # "random" or "extra" or integer
            # 如果提供了重映射，则记录相关信息
            if self.remap is not None:
                logpy.info(
                    f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                    f"Using {self.unknown_index} for unknown indices."
                )
    
        # 定义前向传播方法，接收输入张量 z 和可选的温度参数
        def forward(
            self, z: torch.Tensor, temp: Optional[float] = None, return_logits: bool = False
        ) -> Tuple[torch.Tensor, Dict]:
            # 在评估模式下强制 hard=True，因为必须进行量化。
            # 实际上，始终为真似乎也有效
            hard = self.straight_through if self.training else True
            # 设置温度，如果未提供，则使用默认值
            temp = self.temperature if temp is None else temp
            # 初始化输出字典
            out_dict = {}
            # 通过卷积层计算 logits
            logits = self.proj(z)
            # 如果提供了重映射
            if self.remap is not None:
                # 创建与 logits 同样形状的全零张量
                full_zeros = torch.zeros_like(logits)
                # 仅保留使用的 logits
                logits = logits[:, self.used, ...]
    
            # 使用 Gumbel-Softmax 函数生成软 one-hot 编码
            soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
            # 如果提供了重映射
            if self.remap is not None:
                # 将未使用的条目设置为零
                full_zeros[:, self.used, ...] = soft_one_hot
                soft_one_hot = full_zeros
            # 根据软 one-hot 编码和嵌入权重计算量化的 z
            z_q = einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)
    
            # 计算 KL 散度损失
            qy = F.softmax(logits, dim=1)
            # 计算散度并取平均
            diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
            # 将散度损失存储到输出字典中
            out_dict[self.loss_key] = diff
    
            # 计算 soft_one_hot 编码的最大索引
            ind = soft_one_hot.argmax(dim=1)
            # 如果提供了重映射，将索引转换为使用的索引
            if self.remap is not None:
                ind = self.remap_to_used(ind)
    
            # 如果需要返回 logits，则将其存储到输出字典中
            if return_logits:
                out_dict["logits"] = logits
    
            # 返回量化的 z 和输出字典
            return z_q, out_dict
    # 获取代码本条目的方法，根据给定的索引和形状
    def get_codebook_entry(self, indices, shape):
        # TODO: 当前形状参数尚不可选
        b, h, w, c = shape  # 解包形状参数，获取批次、身高、宽度和通道数
        # 确保索引的总数与给定的形状匹配
        assert b * h * w == indices.shape[0]
        # 重排列索引，将其形状调整为 (b, h, w)
        indices = rearrange(indices, "(b h w) -> b h w", b=b, h=h, w=w)
        # 如果存在重映射，则将索引映射到所有可能的值
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        # 将索引转换为独热编码，调整维度顺序并转换为浮点数
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        # 通过爱因斯坦求和约定计算最终的量化表示 z_q
        z_q = einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        # 返回量化后的表示
        return z_q
# 定义向量量化类，继承自抽象量化器
class VectorQuantizer(AbstractQuantizer):
    """
    ____________________________________________
    VQ-VAE 的离散化瓶颈部分。
    输入:
    - n_e : 嵌入的数量
    - e_dim : 嵌入的维度
    - beta : 在损失项中使用的承诺成本,
        beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    # 初始化方法，定义类的参数
    def __init__(
        self,
        n_e: int,  # 嵌入的数量
        e_dim: int,  # 嵌入的维度
        beta: float = 0.25,  # 默认承诺成本
        remap: Optional[str] = None,  # 可选的重映射文件路径
        unknown_index: str = "random",  # 未知索引的处理方式
        sane_index_shape: bool = False,  # 是否保持合理的索引形状
        log_perplexity: bool = False,  # 是否记录困惑度
        embedding_weight_norm: bool = False,  # 是否使用权重归一化
        loss_key: str = "loss/vq",  # 损失的键
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存嵌入数量
        self.n_e = n_e
        # 保存嵌入维度
        self.e_dim = e_dim
        # 保存承诺成本
        self.beta = beta
        # 保存损失键
        self.loss_key = loss_key

        # 如果不使用权重归一化
        if not embedding_weight_norm:
            # 初始化嵌入层，权重范围均匀分布
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            # 使用权重归一化的嵌入层
            self.embedding = torch.nn.utils.weight_norm(nn.Embedding(self.n_e, self.e_dim), dim=1)

        # 保存重映射参数
        self.remap = remap
        # 如果指定了重映射
        if self.remap is not None:
            # 从重映射文件中加载已使用的索引
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            # 设置重新嵌入的数量
            self.re_embed = self.used.shape[0]
        else:
            # 否则未使用的索引为 None
            self.used = None
            # 重新嵌入的数量为 n_e
            self.re_embed = n_e
        # 如果未知索引是 "extra"
        if unknown_index == "extra":
            # 设置未知索引为重新嵌入的数量
            self.unknown_index = self.re_embed
            # 重新嵌入的数量加一
            self.re_embed = self.re_embed + 1
        else:
            # 确保未知索引是 "random"、"extra" 或整数
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            # 保存未知索引的值
            self.unknown_index = unknown_index  # "random" 或 "extra" 或整数
        # 如果指定了重映射，记录信息
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

        # 保存是否保持合理的索引形状的标志
        self.sane_index_shape = sane_index_shape
        # 保存是否记录困惑度的标志
        self.log_perplexity = log_perplexity

    # 前向传播方法，定义输入的处理
    def forward(
        self,
        z: torch.Tensor,  # 输入张量
    ) -> Tuple[torch.Tensor, Dict]:  # 定义返回类型为元组，包含一个张量和一个字典
        do_reshape = z.ndim == 4  # 检查 z 的维度是否为 4，决定是否需要重塑
        if do_reshape:  # 如果 z 是 4 维的
            # reshape z -> (batch, height, width, channel) and flatten  # 重塑 z 的维度为 (batch, height, width, channel) 并扁平化
            z = rearrange(z, "b c h w -> b h w c").contiguous()  # 重新排列 z 的维度，并保证内存连续性

        else:  # 如果 z 不是 4 维的
            assert z.ndim < 4, "No reshaping strategy for inputs > 4 dimensions defined"  # 断言 z 的维度小于 4
            z = z.contiguous()  # 确保 z 的内存是连续的

        z_flattened = z.view(-1, self.e_dim)  # 将 z 重塑为 (batch_size, e_dim) 的形状
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z  # 计算 z 到嵌入 e_j 的距离

        d = (  # 计算每个 z 到嵌入的距离
            torch.sum(z_flattened**2, dim=1, keepdim=True)  # 计算 z_flattened 的平方和
            + torch.sum(self.embedding.weight**2, dim=1)  # 计算嵌入权重的平方和
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))  # 计算 z_flattened 与嵌入的内积
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # 找到每个 z 到嵌入距离的最小索引
        z_q = self.embedding(min_encoding_indices).view(z.shape)  # 根据最小索引获取量化的嵌入，并重塑为原始 z 的形状
        loss_dict = {}  # 初始化损失字典
        if self.log_perplexity:  # 如果需要记录困惑度
            perplexity, cluster_usage = measure_perplexity(min_encoding_indices.detach(), self.n_e)  # 计算困惑度和集群使用情况
            loss_dict.update({"perplexity": perplexity, "cluster_usage": cluster_usage})  # 更新损失字典

        # compute loss for embedding  # 计算嵌入的损失
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)  # 计算损失值
        loss_dict[self.loss_key] = loss  # 将损失添加到损失字典中

        # preserve gradients  # 保留梯度
        z_q = z + (z_q - z).detach()  # 将量化的 z_q 与原始 z 结合，保留梯度

        # reshape back to match original input shape  # 重新调整形状以匹配原始输入形状
        if do_reshape:  # 如果之前进行了重塑
            z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()  # 将 z_q 的维度调整回 (batch, channel, height, width)

        if self.remap is not None:  # 如果需要重映射
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # 添加批次维度
            min_encoding_indices = self.remap_to_used(min_encoding_indices)  # 对最小编码索引进行重映射
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # 扁平化为一维

        if self.sane_index_shape:  # 如果索引形状正常
            if do_reshape:  # 如果之前进行了重塑
                min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])  # 将索引重塑为与 z_q 形状相匹配
            else:  # 如果没有重塑
                min_encoding_indices = rearrange(min_encoding_indices, "(b s) 1 -> b s", b=z_q.shape[0])  # 重新排列为 (batch, size)

        loss_dict["min_encoding_indices"] = min_encoding_indices  # 将最小编码索引添加到损失字典中

        return z_q, loss_dict  # 返回量化的 z_q 和损失字典

    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:  # 定义方法获取代码本条目
        # shape specifying (batch, height, width, channel)  # shape 指定 (batch, height, width, channel)
        if self.remap is not None:  # 如果需要重映射
            assert shape is not None, "Need to give shape for remap"  # 断言必须提供形状以进行重映射
            indices = indices.reshape(shape[0], -1)  # 添加批次维度
            indices = self.unmap_to_all(indices)  # 对索引进行反向映射
            indices = indices.reshape(-1)  # 再次扁平化

        # get quantized latent vectors  # 获取量化的潜在向量
        z_q = self.embedding(indices)  # 根据索引获取嵌入

        if shape is not None:  # 如果提供了形状
            z_q = z_q.view(shape)  # 将 z_q 重塑为指定的形状
            # reshape back to match original input shape  # 重新调整形状以匹配原始输入形状
            z_q = z_q.permute(0, 3, 1, 2).contiguous()  # 调整维度顺序并确保内存连续

        return z_q  # 返回量化后的 z_q
# 定义一个名为 EmbeddingEMA 的神经网络模块，继承自 nn.Module
class EmbeddingEMA(nn.Module):
    # 初始化函数，接收令牌数量、码本维度、衰减因子和小常数作为参数
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        # 调用父类的初始化方法
        super().__init__()
        # 设置衰减因子
        self.decay = decay
        # 设置小常数以避免除零
        self.eps = eps
        # 生成一个随机的权重矩阵，形状为 (num_tokens, codebook_dim)
        weight = torch.randn(num_tokens, codebook_dim)
        # 将权重定义为不可训练的参数
        self.weight = nn.Parameter(weight, requires_grad=False)
        # 初始化集群大小为零，定义为不可训练的参数
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        # 复制权重并将其定义为不可训练的参数，用于存储嵌入平均值
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        # 设置更新标志为真
        self.update = True

    # 前向传播函数，接收嵌入 ID 并返回对应的嵌入向量
    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    # 更新集群大小的指数移动平均
    def cluster_size_ema_update(self, new_cluster_size):
        # 按衰减因子更新当前集群大小
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    # 更新嵌入平均值的指数移动平均
    def embed_avg_ema_update(self, new_embed_avg):
        # 按衰减因子更新当前嵌入平均值
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    # 更新权重，基于平滑的集群大小
    def weight_update(self, num_tokens):
        # 计算集群大小的总和
        n = self.cluster_size.sum()
        # 计算平滑的集群大小，以避免除零错误
        smoothed_cluster_size = (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        # 用平滑的集群大小对嵌入平均值进行归一化
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        # 用归一化的嵌入更新权重
        self.weight.data.copy_(embed_normalized)


# 定义一个名为 EMAVectorQuantizer 的抽象量化器类，继承自 AbstractQuantizer
class EMAVectorQuantizer(AbstractQuantizer):
    # 初始化函数，接收嵌入数量、嵌入维度、β、衰减因子、小常数和其他参数
    def __init__(
        self,
        n_embed: int,
        embedding_dim: int,
        beta: float,
        decay: float = 0.99,
        eps: float = 1e-5,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        loss_key: str = "loss/vq",
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置码本维度
        self.codebook_dim = embedding_dim
        # 设置令牌数量
        self.num_tokens = n_embed
        # 设置 β 值
        self.beta = beta
        # 设置损失键
        self.loss_key = loss_key

        # 初始化嵌入 EMA 模块
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

        # 处理重映射参数
        self.remap = remap
        if self.remap is not None:
            # 如果提供了重映射路径，则加载并注册重映射后的索引
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            # 重新嵌入的数量为重映射后索引的形状
            self.re_embed = self.used.shape[0]
        else:
            # 如果没有重映射，则用原令牌数量初始化
            self.used = None
            self.re_embed = n_embed
        # 处理未知索引
        if unknown_index == "extra":
            # 如果未知索引为 "extra"，则更新重新嵌入数量
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            # 确保未知索引为有效类型
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            # 设置未知索引为提供的值
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        # 如果存在重映射，则记录重映射信息
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
    # 定义前向传播函数，接受一个张量 z，返回量化后的张量和字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # 将 z 的形状调整为 (batch, height, width, channel) 并扁平化
        # z, 'b c h w -> b h w c'
        z = rearrange(z, "b c h w -> b h w c")  # 调整 z 的维度顺序
        z_flattened = z.reshape(-1, self.codebook_dim)  # 将 z 扁平化为二维张量

        # 计算 z 与嵌入 e_j 的距离 (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            z_flattened.pow(2).sum(dim=1, keepdim=True)  # 计算 z 的平方和
            + self.embedding.weight.pow(2).sum(dim=1)  # 计算嵌入的平方和
            - 2 * torch.einsum("bd,nd->bn", z_flattened, self.embedding.weight)  # 计算 z 和嵌入的点积
        )  # 'n d -> d n'

        # 找到每个 z 的最小距离对应的编码索引
        encoding_indices = torch.argmin(d, dim=1)

        # 根据编码索引获取量化后的 z，并调整形状以匹配原始 z
        z_q = self.embedding(encoding_indices).view(z.shape)  
        # 对编码进行独热编码，转换为与 z 相同的数据类型
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)  
        # 计算编码的平均概率
        avg_probs = torch.mean(encodings, dim=0)  
        # 计算困惑度
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))  

        # 如果处于训练状态且允许更新嵌入
        if self.training and self.embedding.update:
            # 更新 EMA 聚类大小
            encodings_sum = encodings.sum(0)  
            self.embedding.cluster_size_ema_update(encodings_sum)  # 更新聚类大小的 EMA
            # 更新 EMA 嵌入平均值
            embed_sum = encodings.transpose(0, 1) @ z_flattened  # 计算加权和
            self.embedding.embed_avg_ema_update(embed_sum)  # 更新嵌入平均值的 EMA
            # 规范化嵌入平均值并更新权重
            self.embedding.weight_update(self.num_tokens)  

        # 计算嵌入的损失
        loss = self.beta * F.mse_loss(z_q.detach(), z)  # 计算量化 z 与原 z 的均方误差损失

        # 保留梯度
        z_q = z + (z_q - z).detach()  # 使用 z 的值加上 z_q 的变化，但不计算梯度

        # 将 z_q 的形状调整回原始输入的形状
        # z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, "b h w c -> b c h w")  # 恢复到原始的维度顺序

        # 创建一个字典以返回损失和其他信息
        out_dict = {
            self.loss_key: loss,  # 将损失放入字典
            "encodings": encodings,  # 包含独热编码
            "encoding_indices": encoding_indices,  # 包含编码索引
            "perplexity": perplexity,  # 包含困惑度
        }

        # 返回量化后的 z 和输出字典
        return z_q, out_dict  
# 定义一个带有输入投影的向量量化类，继承自 VectorQuantizer
class VectorQuantizerWithInputProjection(VectorQuantizer):
    # 初始化方法，接受输入维度、编码数量、码本维度等参数
    def __init__(
        self,
        input_dim: int,  # 输入数据的维度
        n_codes: int,  # 编码数量
        codebook_dim: int,  # 码本的维度
        beta: float = 1.0,  # 调整项的超参数，默认值为1.0
        output_dim: Optional[int] = None,  # 输出维度，可选
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法
        super().__init__(n_codes, codebook_dim, beta, **kwargs)
        # 创建输入投影层，将输入维度映射到码本维度
        self.proj_in = nn.Linear(input_dim, codebook_dim)
        # 设置输出维度属性
        self.output_dim = output_dim
        # 如果指定了输出维度，则创建输出投影层
        if output_dim is not None:
            self.proj_out = nn.Linear(codebook_dim, output_dim)
        else:
            # 如果没有指定输出维度，则使用恒等映射
            self.proj_out = nn.Identity()

    # 前向传播方法，接受输入张量并返回量化结果和损失字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        rearr = False  # 初始化重排列标志
        in_shape = z.shape  # 获取输入张量的形状

        # 如果输入张量的维度大于3，则进行重排列
        if z.ndim > 3:
            rearr = self.output_dim is not None  # 检查是否需要重排列
            # 将输入张量从 (batch, channels, ...) 转换为 (batch, ..., channels)
            z = rearrange(z, "b c ... -> b (...) c")
        # 将输入张量投影到码本维度
        z = self.proj_in(z)
        # 调用父类的前向方法进行量化，获得量化结果和损失字典
        z_q, loss_dict = super().forward(z)

        # 将量化结果通过输出投影层
        z_q = self.proj_out(z_q)
        # 如果需要重排列，根据输入形状调整输出张量
        if rearr:
            # 如果输入维度为4，重排列为 (batch, channels, height, width)
            if len(in_shape) == 4:
                z_q = rearrange(z_q, "b (h w) c -> b c h w ", w=in_shape[-1])
            # 如果输入维度为5，重排列为 (batch, channels, time, height, width)
            elif len(in_shape) == 5:
                z_q = rearrange(z_q, "b (t h w) c -> b c t h w ", w=in_shape[-1], h=in_shape[-2])
            else:
                # 如果输入维度不支持重排列，则抛出异常
                raise NotImplementedError(f"rearranging not available for {len(in_shape)}-dimensional input.")

        # 返回量化结果和损失字典
        return z_q, loss_dict
```