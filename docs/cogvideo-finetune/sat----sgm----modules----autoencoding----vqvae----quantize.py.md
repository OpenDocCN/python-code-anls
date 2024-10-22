# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\quantize.py`

```py
# 导入 PyTorch 和其他必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange

# 定义一个改进版的向量量化器类，继承自 nn.Module
class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # 初始化方法，接受多个参数用于配置向量量化器
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True):
        # 调用父类构造函数
        super().__init__()
        # 设置向量数量
        self.n_e = n_e
        # 设置嵌入维度
        self.e_dim = e_dim
        # 设置 beta 值
        self.beta = beta
        # 设置是否使用旧版兼容性
        self.legacy = legacy

        # 创建一个嵌入层，尺寸为 (n_e, e_dim)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # 将嵌入权重初始化为均匀分布
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # 如果提供了 remap 参数
        self.remap = remap
        if self.remap is not None:
            # 从文件加载已使用的索引并注册为缓冲区
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            # 设置重新嵌入的大小
            self.re_embed = self.used.shape[0]
            # 设置未知索引，默认为 "random"
            self.unknown_index = unknown_index  # "random" 或 "extra" 或整数
            if self.unknown_index == "extra":
                # 如果未知索引是 "extra"，调整索引值
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            # 打印重映射信息
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            # 如果没有重映射，则重新嵌入大小等于 n_e
            self.re_embed = n_e

        # 设置是否使用合理的索引形状
        self.sane_index_shape = sane_index_shape

    # 将索引映射到已使用的索引
    def remap_to_used(self, inds):
        # 记录输入索引的形状
        ishape = inds.shape
        # 确保输入至少有两个维度
        assert len(ishape) > 1
        # 将索引重塑为 (批量大小, -1) 的形状
        inds = inds.reshape(ishape[0], -1)
        # 将已使用的索引移到当前设备
        used = self.used.to(inds)
        # 检查 inds 是否在 used 中匹配
        match = (inds[:, :, None] == used[None, None, ...]).long()
        # 找到匹配的最大值索引
        new = match.argmax(-1)
        # 查找未知索引
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            # 随机生成未知索引
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            # 用指定的未知索引填充未知位置
            new[unknown] = self.unknown_index
        # 返回重塑后的新索引
        return new.reshape(ishape)

    # 将已使用的索引映射回所有索引
    def unmap_to_all(self, inds):
        # 记录输入索引的形状
        ishape = inds.shape
        # 确保输入至少有两个维度
        assert len(ishape) > 1
        # 将索引重塑为 (批量大小, -1) 的形状
        inds = inds.reshape(ishape[0], -1)
        # 将已使用的索引移到当前设备
        used = self.used.to(inds)
        # 如果重新嵌入的大小大于已使用的索引数量，处理额外的标记
        if self.re_embed > self.used.shape[0]:  # extra token
            # 将超出范围的索引设为零
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        # 使用索引反向收集所有标记
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        # 返回重塑后的标记
        return back.reshape(ishape)
    # 前向传播函数，处理输入 z，选择温度和日志缩放参数
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        # 确保温度参数为 None 或 1.0，适用于 Gumbel 接口
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        # 确保日志缩放参数为 False，适用于 Gumbel 接口
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        # 确保返回日志参数为 False，适用于 Gumbel 接口
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # 将 z 变形为 (batch, height, width, channel) 并展平
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        # 将 z 展平为二维张量，形状为 (batch_size, embedding_dim)
        z_flattened = z.view(-1, self.e_dim)
        # 计算 z 到嵌入 e_j 的距离 (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            # 计算 z_flattened 的平方和，保留维度
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            # 加上嵌入权重的平方和
            + torch.sum(self.embedding.weight**2, dim=1)
            # 减去 2 * z_flattened 和嵌入权重的内积
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))
        )
        # 找到距离最近的编码索引
        min_encoding_indices = torch.argmin(d, dim=1)
        # 根据最小编码索引获取量化的 z，并调整为原始形状
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None  # 初始化困惑度为 None
        min_encodings = None  # 初始化最小编码为 None

        # 计算嵌入的损失
        if not self.legacy:
            # 计算损失，考虑 beta 和两个均方差项
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            # 计算损失，考虑 beta 和两个均方差项（顺序不同）
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # 保持梯度
        z_q = z + (z_q - z).detach()

        # 重新调整 z_q 的形状以匹配原始输入形状
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        if self.remap is not None:
            # 如果需要重映射，将最小编码索引展平并添加批次维度
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            # 使用重映射函数
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            # 再次展平
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            # 确保最小编码索引形状合理，重新调整形状
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        # 返回量化的 z、损失及其他信息
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    # 根据索引获取代码簿条目，返回形状为 (batch, height, width, channel)
    def get_codebook_entry(self, indices, shape):
        # shape 指定 (batch, height, width, channel)
        if self.remap is not None:
            # 如果需要重映射，展平索引并添加批次维度
            indices = indices.reshape(shape[0], -1)  # add batch axis
            # 使用反重映射函数
            indices = self.unmap_to_all(indices)
            # 再次展平
            indices = indices.reshape(-1)  # flatten again

        # 获取量化的潜在向量
        z_q = self.embedding(indices)

        if shape is not None:
            # 如果形状不为 None，重新调整 z_q 的形状
            z_q = z_q.view(shape)
            # 重新调整以匹配原始输入形状
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # 返回量化后的潜在向量
        return z_q
# 定义 GumbelQuantize 类，继承自 nn.Module
class GumbelQuantize(nn.Module):
    """
    归功于 @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (谢谢！)
    Gumbel Softmax 技巧量化器
    使用 Gumbel-Softmax 的分类重参数化，Jang 等人 2016
    https://arxiv.org/abs/1611.01144
    """

    # 初始化方法，定义类的属性
    def __init__(
        self,
        num_hiddens,  # 隐藏层的神经元数量
        embedding_dim,  # 嵌入维度
        n_embed,  # 嵌入的数量
        straight_through=True,  # 是否使用直通估计
        kl_weight=5e-4,  # KL 散度的权重
        temp_init=1.0,  # 初始温度
        use_vqinterface=True,  # 是否使用 VQ 接口
        remap=None,  # 重新映射参数
        unknown_index="random",  # 未知索引的处理方式
    ):
        super().__init__()  # 调用父类初始化

        self.embedding_dim = embedding_dim  # 设置嵌入维度
        self.n_embed = n_embed  # 设置嵌入数量

        self.straight_through = straight_through  # 保存直通估计的状态
        self.temperature = temp_init  # 设置温度
        self.kl_weight = kl_weight  # 设置 KL 权重

        # 定义卷积层，将隐藏层映射到嵌入空间
        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        # 定义嵌入层，将索引映射到嵌入向量
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface  # 保存 VQ 接口使用状态

        self.remap = remap  # 保存重新映射参数
        if self.remap is not None:  # 如果存在重新映射
            # 注册用于存储重新映射的张量
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]  # 重新映射的嵌入数量
            self.unknown_index = unknown_index  # 保存未知索引
            if self.unknown_index == "extra":  # 如果未知索引为“extra”
                self.unknown_index = self.re_embed  # 设置未知索引为重新映射数量
                self.re_embed = self.re_embed + 1  # 增加重新映射数量
            # 打印重新映射的信息
            print(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_embed  # 否则，重新映射数量等于嵌入数量

    # 将索引重新映射到已使用的索引
    def remap_to_used(self, inds):
        ishape = inds.shape  # 获取输入的形状
        assert len(ishape) > 1  # 确保输入维度大于1
        inds = inds.reshape(ishape[0], -1)  # 将索引重塑为二维形状
        used = self.used.to(inds)  # 将已使用的张量移动到索引的设备上
        match = (inds[:, :, None] == used[None, None, ...]).long()  # 计算匹配矩阵
        new = match.argmax(-1)  # 获取匹配的最大值索引
        unknown = match.sum(2) < 1  # 检查未知索引
        if self.unknown_index == "random":  # 如果未知索引为随机
            # 随机生成未知索引
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index  # 否则设置为未知索引
        return new.reshape(ishape)  # 返回重塑后的新索引

    # 将索引映射回所有索引
    def unmap_to_all(self, inds):
        ishape = inds.shape  # 获取输入的形状
        assert len(ishape) > 1  # 确保输入维度大于1
        inds = inds.reshape(ishape[0], -1)  # 将索引重塑为二维形状
        used = self.used.to(inds)  # 将已使用的张量移动到索引的设备上
        if self.re_embed > self.used.shape[0]:  # 如果有额外的标记
            inds[inds >= self.used.shape[0]] = 0  # 将超过范围的索引设置为零
        # 根据输入索引从已使用张量中收集值
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)  # 返回重塑后的结果
    # 定义前向传播方法，接受潜在变量 z、温度 temp 和返回 logits 的标志
    def forward(self, z, temp=None, return_logits=False):
        # 在评估模式下强制硬性为 True，因为必须进行量化。实际上，总是设为 True 似乎也可以
        hard = self.straight_through if self.training else True
        # 如果未提供温度，则使用类的温度属性
        temp = self.temperature if temp is None else temp
    
        # 将输入 z 投影到 logits 空间
        logits = self.proj(z)
        if self.remap is not None:
            # 仅继续使用的 logits
            full_zeros = torch.zeros_like(logits)  # 创建与 logits 同形状的零张量
            logits = logits[:, self.used, ...]  # 只保留使用的 logits
    
        # 使用 Gumbel-softmax 生成软 one-hot 编码
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # 返回到所有条目，但未使用的设置为零
            full_zeros[:, self.used, ...] = soft_one_hot  # 将使用的编码放入全零张量中
            soft_one_hot = full_zeros  # 更新为全零张量
    
        # 计算量化后的表示
        z_q = einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)
    
        # 加上对先验损失的 KL 散度
        qy = F.softmax(logits, dim=1)  # 对 logits 进行 softmax 操作
        # 计算 KL 散度
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
    
        # 找到软 one-hot 编码中最大值的索引
        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)  # 进行重映射
        if self.use_vqinterface:
            if return_logits:
                # 如果需要返回 logits，则返回量化后的表示、KL 散度和索引
                return z_q, diff, (None, None, ind), logits
            # 如果不需要返回 logits，则返回量化后的表示、KL 散度和索引
            return z_q, diff, (None, None, ind)
        # 返回量化后的表示、KL 散度和索引
        return z_q, diff, ind
    
    # 定义获取代码本条目的方法，接受索引和形状
    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape  # 解构形状信息
        assert b * h * w == indices.shape[0]  # 确保索引数量与形状一致
        indices = rearrange(indices, "(b h w) -> b h w", b=b, h=h, w=w)  # 重新排列索引形状
        if self.remap is not None:
            indices = self.unmap_to_all(indices)  # 如果需要，进行反映射
        # 创建 one-hot 编码并调整维度顺序
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        # 计算量化后的表示
        z_q = einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        return z_q  # 返回量化后的表示
```