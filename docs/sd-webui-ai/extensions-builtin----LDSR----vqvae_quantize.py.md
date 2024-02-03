# `stable-diffusion-webui\extensions-builtin\LDSR\vqvae_quantize.py`

```py
# 从指定链接处导入代码，包含了版权声明和许可证信息
# 代码版权归属于 Patrick Esser、Robin Rombach 和 Björn Ommer
# 允许任何人免费获取该软件及相关文档文件的副本，并在不受限制的情况下处理该软件
# 包括但不限于使用、复制、修改、合并、发布、分发、许可和/或销售该软件的副本
# 并允许将该软件提供给接收软件的人，但需要遵守以下条件
# 在所有副本或实质部分的软件中必须包含上述版权声明和此许可声明
# 本软件按原样提供，不提供任何形式的担保，包括但不限于适销性、特定用途适用性和非侵权性的担保
# 作者或版权持有人不对任何索赔、损害或其他责任负责，无论是合同、侵权行为还是其他方式，由软件或使用软件引起的、或与软件或使用软件有关的
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# 定义一个名为 VectorQuantizer2 的类，是 VectorQuantizer 的改进版本，可以作为一个可替换的组件使用
# 主要避免了昂贵的矩阵乘法，并允许后续重新映射索引
class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # 注意：由于一个 bug，beta 项被应用到了错误的项上。为了向后兼容，默认使用有错误的版本，但可以指定 legacy=False 来修复它
    # 初始化函数，设置模型参数和属性
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化模型参数
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        # 创建一个嵌入层，用于将输入索引映射为密集向量
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # 初始化嵌入层的权重
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # 设置重新映射参数
        self.remap = remap
        if self.remap is not None:
            # 加载重新映射数据
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            # 处理未知索引的情况
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            # 打印重新映射信息
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        # 设置是否保持索引形状
        self.sane_index_shape = sane_index_shape

    # 将索引映射到已使用的索引
    def remap_to_used(self, inds):
        ishape = inds.shape
        # 确保输入索引的维度大于1
        assert len(ishape) > 1
        # 重塑输入索引的形状
        inds = inds.reshape(ishape[0], -1)
        # 将已使用的索引转换为与输入索引相同的设备
        used = self.used.to(inds)
        # 创建匹配矩阵
        match = (inds[:, :, None] == used[None, None, ...]).long()
        # 找到匹配的索引
        new = match.argmax(-1)
        # 找到未知索引
        unknown = match.sum(2) < 1
        # 处理未知索引的情况
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        # 重塑输出索引的形状
        return new.reshape(ishape)
    # 将输入的索引映射到所有可能的索引
    def unmap_to_all(self, inds):
        # 获取输入索引的形状
        ishape = inds.shape
        # 确保输入索引的维度大于1
        assert len(ishape) > 1
        # 将输入索引重塑为二维数组
        inds = inds.reshape(ishape[0], -1)
        # 使用self.used对inds进行索引
        used = self.used.to(inds)
        # 如果重新嵌入的数量大于self.used的行数，表示有额外的标记
        if self.re_embed > self.used.shape[0]:  # extra token
            # 将大于等于self.used.shape[0]的索引设置为0
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        # 使用torch.gather函数根据索引获取对应的值
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        # 将结果重塑为原始形状
        return back.reshape(ishape)

    # 获取代码本体条目
    def get_codebook_entry(self, indices, shape):
        # shape参数指定了(batch, height, width, channel)的形状
        if self.remap is not None:
            # 将索引重塑为(batch, -1)的形状，添加批次维度
            indices = indices.reshape(shape[0], -1)  # add batch axis
            # 对所有索引进行unmap_to_all操作
            indices = self.unmap_to_all(indices)
            # 再次将索引重塑为一维数组
            indices = indices.reshape(-1)  # flatten again

        # 获取量化的潜在向量
        z_q = self.embedding(indices)

        # 如果shape不为None
        if shape is not None:
            # 将z_q重塑为指定的形状
            z_q = z_q.view(shape)
            # 将形状重新排列以匹配原始输入形状
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
```