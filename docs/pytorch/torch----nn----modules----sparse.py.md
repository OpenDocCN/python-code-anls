# `.\pytorch\torch\nn\modules\sparse.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型定义
from typing import Optional

# 引入 PyTorch 库
import torch
from torch import Tensor
from torch.nn import functional as F, init  # 导入神经网络函数和初始化模块
from torch.nn.parameter import Parameter  # 导入参数模块

# 导入自定义的模块
from .module import Module

# 定义导出的模块列表
__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings 词嵌入的字典大小
        embedding_dim (int): the size of each embedding vector 每个嵌入向量的大小
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                     the embedding vector at :attr:`padding_idx` will default to all zeros,
                                     but can be updated to another value to be used as the padding vector.
                                     如果指定，这些索引位置的条目不会贡献梯度;因此，梯度向量在训练过程中不会更新
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`大于则被 renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the option. Default ``2``
 Input shape Output input shape mini-batch
 tensor w.r.t details
 tensors.

  contain  integer . different
 support tensor.SGD adam
    # 定义一个包含嵌入矩阵的神经网络模块，用于将整数索引映射到固定大小的稠密向量。
    class Embedding(Module):
        # 当 max_norm 不为 None 时，Embedding 的 forward 方法将会原地修改 weight 张量。
        # 由于梯度计算所需的张量不能原地修改，因此在调用 Embedding 的 forward 方法之前，
        # 在 max_norm 不为 None 时需要克隆 Embedding.weight。
        
        # 示例：在实例化 Embedding 时，指定 max_norm=True。
        n, d, m = 3, 5, 7
        embedding = nn.Embedding(n, d, max_norm=True)
        
        # 创建一个大小为 (m, d) 的随机张量 W，要求其梯度计算。
        W = torch.randn((m, d), requires_grad=True)
        
        # 创建一个包含整数索引的张量 idx。
        idx = torch.tensor([1, 2])
        
        # 克隆 embedding.weight，并与 W 的转置矩阵相乘，以确保此操作可微分。
        a = embedding.weight.clone() @ W.t()  # weight must be cloned for this to be differentiable
        
        # 调用 embedding(idx)，将会原地修改 weight。
        b = embedding(idx) @ W.t()
        
        # 计算张量 a 和 b 的张量加法，其中 a 维度扩展为 (1, d)，b 维度扩展为 (2, d)。
        out = (a.unsqueeze(0) + b.unsqueeze(1))
        
        # 计算 out 的 sigmoid 激活，并计算所有元素的乘积作为损失值。
        loss = out.sigmoid().prod()
        
        # 反向传播损失值的梯度。
        loss.backward()
    
    # 示例：创建一个包含 10 个大小为 3 的张量的 Embedding 模块。
    embedding = nn.Embedding(10, 3)
    
    # 示例：创建一个大小为 (2, 4) 的批次张量，包含整数索引。
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    embedding(input)
    
    # 示例：创建一个包含 padding_idx 的 Embedding 模块。
    embedding = nn.Embedding(10, 3, padding_idx=0)
    
    # 示例：创建一个大小为 (1, 4) 的批次张量，包含整数索引，其中包含 padding_idx。
    input = torch.LongTensor([[0, 2, 0, 5]])
    embedding(input)
    
    # 示例：修改 padding 向量的例子。
    padding_idx = 0
    embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
    
    # 输出 Embedding 模块的权重，显示其包含 padding 向量。
    embedding.weight
    
    # 使用 torch.no_grad() 上下文管理器，将 padding 向量修改为全为 1 的张量。
    with torch.no_grad():
        embedding.weight[padding_idx] = torch.ones(3)
    
    # 输出修改后的 Embedding 模块的权重。
    embedding.weight
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool


# 定义类的成员变量，分别表示归一化类型、是否按频率缩放梯度、权重张量、是否冻结权重、是否稀疏处理
norm_type: float
scale_grad_by_freq: bool
weight: Tensor
freeze: bool
sparse: bool



    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:


# 类的初始化方法，设置对象的初始状态和属性
def __init__(
    self,
    num_embeddings: int,            # 嵌入的数量
    embedding_dim: int,             # 每个嵌入向量的维度
    padding_idx: Optional[int] = None,   # 可选参数，填充索引
    max_norm: Optional[float] = None,    # 可选参数，最大范数限制
    norm_type: float = 2.0,          # 归一化类型，默认为2.0
    scale_grad_by_freq: bool = False,    # 是否按频率缩放梯度，默认为False
    sparse: bool = False,            # 是否使用稀疏张量，默认为False
    _weight: Optional[Tensor] = None,    # 可选参数，权重张量
    _freeze: bool = False,           # 是否冻结权重，默认为False
    device=None,                    # 设备
    dtype=None,                     # 数据类型
) -> None:



        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
                requires_grad=not _freeze,
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = Parameter(_weight, requires_grad=not _freeze)

        self.sparse = sparse


# 初始化方法的具体实现，设置对象的属性和状态
factory_kwargs = {"device": device, "dtype": dtype}   # 使用设备和数据类型创建工厂参数
super().__init__()   # 调用父类的初始化方法
self.num_embeddings = num_embeddings   # 设置嵌入数量
self.embedding_dim = embedding_dim     # 设置嵌入维度
if padding_idx is not None:
    # 处理填充索引
    if padding_idx > 0:
        assert (
            padding_idx < self.num_embeddings
        ), "Padding_idx must be within num_embeddings"
    elif padding_idx < 0:
        assert (
            padding_idx >= -self.num_embeddings
        ), "Padding_idx must be within num_embeddings"
        padding_idx = self.num_embeddings + padding_idx
self.padding_idx = padding_idx     # 设置填充索引
self.max_norm = max_norm           # 设置最大范数限制
self.norm_type = norm_type         # 设置归一化类型
self.scale_grad_by_freq = scale_grad_by_freq   # 设置是否按频率缩放梯度
if _weight is None:
    # 如果权重为空，则创建新的参数张量，并根据是否冻结设置梯度计算
    self.weight = Parameter(
        torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
        requires_grad=not _freeze,
    )
    self.reset_parameters()     # 调用重置参数方法
else:
    # 如果给定了权重，确保其形状与num_embeddings和embedding_dim匹配
    assert list(_weight.shape) == [
        num_embeddings,
        embedding_dim,
    ], "Shape of weight does not match num_embeddings and embedding_dim"
    self.weight = Parameter(_weight, requires_grad=not _freeze)

self.sparse = sparse   # 设置是否使用稀疏张量



    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()


# 重置参数方法，初始化权重并填充填充索引为零
def reset_parameters(self) -> None:
    init.normal_(self.weight)   # 使用正态分布初始化权重
    self._fill_padding_idx_with_zero()   # 调用填充填充索引为零的私有方法



    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


# 填充填充索引为零的私有方法
def _fill_padding_idx_with_zero(self) -> None:
    if self.padding_idx is not None:
        with torch.no_grad():   # 禁止梯度计算
            self.weight[self.padding_idx].fill_(0)   # 将填充索引位置的权重值填充为零



    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


# 前向传播方法，使用PyTorch的嵌入函数计算结果
def forward(self, input: Tensor) -> Tensor:
    return F.embedding(
        input,
        self.weight,
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )



    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)


# 返回对象的额外信息的方法，以字符串形式表示
def extra_repr(self) -> str:
    s = "{num_embeddings}, {embedding_dim}"   # 初始化字符串模板
    if self.padding_idx is not None:
        s += ", padding_idx={padding_idx}"   # 如果存在填充索引，添加到字符串
    if self.max_norm is not None:
        s += ", max_norm={max_norm}"   # 如果存在最大范数限制，添加到字符串
    if self.norm_type != 2:
        s += ", norm_type={norm_type}"   # 如果归一化类型不为2，默认值，添加到字符串
    if self.scale_grad_by_freq is not False:
        s += ", scale_grad_by_freq={scale_grad_by_freq}"   # 如果按频率缩放梯度为True，添加到字符串
    if self.sparse is not False:
        s += ", sparse=True"   # 如果使用稀疏张量为True，添加到字符串
    return s.format(**self.__dict__)   # 使用对象的属性格式化字符串



    @classmethod


# 类方法的声明
@classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze=True,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        r"""Create Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (bool, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                         therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                         i.e. it remains as a fixed "pad".
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (bool, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        # 断言确保输入的 embeddings 张量是二维的
        assert (
            embeddings.dim() == 2
        ), "Embeddings parameter is expected to be 2-dimensional"
        # 获取 embeddings 张量的行数和列数
        rows, cols = embeddings.shape
        # 创建一个 Embedding 实例，使用给定的参数进行初始化
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            _freeze=freeze,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        # 返回创建的 Embedding 实例
        return embedding
# 定义一个名为 EmbeddingBag 的类，继承自 Module 类
class EmbeddingBag(Module):
    # 文档字符串，描述该类的作用和功能
    r"""Compute sums or means of 'bags' of embeddings, without instantiating the intermediate embeddings.

    For bags of constant length, no :attr:`per_sample_weights`, no indices equal to :attr:`padding_idx`,
    and with 2D inputs, this class

        * with ``mode="sum"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.sum(dim=1)``,
        * with ``mode="mean"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.mean(dim=1)``,
        * with ``mode="max"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.max(dim=1)``.

    However, :class:`~torch.nn.EmbeddingBag` is much more time and memory efficient than using a chain of these
    operations.

    EmbeddingBag also supports per-sample weights as an argument to the forward
    pass. This scales the output of the Embedding before performing a weighted
    reduction as specified by ``mode``. If :attr:`per_sample_weights` is passed, the
    only supported ``mode`` is ``"sum"``, which computes a weighted sum according to
    :attr:`per_sample_weights`.
    """
    # 定义一个类，用于创建一个 EmbeddingBag 模块
    class EmbeddingBag(Module):
        # 初始化函数，设置模块的初始状态
        def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2.0,
                     scale_grad_by_freq=False, mode='mean', sparse=False,
                     include_last_offset=False, padding_idx=None):
            # 初始化权重矩阵，形状为 (num_embeddings, embedding_dim)，从标准正态分布中抽取初始值
            self.weight = Parameter(torch.empty(num_embeddings, embedding_dim, dtype=torch.float).normal_(0, 1))
            
            # 将参数保存到实例变量中
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.mode = mode
            self.sparse = sparse
            self.include_last_offset = include_last_offset
            self.padding_idx = padding_idx
    
        # 省略其他方法和实现细节
    # 定义常量列表，包含了该类中的常量参数名
    __constants__ = [
        "num_embeddings",           # 嵌入矩阵的行数（词汇表的大小）
        "embedding_dim",            # 嵌入向量的维度
        "max_norm",                 # 最大范数值，用于进行范数约束
        "norm_type",                # 范数的类型，默认为2范数
        "scale_grad_by_freq",       # 是否按频率缩放梯度
        "mode",                     # 嵌入操作的模式，如'mean'或'sum'
        "sparse",                   # 是否使用稀疏梯度
        "include_last_offset",      # 是否包含最后一个偏移量
        "padding_idx",              # 填充索引，用于指定填充的词向量
    ]
    
    num_embeddings: int             # 嵌入矩阵的行数（词汇表的大小）
    embedding_dim: int              # 嵌入向量的维度
    max_norm: Optional[float]       # 最大范数值，用于进行范数约束（可选）
    norm_type: float                # 范数的类型，默认为2范数
    scale_grad_by_freq: bool        # 是否按频率缩放梯度
    weight: Tensor                  # 嵌入层的权重矩阵
    mode: str                       # 嵌入操作的模式，如'mean'或'sum'
    sparse: bool                    # 是否使用稀疏梯度
    include_last_offset: bool       # 是否包含最后一个偏移量
    padding_idx: Optional[int]      # 填充索引，用于指定填充的词向量（可选）
    
    def __init__(
        self,
        num_embeddings: int,        # 嵌入矩阵的行数（词汇表的大小）
        embedding_dim: int,         # 嵌入向量的维度
        max_norm: Optional[float] = None,  # 最大范数值，用于进行范数约束（可选）
        norm_type: float = 2.0,     # 范数的类型，默认为2范数
        scale_grad_by_freq: bool = False,  # 是否按频率缩放梯度
        mode: str = "mean",         # 嵌入操作的模式，默认为'mean'
        sparse: bool = False,       # 是否使用稀疏梯度
        _weight: Optional[Tensor] = None,   # 权重矩阵的初始值（可选）
        include_last_offset: bool = False,  # 是否包含最后一个偏移量
        padding_idx: Optional[int] = None,  # 填充索引，用于指定填充的词向量（可选）
        device=None,
        dtype=None,
        ) -> None:
        # 初始化函数，设置嵌入层的参数
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        # 如果有填充索引，确保填充索引在有效范围内
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        # 如果没有提供初始化权重，则创建一个空的权重矩阵，并重置参数
        if _weight is None:
            self.weight = Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
            )
            self.reset_parameters()
        else:
            # 如果提供了初始化权重，则确保其形状与 num_embeddings 和 embedding_dim 匹配
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = Parameter(_weight)
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset

    def reset_parameters(self) -> None:
        # 重置权重矩阵的参数，使用正态分布初始化
        init.normal_(self.weight)
        # 将填充索引位置的权重设置为零
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        # 如果存在填充索引，使用零填充其对应位置的权重
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(
        self,
        input: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        EmbeddingBag 的前向传播方法。

        Args:
            input (Tensor): 包含索引到嵌入矩阵的袋子的张量。
            offsets (Tensor, optional): 仅在 input 是 1D 时使用。offsets 确定 input 中每个袋子（序列）的起始索引位置。
            per_sample_weights (Tensor, optional): 一个浮点 / 双精度权重张量，或者为 None 表示所有权重都应为 1。
                如果指定了 per_sample_weights，则它必须与 input 具有完全相同的形状，并且如果 offsets 不为 None，则被视为具有相同的 offsets。
                仅支持 mode='sum'。

        Returns:
            返回形状为 `(B, embedding_dim)` 的张量。

        .. note::

            关于 input 和 offsets 的一些注意事项：

            - input 和 offsets 必须是相同类型，即 int 或 long

            - 如果 input 是形状为 `(B, N)` 的 2D 张量，则将其视为 ``B`` 个固定长度为 ``N`` 的袋子（序列），
              根据 mode 的不同返回以某种方式聚合的 ``B`` 个值。此时忽略 offsets，并且在这种情况下要求 offsets 为 None。

            - 如果 input 是形状为 `(N)` 的 1D 张量，则将其视为多个袋子（序列）的串联。
              offsets 必须是一个形状为 `(B)` 的 1D 张量，包含 input 中每个袋子的起始索引位置。
              因此，对于形状为 `(B)` 的 offsets，input 将被视为具有 ``B`` 个袋子。
              空袋子（即长度为 0 的袋子）的返回向量将被填充为零。
        """
        return F.embedding_bag(
            input,
            self.weight,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )

    def extra_repr(self) -> str:
        """
        返回一个描述性的字符串，用于表示当前 EmbeddingBag 的配置。

        Returns:
            返回格式化后的字符串，包含 num_embeddings 和 embedding_dim。
            如果设置了 max_norm，则包含 max_norm。
            如果 norm_type 不等于 2，则包含 norm_type。
            如果 scale_grad_by_freq 不为 False，则包含 scale_grad_by_freq。
            包含 mode。
            如果设置了 padding_idx，则包含 padding_idx。
        """
        s = "{num_embeddings}, {embedding_dim}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        s += ", mode={mode}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        return s.format(**{k: repr(v) for k, v in self.__dict__.items()})

    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor,
        freeze: bool = True,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
    ) -> "EmbeddingBag":
        r"""Create EmbeddingBag instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the EmbeddingBag.
                First dimension is being passed to EmbeddingBag as 'num_embeddings', second as 'embedding_dim'.
            freeze (bool, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embeddingbag.weight.requires_grad = False``. Default: ``True``
            max_norm (float, optional): See module initialization documentation. Default: ``None``
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (bool, optional): See module initialization documentation. Default ``False``.
            mode (str, optional): See module initialization documentation. Default: ``"mean"``
            sparse (bool, optional): See module initialization documentation. Default: ``False``.
            include_last_offset (bool, optional): See module initialization documentation. Default: ``False``.
            padding_idx (int, optional): See module initialization documentation. Default: ``None``.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embeddingbag = nn.EmbeddingBag.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([[1, 0]])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embeddingbag(input)
            tensor([[ 2.5000,  3.7000,  4.6500]])

        """
        # 检查输入的嵌入张量是否为二维
        assert (
            embeddings.dim() == 2
        ), "Embeddings parameter is expected to be 2-dimensional"
        
        # 获取嵌入张量的行数和列数
        rows, cols = embeddings.shape
        
        # 使用类构造函数创建 EmbeddingBag 实例
        embeddingbag = cls(
            num_embeddings=rows,                  # 设置嵌入的数量为行数
            embedding_dim=cols,                   # 设置每个嵌入的维度为列数
            _weight=embeddings,                   # 使用给定的权重张量作为初始权重
            max_norm=max_norm,                    # 设置最大范数约束
            norm_type=norm_type,                  # 设置范数的类型
            scale_grad_by_freq=scale_grad_by_freq, # 是否按频率缩放梯度
            mode=mode,                            # 池化操作的模式
            sparse=sparse,                        # 是否使用稀疏梯度
            include_last_offset=include_last_offset, # 是否包含最后一个偏移
            padding_idx=padding_idx,              # 设置填充索引
        )
        
        # 根据 freeze 参数设置权重是否需要梯度更新
        embeddingbag.weight.requires_grad = not freeze
        
        # 返回创建的 EmbeddingBag 实例
        return embeddingbag
```