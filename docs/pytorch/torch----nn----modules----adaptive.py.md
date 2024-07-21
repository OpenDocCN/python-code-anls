# `.\pytorch\torch\nn\modules\adaptive.py`

```py
# mypy: allow-untyped-defs

# 导入命名元组和类型提示需要的模块
from collections import namedtuple
from typing import List, Sequence

# 导入 PyTorch 相关模块
import torch
import torch.nn.functional as F
from torch import Tensor

# 导入自定义模块
from .container import ModuleList, Sequential
from .linear import Linear
from .module import Module

# 定义模块对外暴露的接口列表
__all__ = ["AdaptiveLogSoftmaxWithLoss"]

# 命名元组用于保存模块输出和损失
_ASMoutput = namedtuple("_ASMoutput", ["output", "loss"])

# 自定义的适应性 LogSoftmax 损失模块，继承自 PyTorch 的 Module 类
class AdaptiveLogSoftmaxWithLoss(Module):
    r"""Efficient softmax approximation.

    As described in
    `Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin,
    Moustapha Ciss\u00e9, David Grangier, and Herv\u00e9 J\u00e9gou
    <https://arxiv.org/abs/1609.04309>`__.

    Adaptive softmax is an approximate strategy for training models with large
    output spaces. It is most effective when the label distribution is highly
    imbalanced, for example in natural language modelling, where the word
    frequency distribution approximately follows the `Zipf's law`_.

    Adaptive softmax partitions the labels into several clusters, according to
    their frequency. These clusters may contain different number of targets
    each.
    Additionally, clusters containing less frequent labels assign lower
    dimensional embeddings to those labels, which speeds up the computation.
    For each minibatch, only clusters for which at least one target is
    present are evaluated.

    The idea is that the clusters which are accessed frequently
    (like the first one, containing most frequent labels), should also be cheap
    to compute -- that is, contain a small number of assigned labels.

    We highly recommend taking a look at the original paper for more details.

    * :attr:`cutoffs` should be an ordered Sequence of integers sorted
      in the increasing order.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the adaptive softmax, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster.

    * :attr:`div_value` is used to compute the size of each additional cluster,
      which is given as
      :math:`\left\lfloor\frac{\texttt{in\_features}}{\texttt{div\_value}^{idx}}\right\rfloor`,
      where :math:`idx` is the cluster index (with clusters
      for less frequent words having larger indices,
      and indices starting from :math:`1`).

    * :attr:`head_bias` if set to True, adds a bias term to the 'head' of the
      adaptive softmax. See paper for details. Set to False in the official
      implementation.
    # 定义一个模块，用于计算自适应 softmax 的输出和损失
    """
    .. warning::
        传入此模块的标签应按其频率进行排序。这意味着最频繁的标签应该由索引 `0` 表示，
        最不频繁的标签应该由索引 `n_classes - 1` 表示。
    
    .. note::
        该模块返回一个 ``NamedTuple``，包含 ``output`` 和 ``loss`` 字段。详细信息请参阅进一步的文档。
    
    .. note::
        要计算所有类别的对数概率，可以使用 ``log_prob`` 方法。
    
    Args:
        in_features (int): 输入张量中的特征数
        n_classes (int): 数据集中的类数
        cutoffs (Sequence): 用于将目标分配到其桶中的截止值
        div_value (float, optional): 计算集群大小的指数的值。默认值：4.0
        head_bias (bool, optional): 如果 ``True``，则向自适应 softmax 的 'head' 添加偏置项。默认值：``False``
    
    Returns:
        ``NamedTuple``，包含 ``output`` 和 ``loss`` 字段：
            * **output** 是一个大小为 ``N`` 的张量，包含每个样本的计算目标对数概率
            * **loss** 是一个标量，表示计算的负对数似然损失
    
    Shape:
        - input: :math:`(N, \texttt{in\_features})` 或 :math:`(\texttt{in\_features})`
        - target: :math:`(N)` 或 :math:`()`，其中每个值满足 :math:`0 <= \texttt{target[i]} <= \texttt{n\_classes}`
        - output1: :math:`(N)` 或 :math:`()`
        - output2: ``Scalar``
    """
    ) -> None:
        # 定义构造函数，接收设备和数据类型作为关键字参数
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类的构造函数
        super().__init__()

        # 将 cutoffs 转换为列表
        cutoffs = list(cutoffs)

        # 如果 cutoffs 的长度为0，则抛出数值错误异常
        if len(cutoffs) == 0:
            raise ValueError("cutoffs should be a sequence of length larger than 0")

        # 检查 cutoffs 是否满足排序、范围、唯一性和类型的条件
        if (
            (cutoffs != sorted(cutoffs))
            or (min(cutoffs) <= 0)
            or (max(cutoffs) > (n_classes - 1))
            or (len(set(cutoffs)) != len(cutoffs))
            or any(int(c) != c for c in cutoffs)
        ):
            raise ValueError(
                "cutoffs should be a sequence of unique, positive "
                "integers sorted in an increasing order, where "
                "each value is between 1 and n_classes-1"
            )

        # 初始化对象的属性
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        # 计算头部的短列表大小、簇的数量和头部的总大小
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        # 创建头部线性层对象
        self.head = Linear(
            self.in_features, self.head_size, bias=self.head_bias, **factory_kwargs
        )
        # 初始化尾部模块列表
        self.tail = ModuleList()

        # 遍历每一个簇，初始化对应的投影层
        for i in range(self.n_clusters):
            # 计算当前簇的隐藏层和输出层大小
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            # 创建当前簇的投影层
            projection = Sequential(
                Linear(self.in_features, hsz, bias=False, **factory_kwargs),
                Linear(hsz, osz, bias=False, **factory_kwargs),
            )

            # 将当前投影层添加到尾部模块列表中
            self.tail.append(projection)

    def reset_parameters(self) -> None:
        # 重置头部的参数
        self.head.reset_parameters()
        # 遍历每一个尾部的投影层，并重置其参数
        for i2h, h2o in self.tail:
            i2h.reset_parameters()
            h2o.reset_parameters()
    # 定义一个方法，接收输入和目标张量，返回_ASMoutput类型的结果
    def forward(self, input_: Tensor, target_: Tensor) -> _ASMoutput:
        # 确定目标张量的维度
        targ_dim = target_.dim()

        # 如果目标维度为1
        if targ_dim == 1:
            # 检查输入张量和目标张量在批处理维度上是否大小相同，如果不同则抛出异常
            if input_.size(0) != target_.size(0):
                raise RuntimeError(
                    "Input and target should have the same size "
                    "in the batch dimension."
                )
            # 检查输入张量的维度是否为2，如果不是则抛出异常
            if input_.dim() != 2:
                raise RuntimeError(
                    "1D target tensor expects 2D input tensors, "
                    "but found inputs with size",
                    input_.size(),
                )
        # 如果目标维度为0
        elif targ_dim == 0:
            # 检查输入张量维度是否为1，如果不是则抛出异常
            if input_.dim() != 1:
                raise RuntimeError(
                    "0D target tensor expects 1D input tensors, "
                    "but found inputs with size",
                    input_.size(),
                )
        # 如果目标维度既不为0也不为1
        else:
            # 抛出异常，表示不支持多目标张量
            raise RuntimeError(
                "0D or 1D target tensor expected, " "multi-target not supported"
            )

        # 判断是否为批处理数据
        is_batched = targ_dim > 0
        # 如果为批处理数据，保持输入张量不变，否则在第0维度上增加一个维度
        input = input_ if is_batched else input_.unsqueeze(0)
        # 如果为批处理数据，保持目标张量不变，否则在第0维度上增加一个维度
        target = target_ if is_batched else target_.unsqueeze(0)

        # 初始化已使用的行数和批处理大小
        used_rows = 0
        batch_size = target.size(0)

        # 创建和输入相同数据类型的全零张量output，创建和目标张量相同数据类型的空张量gather_inds
        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        # 定义一个截断值列表，包括0和self.cutoffs
        cutoff_values = [0] + self.cutoffs
        # 遍历截断值列表
        for i in range(len(cutoff_values) - 1):
            # 获取当前截断值和下一个截断值
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            # 根据截断值范围创建目标掩码
            target_mask = (target >= low_idx) & (target < high_idx)
            # 获取非零元素的索引，并去除维度为1
            row_indices = target_mask.nonzero().squeeze()

            # 如果索引数量为0，则继续下一次循环
            if row_indices.numel() == 0:
                continue

            # 如果为第一个截断值
            if i == 0:
                # 在gather_inds指定索引位置插入目标张量对应位置的值
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            # 如果不是第一个截断值
            else:
                # 计算相对目标值，输入张量的子集
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(0, row_indices)

                # 对子集进行操作，获取局部概率并更新output和gather_inds
                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1
                gather_inds.index_fill_(0, row_indices, cluster_index)
                cluster_logprob = F.log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            # 更新已使用的行数
            used_rows += row_indices.numel()

        # 如果已使用的行数不等于批处理大小，则抛出异常
        if used_rows != batch_size:
            raise RuntimeError(
                f"Target values should be in [0, {self.n_classes - 1}], "
                f"but values in range [{target.min().item()}, {target.max().item()}] "
                "were found. "
            )

        # 对输入张量进行操作，获取头部输出和头部概率
        head_output = self.head(input)
        head_logprob = F.log_softmax(head_output, dim=1)
        # 更新output并计算损失值
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = (-output).mean()

        # 如果不是批处理数据，将output压缩为1维
        if not is_batched:
            output = output.squeeze(0)

        # 返回_ASMoutput类型的结果
        return _ASMoutput(output, loss)
    def _get_full_log_prob(self, input, head_output):
        """Given input tensor, and output of ``self.head``, compute the log of the full distribution."""
        # 创建一个空的输出张量，形状为 (batch_size, self.n_classes)
        out = input.new_empty((head_output.size(0), self.n_classes))
        # 对 self.head 输出的概率分布进行 log_softmax 处理
        head_logprob = F.log_softmax(head_output, dim=1)

        # 将头部输出的前 shortlist_size 个类别的 log 概率赋值给 out
        out[:, : self.shortlist_size] = head_logprob[:, : self.shortlist_size]

        # 遍历每个子群组的起始和结束索引，计算每个子群组的 log 概率
        for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
            # 使用第 i 个子群组的网络层计算输出
            cluster_output = self.tail[i](input)
            # 对子群组输出的概率分布进行 log_softmax 处理
            cluster_logprob = F.log_softmax(cluster_output, dim=1)
            # 计算每个样本在第 i 个子群组的输出 log 概率，加上对应头部的 log 概率
            output_logprob = cluster_logprob + head_logprob[
                :, self.shortlist_size + i
            ].unsqueeze(1)

            # 将计算得到的输出 log 概率赋值给 out 的相应位置
            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, input: Tensor) -> Tensor:
        r"""Compute log probabilities for all :math:`\texttt{n\_classes}`.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N, \texttt{n\_classes})`

        """
        # 计算头部网络的输出
        head_output = self.head(input)
        # 调用 _get_full_log_prob 方法计算完整分布的 log 概率，并返回结果
        return self._get_full_log_prob(input, head_output)

    def predict(self, input: Tensor) -> Tensor:
        r"""Return the class with the highest probability for each example in the input minibatch.

        This is equivalent to ``self.log_prob(input).argmax(dim=1)``, but is more efficient in some cases.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N)`
        """
        # 计算头部网络的输出
        head_output = self.head(input)
        # 根据头部网络的输出计算初始预测
        output = torch.argmax(head_output, dim=1)
        # 判断哪些样本的初始预测不在 shortlist 中
        not_in_shortlist = output >= self.shortlist_size
        # 判断是否所有样本的初始预测都在 shortlist 中
        all_in_shortlist = not (not_in_shortlist.any())

        if all_in_shortlist:
            # 如果所有样本的初始预测都在 shortlist 中，直接返回初始预测
            return output

        elif not_in_shortlist.all():
            # 如果所有样本的初始预测都不在 shortlist 中，计算完整分布的 log 概率并返回最大概率的类别
            log_prob = self._get_full_log_prob(input, head_output)
            return torch.argmax(log_prob, dim=1)

        else:
            # 对于部分样本初始预测不在 shortlist 中的情况，仅计算这部分样本的完整分布 log 概率并返回最大概率的类别
            log_prob = self._get_full_log_prob(
                input[not_in_shortlist], head_output[not_in_shortlist]
            )
            output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
            return output
```