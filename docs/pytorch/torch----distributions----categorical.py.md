# `.\pytorch\torch\distributions\categorical.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 导入 nan 函数
from torch import nan
# 导入 constraints 模块
from torch.distributions import constraints
# 导入 Distribution 类
from torch.distributions.distribution import Distribution
# 导入 lazy_property, logits_to_probs, probs_to_logits 函数
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits

# 定义模块的公开接口列表
__all__ = ["Categorical"]

# 定义 Categorical 类，继承自 Distribution 类
class Categorical(Distribution):
    r"""
    创建由 `probs` 或 `logits` 参数化的分类分布（但不能同时使用两者）。

    .. note::
        等同于从 :func:`torch.multinomial` 中抽样的分布。

    样本是来自于 :math:`\{0, \ldots, K-1\}` 的整数，其中 `K` 是 `probs.size(-1)`。

    如果 `probs` 是长度为 `K` 的一维张量，则每个元素是在该索引处抽取类的相对概率。

    如果 `probs` 是 N 维的，则前 N-1 维被视为批次的相对概率向量。

    .. note:: `probs` 参数必须是非负的、有限的，并且总和非零，将被归一化为沿最后一个维度和为 1。
              `probs` 将返回这个归一化后的值。
              `logits` 参数将被解释为未归一化的对数概率，因此可以是任意实数。
              它也将被归一化，以便结果概率沿最后一个维度和为 1。
              `logits` 将返回这个归一化后的值。

    参见：:func:`torch.multinomial`

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # 各类别有相同的概率抽取
        tensor(3)

    Args:
        probs (Tensor): 事件概率
        logits (Tensor): 事件对数概率（未归一化）
    """
    
    # 参数约束字典，定义了 probs 和 logits 的约束条件
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    # 表示支持列举所有支持的值
    has_enumerate_support = True

    # 构造函数，初始化分类分布对象
    def __init__(self, probs=None, logits=None, validate_args=None):
        # 检查参数，确保只有 probs 或 logits 中的一个被指定
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            # 检查 probs 的维度，至少应为一维
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            # 归一化 probs
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            # 检查 logits 的维度，至少应为一维
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # 归一化 logits
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        
        # 根据参数选择实际使用的参数（归一化后的 probs 或 logits）
        self._param = self.probs if probs is not None else self.logits
        # 计算事件数量
        self._num_events = self._param.size()[-1]
        # 计算批次形状
        batch_shape = (
            self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        )
        # 调用父类的构造函数进行初始化
        super().__init__(batch_shape, validate_args=validate_args)
    # 将当前对象的形状扩展为给定的批次形状，并返回一个新的 Categorical 对象
    def expand(self, batch_shape, _instance=None):
        # 创建一个新的 Categorical 对象，确保 _instance 是 Categorical 类型的实例
        new = self._get_checked_instance(Categorical, _instance)
        # 将 batch_shape 转换为 torch.Size 类型
        batch_shape = torch.Size(batch_shape)
        # 计算参数的形状，即批次形状加上事件数量的形状
        param_shape = batch_shape + torch.Size((self._num_events,))
        # 如果对象包含 'probs' 属性，则将其扩展到 param_shape，并设置新对象的参数为 probs
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        # 如果对象包含 'logits' 属性，则将其扩展到 param_shape，并设置新对象的参数为 logits
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = new.logits
        # 设置新对象的事件数量为当前对象的事件数量
        new._num_events = self._num_events
        # 调用父类的初始化方法，初始化新对象的批次形状，validate_args 设置为 False
        super(Categorical, new).__init__(batch_shape, validate_args=False)
        # 设置新对象的 validate_args 属性与当前对象相同
        new._validate_args = self._validate_args
        # 返回新创建的 Categorical 对象
        return new

    # 创建并返回与当前对象参数相同类型的新对象
    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    # 返回支持该分布的约束条件，为离散值且事件维度为 0
    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self._num_events - 1)

    # 惰性加载属性，返回 logits 属性的值，通过 probs_to_logits 函数计算得到
    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    # 惰性加载属性，返回 probs 属性的值，通过 logits_to_probs 函数计算得到
    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)

    # 返回参数的形状，即 _param 的大小
    @property
    def param_shape(self):
        return self._param.size()

    # 返回分布的均值，使用 NaN 填充
    @property
    def mean(self):
        return torch.full(
            self._extended_shape(),
            nan,
            dtype=self.probs.dtype,
            device=self.probs.device,
        )

    # 返回分布的众数，即在最后一个维度上取最大值的索引
    @property
    def mode(self):
        return self.probs.argmax(axis=-1)

    # 返回分布的方差，使用 NaN 填充
    @property
    def variance(self):
        return torch.full(
            self._extended_shape(),
            nan,
            dtype=self.probs.dtype,
            device=self.probs.device,
        )

    # 从分布中抽样，返回抽样的值
    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        # 将 probs 属性转换为二维形状，并使用 multinomial 函数进行多项式抽样
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        # 将抽样结果重塑为扩展形状后返回
        return samples_2d.reshape(self._extended_shape(sample_shape))

    # 返回给定值的对数概率密度值
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 将值转换为长整型并在最后一个维度上增加一个维度
        value = value.long().unsqueeze(-1)
        # 广播张量 value 和 logits，保证它们的形状相同
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        # 使用 gather 函数获取对数概率密度，并在最后一个维度上去除多余的维度
        return log_pmf.gather(-1, value).squeeze(-1)

    # 返回分布的熵值
    def entropy(self):
        # 获取 logits 数据类型的最小实数值，并对 logits 进行截断
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        # 计算 p * log(p) 并在最后一个维度上求和并取负值，得到熵值
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    # 枚举支持分布的所有可能值
    def enumerate_support(self, expand=True):
        num_events = self._num_events
        # 创建一个包含所有可能值的张量，并设置数据类型和设备
        values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        # 如果需要扩展，则在指定维度上扩展张量的形状
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        # 返回所有可能值的张量
        return values
```