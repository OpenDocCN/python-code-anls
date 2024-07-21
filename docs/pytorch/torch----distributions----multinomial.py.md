# `.\pytorch\torch\distributions\multinomial.py`

```py
# 导入 torch 库和相关模块
import torch
from torch import inf
from torch.distributions import Categorical, constraints
from torch.distributions.binomial import Binomial
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

# 仅公开 Multinomial 类给外部使用
__all__ = ["Multinomial"]

# Multinomial 类继承自 Distribution 类，表示一个多项分布
class Multinomial(Distribution):
    r"""
    创建一个多项分布，由 :attr:`total_count` 和 :attr:`probs` 或 :attr:`logits`（但不能同时存在）参数化。
    :attr:`probs` 的最内层维度索引表示各个类别的概率，其它维度表示批次。

    注意：如果只调用 :meth:`log_prob` 方法，则无需指定 :attr:`total_count`（参见下面的示例）。

    .. 注意:: `probs` 参数必须是非负、有限且总和非零的，它将被归一化为最后一个维度上和为1的概率分布。
              `probs` 将返回这个归一化后的值。
              `logits` 参数将被解释为未归一化的对数概率，可以是任意实数。它将被归一化为最后一个维度上和为1的概率分布。
              `logits` 将返回这个归一化后的值。

    -   :meth:`sample` 方法需要一个共享的 `total_count` 来进行参数和样本的采样。
    -   :meth:`log_prob` 方法允许每个参数和样本有不同的 `total_count`。

    示例::

        >>> # xdoctest: +SKIP("FIXME: found invalid values")
        >>> m = Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
        >>> x = m.sample()  # 等概率地采样 0, 1, 2, 3
        tensor([ 21.,  24.,  30.,  25.])

        >>> Multinomial(probs=torch.tensor([1., 1., 1., 1.])).log_prob(x)
        tensor([-4.1338])

    Args:
        total_count (int): 试验的次数
        probs (Tensor): 事件的概率
        logits (Tensor): 事件的对数概率（未归一化）
    """
    
    # 参数约束，probs 必须为一个简单形式的单纯形约束，logits 必须为实数向量约束
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    
    # total_count 属性表示试验总次数
    total_count: int

    # 平均值的属性，返回事件的概率乘以总数
    @property
    def mean(self):
        return self.probs * self.total_count

    # 方差的属性，返回试验总次数乘以事件概率乘以（1 - 事件概率）
    @property
    def variance(self):
        return self.total_count * self.probs * (1 - self.probs)

    # 初始化方法，接受 total_count、probs 或 logits 作为参数
    def __init__(self, total_count=1, probs=None, logits=None, validate_args=None):
        # 如果 total_count 不是整数，抛出异常
        if not isinstance(total_count, int):
            raise NotImplementedError("inhomogeneous total_count is not supported")
        
        # 初始化 total_count 属性
        self.total_count = total_count
        
        # 创建 Categorical 分布对象，使用给定的 probs 或 logits 参数
        self._categorical = Categorical(probs=probs, logits=logits)
        
        # 创建 Binomial 分布对象，使用 total_count 和 self.probs 参数
        self._binomial = Binomial(total_count=total_count, probs=self.probs)
        
        # 获取批次形状和事件形状
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]  # 只保留最后一个维度
        # 调用父类的初始化方法，传入批次形状、事件形状和验证参数
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
    # 使用 batch_shape 创建一个新的 Multinomial 分布对象，如果给定则使用 _instance
    def expand(self, batch_shape, _instance=None):
        # 获取一个验证过的 Multinomial 实例
        new = self._get_checked_instance(Multinomial, _instance)
        # 将 batch_shape 转换为 torch.Size 类型
        batch_shape = torch.Size(batch_shape)
        # 将当前对象的 total_count 属性赋值给新对象
        new.total_count = self.total_count
        # 将当前对象的 _categorical 属性按照 batch_shape 进行扩展赋给新对象的 _categorical 属性
        new._categorical = self._categorical.expand(batch_shape)
        # 调用父类 Multinomial 的构造函数，设置新对象的 batch_shape 和 event_shape
        super(Multinomial, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        # 将当前对象的 _validate_args 属性赋给新对象
        new._validate_args = self._validate_args
        # 返回扩展后的新 Multinomial 对象
        return new

    # 创建一个新的对象，直接委托给 _categorical 对象的 _new 方法
    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    # 使用 constraints 模块的 dependent_property 装饰器，定义支持集合的属性
    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self):
        return constraints.multinomial(self.total_count)

    # 返回 _categorical 对象的 logits 属性
    @property
    def logits(self):
        return self._categorical.logits

    # 返回 _categorical 对象的 probs 属性
    @property
    def probs(self):
        return self._categorical.probs

    # 返回 _categorical 对象的 param_shape 属性
    @property
    def param_shape(self):
        return self._categorical.param_shape

    # 从 Multinomial 分布中抽样，sample_shape 控制抽样的形状
    def sample(self, sample_shape=torch.Size()):
        # 将 sample_shape 转换为 torch.Size 类型
        sample_shape = torch.Size(sample_shape)
        # 对 _categorical 对象进行抽样，形状为 (total_count, sample_shape, batch_shape)
        samples = self._categorical.sample(
            torch.Size((self.total_count,)) + sample_shape
        )
        # 将抽样结果的维度顺序从 (total_count, sample_shape, batch_shape) 调整为 (sample_shape, batch_shape, total_count)
        shifted_idx = list(range(samples.dim()))
        shifted_idx.append(shifted_idx.pop(0))
        samples = samples.permute(*shifted_idx)
        # 创建一个与抽样结果相同形状的全零张量 counts，并根据 samples 进行 scatter_add 操作
        counts = samples.new(self._extended_shape(sample_shape)).zero_()
        counts.scatter_add_(-1, samples, torch.ones_like(samples))
        # 将 counts 转换为与 probs 相同类型的张量并返回
        return counts.type_as(self.probs)

    # 计算 Multinomial 分布的熵
    def entropy(self):
        # 创建一个包含 total_count 值的张量 n
        n = torch.tensor(self.total_count)

        # 计算 _categorical 对象的熵
        cat_entropy = self._categorical.entropy()
        term1 = n * cat_entropy - torch.lgamma(n + 1)

        # 枚举二项式分布的支持集合并计算其概率及相应的权重
        support = self._binomial.enumerate_support(expand=False)[1:]
        binomial_probs = torch.exp(self._binomial.log_prob(support))
        weights = torch.lgamma(support + 1)
        term2 = (binomial_probs * weights).sum([0, -1])

        # 返回熵的计算结果
        return term1 + term2

    # 计算给定值在 Multinomial 分布下的对数概率
    def log_prob(self, value):
        # 如果启用参数验证，则验证给定的抽样值是否有效
        if self._validate_args:
            self._validate_sample(value)
        # 广播 logits 和 value，使它们具有兼容的形状
        logits, value = broadcast_all(self.logits, value)
        # 克隆 logits 张量以确保内存格式连续
        logits = logits.clone(memory_format=torch.contiguous_format)
        # 计算总和为 value 的对数阶乘
        log_factorial_n = torch.lgamma(value.sum(-1) + 1)
        # 计算每个值的对数阶乘并在最后一个维度上求和
        log_factorial_xs = torch.lgamma(value + 1).sum(-1)
        # 将 logits 中值为 -inf 且 value 中对应位置为 0 的元素置为 0
        logits[(value == 0) & (logits == -inf)] = 0
        # 计算 logits 与 value 的乘积在最后一个维度上的和，得到对数概率
        log_powers = (logits * value).sum(-1)
        # 返回对数概率的计算结果
        return log_factorial_n - log_factorial_xs + log_powers
```