# `.\pytorch\torch\distributions\one_hot_categorical.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 从 torch.distributions 中导入 constraints 模块
from torch.distributions import constraints
# 从 torch.distributions.categorical 中导入 Categorical 类
from torch.distributions.categorical import Categorical
# 从 torch.distributions.distribution 中导入 Distribution 类
from torch.distributions.distribution import Distribution

# 定义该模块公开的类名列表
__all__ = ["OneHotCategorical", "OneHotCategoricalStraightThrough"]

# 定义 OneHotCategorical 类，继承自 Distribution 类
class OneHotCategorical(Distribution):
    r"""
    创建一个由 `probs` 或 `logits` 参数化的单热分类分布。

    样本是大小为 `probs.size(-1)` 的单热编码向量。

    .. note:: `probs` 参数必须是非负有限的，并且其和不为零，将被归一化为最后一个维度上和为1的概率值。
              `probs` 将返回这个归一化后的值。
              `logits` 参数将被解释为未归一化的对数概率，因此可以是任意实数。
              它同样将被归一化，使得最后一个维度上的概率和为1。
              `logits` 将返回这个归一化后的值。

    参见：:func:`torch.distributions.Categorical` 关于 `probs` 和 `logits` 的规范。

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # 0, 1, 2, 3 各有相等的概率
        tensor([ 0.,  0.,  0.,  1.])

    Args:
        probs (Tensor): 事件概率
        logits (Tensor): 事件对数概率（未归一化）
    """
    
    # 参数约束，probs 必须是一个单纯形（simplex），logits 必须是一个实数向量
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    # 支持约束，表示支持单热编码
    support = constraints.one_hot
    # 指示是否支持枚举支持（enumerate support）
    has_enumerate_support = True

    # 初始化方法
    def __init__(self, probs=None, logits=None, validate_args=None):
        # 使用 Categorical 类初始化 _categorical 成员变量
        self._categorical = Categorical(probs, logits)
        # 获取批次形状和事件形状
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        # 调用父类 Distribution 的初始化方法
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    # 扩展方法，用于创建新的 OneHotCategorical 实例
    def expand(self, batch_shape, _instance=None):
        # 获取一个已检查的实例
        new = self._get_checked_instance(OneHotCategorical, _instance)
        # 将 batch_shape 转换为 torch.Size
        batch_shape = torch.Size(batch_shape)
        # 扩展 _categorical 成员变量
        new._categorical = self._categorical.expand(batch_shape)
        # 调用父类 Distribution 的初始化方法
        super(OneHotCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        # 设置验证参数标志位
        new._validate_args = self._validate_args
        return new

    # 私有方法，返回一个新的 _categorical 实例
    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    # 属性方法，返回 _categorical 的 _param 属性
    @property
    def _param(self):
        return self._categorical._param

    # 属性方法，返回 _categorical 的 probs 属性
    @property
    def probs(self):
        return self._categorical.probs

    # 属性方法，返回 _categorical 的 logits 属性
    @property
    def logits(self):
        return self._categorical.logits

    # 属性方法，返回 _categorical 的 mean 属性，即 probs
    @property
    def mean(self):
        return self._categorical.probs

    # 属性方法，返回 _categorical 的 variance 属性
    @property
    def variance(self):
        return self._categorical.variance
    # 返回分布中的众数，即具有最高概率的类别
    def mode(self):
        probs = self._categorical.probs  # 获取分类分布的概率数组
        mode = probs.argmax(axis=-1)  # 找到概率数组中最大值的索引，即众数
        return torch.nn.functional.one_hot(mode, num_classes=probs.shape[-1]).to(probs)
        # 将众数转换为 one-hot 编码，用于表示分类的结果

    @property
    # 返回分布的方差
    def variance(self):
        return self._categorical.probs * (1 - self._categorical.probs)
        # 计算二项分布的方差，使用 p(1-p) 公式

    @property
    # 返回分布参数的形状
    def param_shape(self):
        return self._categorical.param_shape
        # 返回二项分布参数的形状信息

    # 从分布中抽取样本
    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)  # 将输入的样本形状转换为 torch.Size 对象
        probs = self._categorical.probs  # 获取分类分布的概率数组
        num_events = self._categorical._num_events  # 获取事件的数量
        indices = self._categorical.sample(sample_shape)  # 从分类分布中抽取样本索引
        return torch.nn.functional.one_hot(indices, num_events).to(probs)
        # 将抽取的索引转换为 one-hot 编码，用于表示抽样结果

    # 计算给定值的对数概率
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)  # 如果需要，验证输入值是否有效
        indices = value.max(-1)[1]  # 找到每行中最大值的索引，即预测的类别
        return self._categorical.log_prob(indices)
        # 返回预测类别的对数概率

    # 计算分布的熵
    def entropy(self):
        return self._categorical.entropy()
        # 返回二项分布的熵值

    # 枚举支持的所有可能值
    def enumerate_support(self, expand=True):
        n = self.event_shape[0]  # 获取事件形状的第一个维度，即事件数目
        values = torch.eye(n, dtype=self._param.dtype, device=self._param.device)  # 创建单位矩阵，表示所有可能的类别
        values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))  # 调整矩阵形状以匹配批处理形状
        if expand:
            values = values.expand((n,) + self.batch_shape + (n,))  # 如果需要扩展，则扩展矩阵形状
        return values
        # 返回表示所有可能类别的张量
class OneHotCategoricalStraightThrough(OneHotCategorical):
    r"""
    Creates a reparameterizable :class:`OneHotCategorical` distribution based on the straight-
    through gradient estimator from [1].

    [1] Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    (Bengio et al., 2013)
    """
    # 设置属性，指示这个分布支持 reparameterization trick
    has_rsample = True

    def rsample(self, sample_shape=torch.Size()):
        # 使用 self.sample 方法生成样本
        samples = self.sample(sample_shape)
        # 获取已缓存的类别分布概率（使用 @lazy_property 装饰器缓存）
        probs = self._categorical.probs  # cached via @lazy_property
        # 返回 reparameterization trick 后的样本
        return samples + (probs - probs.detach())
```