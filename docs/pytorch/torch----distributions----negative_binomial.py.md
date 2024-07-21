# `.\pytorch\torch\distributions\negative_binomial.py`

```
# 引入torch库，用于深度学习任务
import torch
# 引入torch.nn.functional中的F，提供了许多用于神经网络的函数
import torch.nn.functional as F
# 从torch.distributions中引入constraints，用于定义分布的约束条件
from torch.distributions import constraints
# 从torch.distributions.distribution中引入Distribution类，表示概率分布的基类
from torch.distributions.distribution import Distribution
# 从torch.distributions.utils中引入多个工具函数
from torch.distributions.utils import (
    broadcast_all,    # 用于广播输入的张量
    lazy_property,    # 延迟加载属性的装饰器
    logits_to_probs,  # 将logits转换为概率
    probs_to_logits   # 将概率转换为logits
)

# 定义导出的模块列表，只包含"NegativeBinomial"
__all__ = ["NegativeBinomial"]

# 定义NegativeBinomial类，继承自Distribution类
class NegativeBinomial(Distribution):
    r"""
    创建负二项分布，即在达到指定失败次数前的成功独立伯努利试验次数的分布。每次伯努利试验成功的概率为 :attr:`probs`。

    Args:
        total_count (float or Tensor): 非负的伯努利试验失败次数，尽管对于实数值的计数，分布仍然有效
        probs (Tensor): 成功的事件概率，取值在半开区间[0, 1)
        logits (Tensor): 成功概率的对数几率
    """
    
    # 参数约束，指定每个参数的限制条件
    arg_constraints = {
        "total_count": constraints.greater_than_eq(0),  # total_count必须大于等于0
        "probs": constraints.half_open_interval(0.0, 1.0),  # probs必须在半开区间[0.0, 1.0)
        "logits": constraints.real,  # logits必须是实数
    }
    
    # 支持的值的约束，这里是非负整数
    support = constraints.nonnegative_integer

    # 初始化方法，定义了NegativeBinomial分布的参数和验证参数的有效性
    def __init__(self, total_count, probs=None, logits=None, validate_args=None):
        # 如果probs和logits同时为None或者同时不为None，抛出错误
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            # 使用broadcast_all函数，将total_count和probs广播到相同的形状
            (
                self.total_count,
                self.probs,
            ) = broadcast_all(total_count, probs)
            # 将total_count转换为和probs相同类型的张量
            self.total_count = self.total_count.type_as(self.probs)
        else:
            # 使用broadcast_all函数，将total_count和logits广播到相同的形状
            (
                self.total_count,
                self.logits,
            ) = broadcast_all(total_count, logits)
            # 将total_count转换为和logits相同类型的张量
            self.total_count = self.total_count.type_as(self.logits)

        # 根据输入的probs或logits设置_param属性
        self._param = self.probs if probs is not None else self.logits
        # 获取批次形状，并传递给父类Distribution的初始化方法
        batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    # 扩展方法，用于在给定的批次形状下扩展分布
    def expand(self, batch_shape, _instance=None):
        # 获取一个新的实例，确保类型与NegativeBinomial匹配
        new = self._get_checked_instance(NegativeBinomial, _instance)
        # 将扩展后的批次形状转换为torch.Size对象
        batch_shape = torch.Size(batch_shape)
        # 将total_count属性扩展到新的批次形状
        new.total_count = self.total_count.expand(batch_shape)
        # 如果当前实例有probs属性，则将其扩展到新的批次形状
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        # 如果当前实例有logits属性，则将其扩展到新的批次形状
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        # 调用父类Distribution的初始化方法，并传递validate_args=False参数
        super(NegativeBinomial, new).__init__(batch_shape, validate_args=False)
        # 将_validate_args属性设置为当前实例的_validate_args
        new._validate_args = self._validate_args
        # 返回扩展后的新实例
        return new

    # _new方法，用于创建与当前参数类型相同的新张量
    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    # 平均值属性，计算并返回分布的平均值
    @property
    def mean(self):
        return self.total_count * torch.exp(self.logits)

    # 属性
    # 计算众数（mode）的方法，根据总计数和 logits 计算指数，取其整数部分，并确保不小于 0
    def mode(self):
        return ((self.total_count - 1) * self.logits.exp()).floor().clamp(min=0.0)

    # 计算方差（variance）的属性方法，根据均值和 sigmoid 函数处理的 logits 计算得出
    @property
    def variance(self):
        return self.mean / torch.sigmoid(-self.logits)

    # logits 的延迟加载属性方法，将概率转换为 logits
    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    # 概率的延迟加载属性方法，将 logits 转换为概率
    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    # 参数形状的属性方法，返回参数（_param）的尺寸
    @property
    def param_shape(self):
        return self._param.size()

    # Gamma 分布的延迟加载属性方法，使用 total_count 和 logits 计算分布的参数
    @lazy_property
    def _gamma(self):
        # 注意避免验证（validate_args），因为 self.total_count 可能为零
        return torch.distributions.Gamma(
            concentration=self.total_count,
            rate=torch.exp(-self.logits),
            validate_args=False,
        )

    # 采样方法，使用 Gamma 分布采样，并返回泊松分布的样本
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return torch.poisson(rate)

    # 计算对数概率的方法，根据总计数和 value 计算未归一化的对数概率，然后减去归一化项
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        log_unnormalized_prob = self.total_count * F.logsigmoid(
            -self.logits
        ) + value * F.logsigmoid(self.logits)

        log_normalization = (
            -torch.lgamma(self.total_count + value)
            + torch.lgamma(1.0 + value)
            + torch.lgamma(self.total_count)
        )
        # 处理特殊情况，当 self.total_count 和 value 都为 0 时，概率为 1，但 lgamma(0) 为无穷大
        # 使用不在原地修改张量的方法来处理这种情况，以允许 Jit 编译
        log_normalization = log_normalization.masked_fill(
            self.total_count + value == 0.0, 0.0
        )

        return log_unnormalized_prob - log_normalization
```