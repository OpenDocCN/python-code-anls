# `.\pytorch\torch\distributions\transformed_distribution.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型定义
from typing import Dict

# 引入 PyTorch 库
import torch
# 从 PyTorch 分布模块中引入约束
from torch.distributions import constraints
# 从 PyTorch 分布模块中引入基类 Distribution
from torch.distributions.distribution import Distribution
# 从 PyTorch 分布模块中引入 Independent 类
from torch.distributions.independent import Independent
# 从 PyTorch 分布模块中引入变换类 Transform 和组合变换类 ComposeTransform
from torch.distributions.transforms import ComposeTransform, Transform
# 从 PyTorch 分布模块中引入工具函数 _sum_rightmost
from torch.distributions.utils import _sum_rightmost

# 声明模块中公开的类和函数列表
__all__ = ["TransformedDistribution"]

# 定义 TransformedDistribution 类，扩展自 Distribution 基类
class TransformedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied::

        X ~ BaseDistribution
        Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
        log p(Y) = log p(X) + log |det (dX/dY)|

    Note that the ``.event_shape`` of a :class:`TransformedDistribution` is the
    maximum shape of its base distribution and its transforms, since transforms
    can introduce correlations among events.

    An example for the usage of :class:`TransformedDistribution` would be::

        # Building a Logistic Distribution
        # X ~ Uniform(0, 1)
        # f = a + b * logit(X)
        # Y ~ f(X) ~ Logistic(a, b)
        base_distribution = Uniform(0, 1)
        transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
        logistic = TransformedDistribution(base_distribution, transforms)

    For more examples, please look at the implementations of
    :class:`~torch.distributions.gumbel.Gumbel`,
    :class:`~torch.distributions.half_cauchy.HalfCauchy`,
    :class:`~torch.distributions.half_normal.HalfNormal`,
    :class:`~torch.distributions.log_normal.LogNormal`,
    :class:`~torch.distributions.pareto.Pareto`,
    :class:`~torch.distributions.weibull.Weibull`,
    :class:`~torch.distributions.relaxed_bernoulli.RelaxedBernoulli` and
    :class:`~torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`
    """

    # 参数约束字典为空字典
    arg_constraints: Dict[str, constraints.Constraint] = {}
    def __init__(self, base_distribution, transforms, validate_args=None):
        # 如果 transforms 是单个 Transform 对象，则将其放入列表中
        if isinstance(transforms, Transform):
            self.transforms = [
                transforms,
            ]
        # 如果 transforms 是列表，则检查列表中的每个元素是否都是 Transform 对象
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError(
                    "transforms must be a Transform or a list of Transforms"
                )
            self.transforms = transforms
        else:
            # 如果 transforms 不是 Transform 或列表，则引发 ValueError 异常
            raise ValueError(
                f"transforms must be a Transform or list, but was {transforms}"
            )

        # 根据 transforms 调整 base_distribution 的形状。
        # 计算 base_distribution 的 batch_shape 和 event_shape 的总和
        base_shape = base_distribution.batch_shape + base_distribution.event_shape
        base_event_dim = len(base_distribution.event_shape)
        
        # 创建组合变换对象，将 transforms 应用于 base_distribution
        transform = ComposeTransform(self.transforms)
        
        # 检查 base_distribution 的形状是否满足 transform 要求的最小形状
        if len(base_shape) < transform.domain.event_dim:
            raise ValueError(
                f"base_distribution needs to have shape with size at least {transform.domain.event_dim}, but got {base_shape}."
            )
        
        # 计算 transform 应用后的 forward_shape 和 inverse_shape
        forward_shape = transform.forward_shape(base_shape)
        expanded_base_shape = transform.inverse_shape(forward_shape)
        
        # 如果 transform 应用后的形状与原始形状不同，则调整 base_distribution 的批量形状
        if base_shape != expanded_base_shape:
            base_batch_shape = expanded_base_shape[
                : len(expanded_base_shape) - base_event_dim
            ]
            base_distribution = base_distribution.expand(base_batch_shape)
        
        # 根据 transform 的 event_dim 调整 base_distribution 的独立性
        reinterpreted_batch_ndims = transform.domain.event_dim - base_event_dim
        if reinterpreted_batch_ndims > 0:
            base_distribution = Independent(
                base_distribution, reinterpreted_batch_ndims
            )
        
        # 将最终处理后的 base_distribution 存储在 self.base_dist 中
        self.base_dist = base_distribution

        # 计算最终的 batch_shape 和 event_shape
        transform_change_in_event_dim = (
            transform.codomain.event_dim - transform.domain.event_dim
        )
        event_dim = max(
            transform.codomain.event_dim,  # transform 影响的事件维度
            base_event_dim + transform_change_in_event_dim,  # base_distribution 影响的事件维度
        )
        
        # 断言 forward_shape 的长度至少为 event_dim
        assert len(forward_shape) >= event_dim
        
        # 根据 event_dim 分割 forward_shape，得到 batch_shape 和 event_shape
        cut = len(forward_shape) - event_dim
        batch_shape = forward_shape[:cut]
        event_shape = forward_shape[cut:]
        
        # 调用父类的初始化方法，传递计算得到的 batch_shape、event_shape 和验证参数
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
    # 扩展当前转换分布对象，返回一个新的实例
    def expand(self, batch_shape, _instance=None):
        # 获取一个经过检查的 TransformedDistribution 实例
        new = self._get_checked_instance(TransformedDistribution, _instance)
        # 将 batch_shape 转换为 torch.Size 对象
        batch_shape = torch.Size(batch_shape)
        # 计算新的形状，由 batch_shape 和当前事件形状组成
        shape = batch_shape + self.event_shape
        # 对当前的变换列表进行反向迭代
        for t in reversed(self.transforms):
            # 对形状应用逆变换
            shape = t.inverse_shape(shape)
        # 提取基础分布的批处理形状
        base_batch_shape = shape[: len(shape) - len(self.base_dist.event_shape)]
        # 扩展基础分布并赋值给新实例的 base_dist
        new.base_dist = self.base_dist.expand(base_batch_shape)
        # 将当前对象的变换列表赋值给新实例
        new.transforms = self.transforms
        # 调用父类的构造函数初始化新实例
        super(TransformedDistribution, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        # 继承当前对象的参数验证标志
        new._validate_args = self._validate_args
        # 返回新实例
        return new

    @constraints.dependent_property(is_discrete=False)
    # 返回支持的约束条件，如果没有变换，则返回基础分布的支持
    def support(self):
        if not self.transforms:
            return self.base_dist.support
        # 否则返回最后一个变换的值域作为支持
        support = self.transforms[-1].codomain
        # 如果事件形状长度大于支持的事件维度，则创建独立约束
        if len(self.event_shape) > support.event_dim:
            support = constraints.independent(
                support, len(self.event_shape) - support.event_dim
            )
        return support

    @property
    # 返回基础分布是否支持 rsample 方法
    def has_rsample(self):
        return self.base_dist.has_rsample

    # 生成样本，首先从基础分布中抽取样本，然后依次应用变换
    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            # 从基础分布中抽取样本
            x = self.base_dist.sample(sample_shape)
            # 对每个变换依次应用 transform
            for transform in self.transforms:
                x = transform(x)
            return x

    # 生成可重参数化的样本，首先从基础分布中抽取可重参数化的样本，然后依次应用变换
    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        # 从基础分布中抽取可重参数化的样本
        x = self.base_dist.rsample(sample_shape)
        # 对每个变换依次应用 transform
        for transform in self.transforms:
            x = transform(x)
        return x
    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        # 如果启用参数验证，则验证样本值的有效性
        if self._validate_args:
            self._validate_sample(value)
        # 获取事件维度
        event_dim = len(self.event_shape)
        # 初始化对数概率为0
        log_prob = 0.0
        # 将值赋给变量y
        y = value
        # 对每个变换进行反向遍历
        for transform in reversed(self.transforms):
            # 使用变换的逆函数计算x
            x = transform.inv(y)
            # 更新事件维度
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            # 更新对数概率，减去变换的对数绝对值行列式的右侧和
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.domain.event_dim,
            )
            # 更新y为x，用于下一个循环
            y = x

        # 添加基础分布对y的对数概率，更新事件维度
        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(y), event_dim - len(self.base_dist.event_shape)
        )
        # 返回计算得到的总对数概率
        return log_prob

    def _monotonize_cdf(self, value):
        """
        This conditionally flips ``value -> 1-value`` to ensure :meth:`cdf` is
        monotone increasing.
        """
        # 初始化变量sign为1
        sign = 1
        # 对每个变换计算其符号的乘积
        for transform in self.transforms:
            sign = sign * transform.sign
        # 如果sign是整数且等于1，则直接返回value
        if isinstance(sign, int) and sign == 1:
            return value
        # 否则应用条件翻转，以确保累积分布函数是单调递增的
        return sign * (value - 0.5) + 0.5

    def cdf(self, value):
        """
        Computes the cumulative distribution function by inverting the
        transform(s) and computing the score of the base distribution.
        """
        # 对每个变换进行反向遍历，应用其逆函数到value上
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        # 如果启用参数验证，则验证基础分布在value处的样本值的有效性
        if self._validate_args:
            self.base_dist._validate_sample(value)
        # 计算基础分布在value处的累积分布函数值
        value = self.base_dist.cdf(value)
        # 对累积分布函数值应用单调化处理
        value = self._monotonize_cdf(value)
        # 返回处理后的累积分布函数值
        return value

    def icdf(self, value):
        """
        Computes the inverse cumulative distribution function using
        transform(s) and computing the score of the base distribution.
        """
        # 对累积分布函数值进行单调化处理
        value = self._monotonize_cdf(value)
        # 计算基础分布在单调化处理后值处的逆累积分布函数值
        value = self.base_dist.icdf(value)
        # 对每个变换应用其函数到value上，计算原始输入值
        for transform in self.transforms:
            value = transform(value)
        # 返回原始输入值
        return value
```