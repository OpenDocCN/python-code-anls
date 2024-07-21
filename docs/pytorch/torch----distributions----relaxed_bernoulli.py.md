# `.\pytorch\torch\distributions\relaxed_bernoulli.py`

```py
# 设置类型检查允许未标注的定义
# 从numbers模块导入Number类
from numbers import Number

# 导入torch库
import torch
# 从torch.distributions模块导入constraints约束对象
from torch.distributions import constraints
# 从torch.distributions.distribution模块导入Distribution类
from torch.distributions.distribution import Distribution
# 从torch.distributions.transformed_distribution模块导入TransformedDistribution类
from torch.distributions.transformed_distribution import TransformedDistribution
# 从torch.distributions.transforms模块导入SigmoidTransform类
from torch.distributions.transforms import SigmoidTransform
# 从torch.distributions.utils模块导入一系列函数和装饰器
from torch.distributions.utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)

# 声明模块对外暴露的类名列表
__all__ = ["LogitRelaxedBernoulli", "RelaxedBernoulli"]


class LogitRelaxedBernoulli(Distribution):
    r"""
    创建一个LogitRelaxedBernoulli分布，参数为`probs`或`logits`（但不能同时有），它是RelaxedBernoulli分布的logit版本。

    样本是在(0, 1)内值的logits。详见[1]了解更多细节。

    Args:
        temperature (Tensor): 松弛温度
        probs (Number, Tensor): 采样`1`的概率
        logits (Number, Tensor): 采样`1`的对数几率

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random
    Variables (Maddison et al., 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al., 2017)
    """
    # 参数的约束条件，probs必须在单位间隔上，logits必须为实数
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    # 分布的支持集合为实数集
    support = constraints.real

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        # 初始化函数，设定温度参数
        self.temperature = temperature
        # 检查probs和logits是否同时为None或同时非None，否则抛出异常
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        # 如果probs非None，广播probs使其与logits形状相同
        if probs is not None:
            is_scalar = isinstance(probs, Number)
            (self.probs,) = broadcast_all(probs)
        else:
            # 如果logits非None，广播logits使其与probs形状相同
            is_scalar = isinstance(logits, Number)
            (self.logits,) = broadcast_all(logits)
        # 将参数保存为self._param，并根据是否为probs选择相应的参数
        self._param = self.probs if probs is not None else self.logits
        # 如果参数是标量，则batch_shape为空torch.Size()，否则为_param的形状
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        # 调用父类的初始化方法，传入batch_shape和validate_args参数
        super().__init__(batch_shape, validate_args=validate_args)

    # 扩展分布的batch_shape
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogitRelaxedBernoulli, _instance)
        # 将batch_shape转换为torch.Size类型
        batch_shape = torch.Size(batch_shape)
        # 将新实例的温度设为当前实例的温度
        new.temperature = self.temperature
        # 如果存在self.probs属性，扩展新实例的probs属性为指定的batch_shape
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        # 如果存在self.logits属性，扩展新实例的logits属性为指定的batch_shape
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        # 调用父类的初始化方法，传入扩展后的batch_shape和validate_args=False
        super(LogitRelaxedBernoulli, new).__init__(batch_shape, validate_args=False)
        # 设置新实例的_validate_args与当前实例一致
        new._validate_args = self._validate_args
        return new

    # 创建新的torch.Tensor对象
    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    # 延迟加载的属性，返回logits属性对应的值
    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    # 延迟加载的属性
    # 计算以对数形式表示的概率
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    # 返回参数的形状
    @property
    def param_shape(self):
        return self._param.size()

    # 从分布中生成样本
    def rsample(self, sample_shape=torch.Size()):
        # 计算扩展后的形状
        shape = self._extended_shape(sample_shape)
        # 获取概率并截断到有效范围
        probs = clamp_probs(self.probs.expand(shape))
        # 生成均匀分布的随机数并截断到有效范围
        uniforms = clamp_probs(
            torch.rand(shape, dtype=probs.dtype, device=probs.device)
        )
        # 应用逆变换采样，并考虑温度参数
        return (
            uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()
        ) / self.temperature

    # 计算给定值的对数概率密度
    def log_prob(self, value):
        # 如果需要验证参数，则验证样本的有效性
        if self._validate_args:
            self._validate_sample(value)
        # 广播操作，使logits和value具有相同的形状
        logits, value = broadcast_all(self.logits, value)
        # 计算logits和value乘以温度后的差异
        diff = logits - value.mul(self.temperature)
        # 返回以对数形式表示的概率密度
        return self.temperature.log() + diff - 2 * diff.exp().log1p()
# 定义一个 RelaxedBernoulli 类，继承自 TransformedDistribution 类
class RelaxedBernoulli(TransformedDistribution):
    r"""
    创建一个 RelaxedBernoulli 分布，由参数 :attr:`temperature`、:attr:`probs` 或 :attr:`logits`
    （但不能同时包含两者）来确定。这是 `Bernoulli` 分布的松弛版本，
    其值位于 (0, 1) 区间内，并支持可重参数化的样本生成。

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = RelaxedBernoulli(torch.tensor([2.2]),
        ...                      torch.tensor([0.1, 0.2, 0.3, 0.99]))
        >>> m.sample()
        tensor([ 0.2951,  0.3442,  0.8918,  0.9021])

    Args:
        temperature (Tensor): 松弛温度
        probs (Number, Tensor): 采样为 `1` 的概率
        logits (Number, Tensor): 采样为 `1` 的对数几率
    """
    
    # 参数约束定义，指定 `probs` 在单位区间内，`logits` 为实数
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    
    # 分布支持的约束条件，仅在单位区间内
    support = constraints.unit_interval
    
    # 标记支持重参数化的样本生成
    has_rsample = True

    # 初始化方法
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        # 创建 LogitRelaxedBernoulli 分布作为基础分布
        base_dist = LogitRelaxedBernoulli(temperature, probs, logits)
        # 调用父类 TransformedDistribution 的初始化方法
        super().__init__(base_dist, SigmoidTransform(), validate_args=validate_args)

    # 扩展方法，返回新的实例并确保正确类型
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedBernoulli, _instance)
        return super().expand(batch_shape, _instance=new)

    # 返回温度参数的属性方法
    @property
    def temperature(self):
        return self.base_dist.temperature

    # 返回对数几率参数的属性方法
    @property
    def logits(self):
        return self.base_dist.logits

    # 返回概率参数的属性方法
    @property
    def probs(self):
        return self.base_dist.probs
```