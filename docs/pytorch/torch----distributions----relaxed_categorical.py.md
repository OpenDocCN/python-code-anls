# `.\pytorch\torch\distributions\relaxed_categorical.py`

```
# mypy: allow-untyped-defs
# 导入 PyTorch 库
import torch
# 导入概率分布约束模块
from torch.distributions import constraints
# 导入 Categorical 分布类
from torch.distributions.categorical import Categorical
# 导入分布基类 Distribution
from torch.distributions.distribution import Distribution
# 导入转换后分布类 TransformedDistribution
from torch.distributions.transformed_distribution import TransformedDistribution
# 导入指数变换类 ExpTransform
from torch.distributions.transforms import ExpTransform
# 导入工具函数 broadcast_all 和 clamp_probs
from torch.distributions.utils import broadcast_all, clamp_probs

# 导出模块列表
__all__ = ["ExpRelaxedCategorical", "RelaxedOneHotCategorical"]

# 创建 ExpRelaxedCategorical 类，继承自 Distribution 类
class ExpRelaxedCategorical(Distribution):
    """
    创建一个 ExpRelaxedCategorical 分布，由参数 temperature 和其中一个参数 probs 或 logits（但不能同时有）决定。
    返回一个在单纯形上的点的对数。基于 OneHotCategorical 的接口。

    实现基于 [1]。

    参见：torch.distributions.OneHotCategorical

    Args:
        temperature (Tensor): 松弛温度
        probs (Tensor): 事件概率
        logits (Tensor): 每个事件的未归一化对数概率

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al., 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al., 2017)
    """

    # 参数约束定义
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    # 分布支持定义，实际支持是其子流形
    support = (
        constraints.real_vector
    )  # 实际上真正的支持是这个的一个子流形。
    has_rsample = True

    # 初始化方法
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        # 创建一个 Categorical 对象
        self._categorical = Categorical(probs, logits)
        # 设置温度参数
        self.temperature = temperature
        # 获取批次形状和事件形状
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        # 调用父类的初始化方法
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    # 扩展方法
    def expand(self, batch_shape, _instance=None):
        # 获取已检查的实例
        new = self._get_checked_instance(ExpRelaxedCategorical, _instance)
        # 转换为 torch.Size 类型的批次形状
        batch_shape = torch.Size(batch_shape)
        # 设置温度参数
        new.temperature = self.temperature
        # 扩展内部的 Categorical 对象
        new._categorical = self._categorical.expand(batch_shape)
        # 调用父类的初始化方法
        super(ExpRelaxedCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        # 设置验证参数
        new._validate_args = self._validate_args
        return new

    # 内部创建新对象的方法
    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    # 参数形状属性
    @property
    def param_shape(self):
        return self._categorical.param_shape

    # logits 属性
    @property
    def logits(self):
        return self._categorical.logits

    # probs 属性
    @property
    def probs(self):
        return self._categorical.probs
    # 根据指定的样本形状生成样本，返回采样分数
    def rsample(self, sample_shape=torch.Size()):
        # 计算扩展后的形状
        shape = self._extended_shape(sample_shape)
        # 生成服从均匀分布的随机数，并根据概率截断
        uniforms = clamp_probs(
            torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        )
        # 生成服从 Gumbel 分布的随机数
        gumbels = -((-(uniforms.log())).log())
        # 计算得分，应用 Gumbel 分布到 logits 上，再除以温度
        scores = (self.logits + gumbels) / self.temperature
        # 返回归一化后的分数
        return scores - scores.logsumexp(dim=-1, keepdim=True)

    # 计算给定值的对数概率
    def log_prob(self, value):
        # 获取分类变量的数量 K
        K = self._categorical._num_events
        # 如果需要验证参数，则验证给定值的样本
        if self._validate_args:
            self._validate_sample(value)
        # 广播 logits 和 value，使它们的形状相容
        logits, value = broadcast_all(self.logits, value)
        # 计算对数尺度，使用温度参数进行计算
        log_scale = torch.full_like(
            self.temperature, float(K)
        ).lgamma() - self.temperature.log().mul(-(K - 1))
        # 计算得分，应用 logits 到 value 乘以温度
        score = logits - value.mul(self.temperature)
        # 对得分进行归一化，然后对最后一个维度求和
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        # 返回对数概率分数加上对数尺度
        return score + log_scale
class RelaxedOneHotCategorical(TransformedDistribution):
    r"""
    Creates a RelaxedOneHotCategorical distribution parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`.
    This is a relaxed version of the :class:`OneHotCategorical` distribution, so
    its samples are on simplex, and are reparametrizable.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = RelaxedOneHotCategorical(torch.tensor([2.2]),
        ...                              torch.tensor([0.1, 0.2, 0.3, 0.4]))
        >>> m.sample()
        tensor([ 0.1294,  0.2324,  0.3859,  0.2523])

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event
    """
    # 约束参数：probs 必须为单纯形，logits 必须为实数向量
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    # 分布的支持集：单纯形
    support = constraints.simplex
    # 是否具有 rsample 方法
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        # 基础分布为 ExpRelaxedCategorical，根据输入的参数构造
        base_dist = ExpRelaxedCategorical(
            temperature, probs, logits, validate_args=validate_args
        )
        # 调用父类 TransformedDistribution 的构造函数进行初始化
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # 创建一个扩展后的实例，用于支持批处理
        new = self._get_checked_instance(RelaxedOneHotCategorical, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def temperature(self):
        # 返回基础分布中的温度参数
        return self.base_dist.temperature

    @property
    def logits(self):
        # 返回基础分布中的 logits 参数
        return self.base_dist.logits

    @property
    def probs(self):
        # 返回基础分布中的 probs 参数
        return self.base_dist.probs
```