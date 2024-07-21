# `.\pytorch\torch\distributions\mixture_same_family.py`

```py
# 使用 mypy 配置选项，允许未标记类型的定义
from typing import Dict  # 导入 Dict 类型用于声明字典类型

import torch  # 导入 PyTorch 库
from torch.distributions import Categorical, constraints  # 导入 Categorical 和 constraints 模块
from torch.distributions.distribution import Distribution  # 导入 Distribution 类

__all__ = ["MixtureSameFamily"]  # 导出模块时的公开接口列表，只包含 MixtureSameFamily 类

class MixtureSameFamily(Distribution):
    r"""
    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all component are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` component) and a component
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.

    Examples::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Construct Gaussian Mixture Model in 1D consisting of 5 equally
        >>> # weighted normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
        >>> gmm = MixtureSameFamily(mix, comp)

        >>> # Construct Gaussian Mixture Model in 2D consisting of 5 equally
        >>> # weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Independent(D.Normal(
        ...          torch.randn(5,2), torch.rand(5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

        >>> # Construct a batch of 3 Gaussian Mixture Models in 2D each
        >>> # consisting of 5 random weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.rand(3,5))
        >>> comp = D.Independent(D.Normal(
        ...         torch.randn(3,5,2), torch.rand(3,5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

    Args:
        mixture_distribution: `torch.distributions.Categorical`-like
            instance. Manages the probability of selecting component.
            The number of categories must match the rightmost batch
            dimension of the `component_distribution`. Must have either
            scalar `batch_shape` or `batch_shape` matching
            `component_distribution.batch_shape[:-1]`
        component_distribution: `torch.distributions.Distribution`-like
            instance. Right-most batch dimension indexes component.
    """
    arg_constraints: Dict[str, constraints.Constraint] = {}  # 参数约束的字典，初始化为空字典
    has_rsample = False  # 不支持 rsample 方法

    def __init__(
        self, mixture_distribution, component_distribution, validate_args=None
        # 构造函数，接受混合分布和组件分布作为参数，validate_args 参数默认为 None
        ):
        ):
            self._mixture_distribution = mixture_distribution
            self._component_distribution = component_distribution

            if not isinstance(self._mixture_distribution, Categorical):
                raise ValueError(
                    " The Mixture distribution needs to be an "
                    " instance of torch.distributions.Categorical"
                )

            if not isinstance(self._component_distribution, Distribution):
                raise ValueError(
                    "The Component distribution need to be an "
                    "instance of torch.distributions.Distribution"
                )

            # Check that batch size matches
            mdbs = self._mixture_distribution.batch_shape
            cdbs = self._component_distribution.batch_shape[:-1]
            for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
                if size1 != 1 and size2 != 1 and size1 != size2:
                    raise ValueError(
                        f"`mixture_distribution.batch_shape` ({mdbs}) is not "
                        "compatible with `component_distribution."
                        f"batch_shape`({cdbs})"
                    )

            # Check that the number of mixture component matches
            km = self._mixture_distribution.logits.shape[-1]
            kc = self._component_distribution.batch_shape[-1]
            if km is not None and kc is not None and km != kc:
                raise ValueError(
                    f"`mixture_distribution component` ({km}) does not"
                    " equal `component_distribution.batch_shape[-1]`"
                    f" ({kc})"
                )
            self._num_component = km

            event_shape = self._component_distribution.event_shape
            self._event_ndims = len(event_shape)
            super().__init__(
                batch_shape=cdbs, event_shape=event_shape, validate_args=validate_args
            )

        def expand(self, batch_shape, _instance=None):
            # Convert batch_shape to torch.Size object
            batch_shape = torch.Size(batch_shape)
            batch_shape_comp = batch_shape + (self._num_component,)
            new = self._get_checked_instance(MixtureSameFamily, _instance)
            # Expand the component distribution's batch shape
            new._component_distribution = self._component_distribution.expand(
                batch_shape_comp
            )
            # Expand the mixture distribution's batch shape
            new._mixture_distribution = self._mixture_distribution.expand(batch_shape)
            new._num_component = self._num_component
            new._event_ndims = self._event_ndims
            event_shape = new._component_distribution.event_shape
            super(MixtureSameFamily, new).__init__(
                batch_shape=batch_shape, event_shape=event_shape, validate_args=False
            )
            new._validate_args = self._validate_args
            return new

        @constraints.dependent_property
        def support(self):
            # FIXME this may have the wrong shape when support contains batched
            # parameters
            # Return the support of the component distribution
            return self._component_distribution.support

        @property
        def mixture_distribution(self):
            # Return the mixture distribution
            return self._mixture_distribution

        @property
    # 返回对象内部的 `_component_distribution` 属性值
    def component_distribution(self):
        return self._component_distribution

    @property
    def mean(self):
        # 获取经过填充混合分布概率维度后的概率值
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        # 计算加权平均值，dim 参数指定了在哪些维度上求和
        return torch.sum(
            probs * self.component_distribution.mean, dim=-1 - self._event_ndims
        )  # [B, E]

    @property
    def variance(self):
        # 根据总方差定理：Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        # 计算在给定条件下的均值方差
        mean_cond_var = torch.sum(
            probs * self.component_distribution.variance, dim=-1 - self._event_ndims
        )
        # 计算在给定条件下的方差均值
        var_cond_mean = torch.sum(
            probs * (self.component_distribution.mean - self._pad(self.mean)).pow(2.0),
            dim=-1 - self._event_ndims,
        )
        return mean_cond_var + var_cond_mean

    def cdf(self, x):
        # 对输入数据进行填充
        x = self._pad(x)
        # 计算组件分布的累积分布函数
        cdf_x = self.component_distribution.cdf(x)
        # 获取混合分布的概率
        mix_prob = self.mixture_distribution.probs

        # 返回加权累积分布函数结果
        return torch.sum(cdf_x * mix_prob, dim=-1)

    def log_prob(self, x):
        # 如果需要验证参数，则进行样本验证
        if self._validate_args:
            self._validate_sample(x)
        # 对输入数据进行填充
        x = self._pad(x)
        # 计算组件分布的对数概率
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        # 对混合分布的 logits 进行对数 softmax
        log_mix_prob = torch.log_softmax(
            self.mixture_distribution.logits, dim=-1
        )  # [B, k]
        # 返回对数概率的对数和
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def sample(self, sample_shape=torch.Size()):
        # 禁用梯度计算
        with torch.no_grad():
            # 获取样本形状和批次形状的长度
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # 混合样本 [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # 组件样本 [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)

            # 沿着 k 维度进行聚集
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1))
            )
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es
            )

            # 使用混合样本聚集组件样本
            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)

    def _pad(self, x):
        # 在最后一个事件维度上增加一个维度
        return x.unsqueeze(-1 - self._event_ndims)

    def _pad_mixture_dimensions(self, x):
        # 获取分布批次维度数和混合分布批次维度数
        dist_batch_ndims = len(self.batch_shape)
        cat_batch_ndims = len(self.mixture_distribution.batch_shape)
        # 计算需要填充的维度数
        pad_ndims = 0 if cat_batch_ndims == 1 else dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        # 重新形状化输入张量，以填充给定维度数
        x = x.reshape(
            xs[:-1]
            + torch.Size(pad_ndims * [1])
            + xs[-1:]
            + torch.Size(self._event_ndims * [1])
        )
        return x
    # 定义类的特殊方法 __repr__()，用于返回对象的“official”字符串表示
    def __repr__(self):
        # 格式化字符串，生成表示对象参数的字符串
        args_string = (
            f"\n  {self.mixture_distribution},\n  {self.component_distribution}"
        )
        # 返回对象的字符串表示，格式为 "MixtureSameFamily(参数字符串)"
        return "MixtureSameFamily" + "(" + args_string + ")"
```