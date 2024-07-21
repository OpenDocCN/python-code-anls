# `.\pytorch\torch\distributions\__init__.py`

```py
r"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functions. This allows the construction of stochastic computation
graphs and stochastic gradient estimators for optimization. This package
generally follows the design of the `TensorFlow Distributions`_ package.

.. _`TensorFlow Distributions`:
    https://arxiv.org/abs/1711.10604

It is not possible to directly backpropagate through random samples. However,
there are two main methods for creating surrogate functions that can be
backpropagated through. These are the score function estimator/likelihood ratio
estimator/REINFORCE and the pathwise derivative estimator. REINFORCE is commonly
seen as the basis for policy gradient methods in reinforcement learning, and the
pathwise derivative estimator is commonly seen in the reparameterization trick
in variational autoencoders. Whilst the score function only requires the value
of samples :math:`f(x)`, the pathwise derivative requires the derivative
:math:`f'(x)`. The next sections discuss these two in a reinforcement learning
example. For more details see
`Gradient Estimation Using Stochastic Computation Graphs`_ .

.. _`Gradient Estimation Using Stochastic Computation Graphs`:
     https://arxiv.org/abs/1506.05254

Score function
^^^^^^^^^^^^^^

When the probability density function is differentiable with respect to its
parameters, we only need :meth:`~torch.distributions.Distribution.sample` and
:meth:`~torch.distributions.Distribution.log_prob` to implement REINFORCE:

.. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta`.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function. Note that we use a negative because optimizers use gradient
descent, whilst the rule above assumes gradient ascent. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()

Pathwise derivative
^^^^^^^^^^^^^^^^^^^

The other way to implement these stochastic/policy gradients would be to use the
reparameterization trick from the
:meth:`~torch.distributions.Distribution.rsample` method, where the
parameterized random variable can be constructed via a parameterized
deterministic function of a parameter-free random variable. The reparameterized
sample therefore becomes differentiable. The code for implementing the pathwise
derivative would be as follows::
    # 使用策略网络计算当前状态的参数
    params = policy_network(state)
    # 根据策略网络输出的参数，创建一个正态分布对象
    m = Normal(*params)
    # 从正态分布中采样一个动作，因为该分布支持采样操作（.rsample()）
    action = m.rsample()
    # 执行环境的下一步操作，传入当前采样的动作，得到下一状态和奖励
    next_state, reward = env.step(action)  # 假设奖励是可微分的
    # 计算损失函数为奖励的负数（这里假设是最大化奖励）
    loss = -reward
    # 反向传播计算损失函数关于策略网络参数的梯度
    loss.backward()
# 导入必要的概率分布类和函数
from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .chi2 import Chi2
from .continuous_bernoulli import ContinuousBernoulli
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exp_family import ExponentialFamily
from .exponential import Exponential
from .fishersnedecor import FisherSnedecor
from .gamma import Gamma
from .geometric import Geometric
from .gumbel import Gumbel
from .half_cauchy import HalfCauchy
from .half_normal import HalfNormal
from .independent import Independent
from .inverse_gamma import InverseGamma
from .kl import _add_kl_info, kl_divergence, register_kl
from .kumaraswamy import Kumaraswamy
from .laplace import Laplace
from .lkj_cholesky import LKJCholesky
from .log_normal import LogNormal
from .logistic_normal import LogisticNormal
from .lowrank_multivariate_normal import LowRankMultivariateNormal
from .mixture_same_family import MixtureSameFamily
from .multinomial import Multinomial
from .multivariate_normal import MultivariateNormal
from .negative_binomial import NegativeBinomial
from .normal import Normal
from .one_hot_categorical import OneHotCategorical, OneHotCategoricalStraightThrough
from .pareto import Pareto
from .poisson import Poisson
from .relaxed_bernoulli import RelaxedBernoulli
from .relaxed_categorical import RelaxedOneHotCategorical
from .studentT import StudentT
from .transformed_distribution import TransformedDistribution
from .transforms import *  # noqa: F403
from .uniform import Uniform
from .von_mises import VonMises
from .weibull import Weibull
from .wishart import Wishart

# 添加 KL 散度相关信息
_add_kl_info()

# 删除 _add_kl_info 函数的引用，因为已经添加了 KL 散度相关信息
del _add_kl_info

# __all__ 列表包含了该模块中所有公共接口的名称，这些接口可以被其他模块导入
__all__ = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Chi2",
    "ContinuousBernoulli",
    "Dirichlet",
    "Distribution",
    "Exponential",
    "ExponentialFamily",
    "FisherSnedecor",
    "Gamma",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "Independent",
    "InverseGamma",
    "Kumaraswamy",
    "LKJCholesky",
    "Laplace",
    "LogNormal",
    "LogisticNormal",
    "LowRankMultivariateNormal",
    "MixtureSameFamily",
    "Multinomial",
    "MultivariateNormal",
    "NegativeBinomial",
    "Normal",
    "OneHotCategorical",
    "OneHotCategoricalStraightThrough",
    "Pareto",
    "RelaxedBernoulli",
    "RelaxedOneHotCategorical",
    "StudentT",
    "Poisson",
    "Uniform",
    "VonMises",
    "Weibull",
    "Wishart",
    "TransformedDistribution",
    "biject_to",
    "kl_divergence",
    "register_kl",
    "transform_to",
]

# 将 transforms 模块中所有的公共接口名称也添加到 __all__ 列表中
__all__.extend(transforms.__all__)
```