# `.\pytorch\test\distributions\test_distributions.py`

```py
# Owner(s): ["module: distributions"]

"""
Note [Randomized statistical tests]
-----------------------------------

This note describes how to maintain tests in this file as random sources
change. This file contains two types of randomized tests:

1. The easier type of randomized test are tests that should always pass but are
   initialized with random data. If these fail something is wrong, but it's
   fine to use a fixed seed by inheriting from common.TestCase.

2. The trickier tests are statistical tests. These tests explicitly call
   set_rng_seed(n) and are marked "see Note [Randomized statistical tests]".
   These statistical tests have a known positive failure rate
   (we set failure_rate=1e-3 by default). We need to balance strength of these
   tests with annoyance of false alarms. One way that works is to specifically
   set seeds in each of the randomized tests. When a random generator
   occasionally changes (as in #4312 vectorizing the Box-Muller sampler), some
   of these statistical tests may (rarely) fail. If one fails in this case,
   it's fine to increment the seed of the failing test (but you shouldn't need
   to increment it more than once; otherwise something is probably actually
   wrong).

3. `test_geometric_sample`, `test_binomial_sample` and `test_poisson_sample`
   are validated against `scipy.stats.` which are not guaranteed to be identical
   across different versions of scipy (namely, they yield invalid results in 1.7+)
"""

import math  # 导入数学库
import numbers  # 导入数字类型库
import unittest  # 导入单元测试框架
from collections import namedtuple  # 导入命名元组
from itertools import product  # 导入迭代器工具
from random import shuffle  # 导入随机打乱函数

from packaging import version  # 导入版本管理工具

import torch  # 导入 PyTorch 库
import torch.autograd.forward_ad as fwAD  # 导入自动求导前向模块

from torch import inf, nan  # 导入无穷大和NaN常量
from torch.autograd import grad  # 导入梯度计算函数
from torch.autograd.functional import jacobian  # 导入雅可比矩阵计算函数
from torch.distributions import (  # 导入分布相关模块
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Cauchy,
    Chi2,
    constraints,
    ContinuousBernoulli,
    Dirichlet,
    Distribution,
    Exponential,
    ExponentialFamily,
    FisherSnedecor,
    Gamma,
    Geometric,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    Independent,
    InverseGamma,
    kl_divergence,
    Kumaraswamy,
    Laplace,
    LKJCholesky,
    LogisticNormal,
    LogNormal,
    LowRankMultivariateNormal,
    MixtureSameFamily,
    Multinomial,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    OneHotCategorical,
    OneHotCategoricalStraightThrough,
    Pareto,
    Poisson,
    RelaxedBernoulli,
    RelaxedOneHotCategorical,
    StudentT,
    TransformedDistribution,
    Uniform,
    VonMises,
    Weibull,
    Wishart,
)
from torch.distributions.constraint_registry import transform_to  # 导入约束转换函数
from torch.distributions.constraints import Constraint, is_dependent  # 导入约束相关函数
from torch.distributions.dirichlet import _Dirichlet_backward  # 导入狄利克雷分布相关函数
from torch.distributions.kl import _kl_expfamily_expfamily  # 导入 KL 散度计算函数
from torch.distributions.transforms import (  # 导入变换相关模块
    AffineTransform,
    CatTransform,
    ExpTransform,
    # 导入identity_transform和StackTransform模块，用于数据转换和堆叠处理
    identity_transform,
    StackTransform,
# 从 torch.distributions.utils 模块导入以下函数和类
# lazy_property: 用于惰性加载属性的装饰器
# probs_to_logits: 将概率转换为 logits（对数几率）的函数
# tril_matrix_to_vec: 将下三角矩阵转换为向量的函数
# vec_to_tril_matrix: 将向量转换为下三角矩阵的函数
from torch.distributions.utils import (
    lazy_property,
    probs_to_logits,
    tril_matrix_to_vec,
    vec_to_tril_matrix,
)

# 从 torch.nn.functional 模块导入 softmax 函数
from torch.nn.functional import softmax

# 从 torch.testing._internal.common_cuda 模块导入 TEST_CUDA 常量
from torch.testing._internal.common_cuda import TEST_CUDA

# 从 torch.testing._internal.common_utils 模块导入以下函数和类
# gradcheck: 用于梯度检查的函数
# load_tests: 用于加载测试的函数，用于 sharding on sandcastle 的自动过滤测试
# run_tests: 运行测试的函数
# set_default_dtype: 设置默认数据类型的函数
# set_rng_seed: 设置随机数生成种子的函数
# skipIfTorchDynamo: 如果不适用于 TorchDynamo 的测试时跳过的装饰器
# TestCase: 测试用例的基类
from torch.testing._internal.common_utils import (
    gradcheck,
    load_tests,
    run_tests,
    set_default_dtype,
    set_rng_seed,
    skipIfTorchDynamo,
    TestCase,
)

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
# 将 load_tests 重新赋值为 torch.testing._internal.common_utils 中的 load_tests 函数，用于在 sandcastle 上进行测试分片时自动过滤测试，这一行代码用于抑制 flake 警告
load_tests = load_tests

# 设置一个标志 TEST_NUMPY 为 True
TEST_NUMPY = True

# 尝试导入 numpy 和 scipy 的相关模块
try:
    import numpy as np
    import scipy.special
    import scipy.stats
# 如果 ImportError，则将 TEST_NUMPY 设为 False
except ImportError:
    TEST_NUMPY = False


def pairwise(Dist, *params):
    """
    创建一对分布 `Dist`，用于测试每个参数与其他参数的组合。
    """
    # 使用每个参数创建张量，并复制 len(p) 次，组成 params1 列表
    params1 = [torch.tensor([p] * len(p)) for p in params]
    # 将 params1 中的张量转置，组成 params2 列表
    params2 = [p.transpose(0, 1) for p in params1]
    # 返回分布 Dist 的两个实例，分别用 params1 和 params2 初始化
    return Dist(*params1), Dist(*params2)


def is_all_nan(tensor):
    """
    检查张量的所有元素是否都为 NaN。
    """
    # 检查张量中所有元素是否都不等于自身（即是否有 NaN）
    return (tensor != tensor).all()


# 创建一个命名元组 Example，包含 Dist 和 params 两个字段
Example = namedtuple("Example", ["Dist", "params"])


# 注册所有分布以进行通用测试。
def _get_examples():
    # 略，此处应有具体实现代码
    pass


def _get_bad_examples():
    # 略，此处应有具体实现代码
    pass


# 定义一个测试类 DistributionsTestCase，继承自 TestCase
class DistributionsTestCase(TestCase):
    def setUp(self):
        """设置测试前的准备工作，假定验证标志被设置为 True。"""
        # 设置 torch.distributions.Distribution 的默认验证参数为 True
        torch.distributions.Distribution.set_default_validate_args(True)
        # 调用父类 TestCase 的 setUp 方法
        super().setUp()


# 使用装饰器 skipIfTorchDynamo，在不适合 TorchDynamo 的情况下跳过测试
@skipIfTorchDynamo("Not a TorchDynamo suitable test")
# 定义测试类 TestDistributions，继承自 DistributionsTestCase
class TestDistributions(DistributionsTestCase):
    # 设置 CUDA 内存泄漏检查标志为 True
    _do_cuda_memory_leak_check = True
    # 设置使用非默认流的 CUDA 标志为 True
    _do_cuda_non_default_stream = True

    def _gradcheck_log_prob(self, dist_ctor, ctor_params):
        """
        对 log_prob 函数进行梯度检查。
        """
        # 使用给定参数 ctor_params 构建分布 dist_ctor
        distribution = dist_ctor(*ctor_params)
        # 生成一个样本 s
        s = distribution.sample()
        # 如果分布不是离散分布，则将样本 s 取消梯度
        if not distribution.support.is_discrete:
            s = s.detach().requires_grad_()

        # 期望的形状为分布的批量形状加上事件形状
        expected_shape = distribution.batch_shape + distribution.event_shape
        # 断言样本 s 的形状与期望形状一致
        self.assertEqual(s.size(), expected_shape)

        # 定义应用函数 apply_fn，计算分布对样本 s 的 log_prob
        def apply_fn(s, *params):
            return dist_ctor(*params).log_prob(s)

        # 对 apply_fn 进行梯度检查，传入样本 s 和参数 ctor_params
        gradcheck(apply_fn, (s,) + tuple(ctor_params), raise_exception=True)

    def _check_forward_ad(self, fn):
        """
        对函数 fn 进行前向自动微分检查。
        """
        # 使用 fwAD.dual_level() 进行双层级别设置
        with fwAD.dual_level():
            # 创建张量 x 和 t，使用 fwAD.make_dual 创建双向梯度张量 dual
            x = torch.tensor(1.0)
            t = torch.tensor(1.0)
            dual = fwAD.make_dual(x, t)
            # 对函数 fn 应用双向梯度张量 dual
            dual_out = fn(dual)
            # 断言 unpack_dual(dual_out) 的切线（tangent）中非零元素的数量为 0
            self.assertEqual(
                torch.count_nonzero(fwAD.unpack_dual(dual_out).tangent).item(), 0
            )
    # 检查 log_prob 方法是否与参考函数匹配
    def _check_log_prob(self, dist, asset_fn):
        # 从分布中抽取样本
        s = dist.sample()
        # 计算样本的对数概率
        log_probs = dist.log_prob(s)
        # 将对数概率展平为一维数组
        log_probs_data_flat = log_probs.view(-1)
        # 将样本数据展平为一维数组
        s_data_flat = s.view(len(log_probs_data_flat), -1)
        # 遍历样本数据和对应的对数概率数据
        for i, (val, log_prob) in enumerate(zip(s_data_flat, log_probs_data_flat)):
            # 对每对样本和对数概率调用给定的 asset_fn 函数
            asset_fn(i, val.squeeze(), log_prob)

    # 检查 .sample() 方法是否与参考函数匹配
    def _check_sampler_sampler(
        self,
        torch_dist,
        ref_dist,
        message,
        multivariate=False,
        circular=False,
        num_samples=10000,
        failure_rate=1e-3,
    ):
        # 从 torch 分布中抽取样本并将其转换为 NumPy 数组
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = torch_samples.cpu().numpy()
        # 从参考分布中生成样本并将其转换为 NumPy 数组
        ref_samples = ref_dist.rvs(num_samples).astype(np.float64)
        # 如果是多变量分布，将样本投影到随机轴上
        if multivariate:
            # 生成一个随机轴
            axis = np.random.normal(size=(1,) + torch_samples.shape[1:])
            axis /= np.linalg.norm(axis)
            # 投影并求和以减少维度
            torch_samples = (axis * torch_samples).reshape(num_samples, -1).sum(-1)
            ref_samples = (axis * ref_samples).reshape(num_samples, -1).sum(-1)
        # 将 torch_samples 和 ref_samples 组合成样本对列表
        samples = [(x, +1) for x in torch_samples] + [(x, -1) for x in ref_samples]
        # 如果是循环分布，将样本映射到圆形分布
        if circular:
            samples = [(np.cos(x), v) for (x, v) in samples]
        # 随机打乱样本顺序，防止稳定排序导致不均匀的分箱
        shuffle(samples)
        # 按照第一个元素排序（样本值）
        samples.sort(key=lambda x: x[0])
        # 提取样本值，丢弃排序后的样本标签
        samples = np.array(samples)[:, 1]

        # 将样本分成大约均值为零、方差为单位方差的分箱
        num_bins = 10
        samples_per_bin = len(samples) // num_bins
        bins = samples.reshape((num_bins, samples_per_bin)).mean(axis=1)
        stddev = samples_per_bin**-0.5
        # 计算阈值以检测偏差
        threshold = stddev * scipy.special.erfinv(1 - 2 * failure_rate / num_bins)
        # 构建包含有关偏差的消息
        message = f"{message}.sample() is biased:\n{bins}"
        # 对每个分箱中的偏差进行断言检查
        for bias in bins:
            self.assertLess(-threshold, bias, message)
            self.assertLess(bias, threshold, message)

    # 如果没有安装 NumPy，则跳过此测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def _check_sampler_discrete(
        self, torch_dist, ref_dist, message, num_samples=10000, failure_rate=1e-3
    ):
    ):
        """
        Runs a Chi2-test for the support, but ignores tail instead of combining
        """
        # 从分布中抽取指定数量的样本，并将其压缩为一维
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        # 如果抽样结果的数据类型是 torch.bfloat16，则转换为 float 类型
        torch_samples = (
            torch_samples.float()
            if torch_samples.dtype == torch.bfloat16
            else torch_samples
        )
        # 将抽样结果转移到 CPU 并转换为 NumPy 数组
        torch_samples = torch_samples.cpu().numpy()
        # 计算抽样结果中每个唯一值的出现次数
        unique, counts = np.unique(torch_samples, return_counts=True)
        # 使用参考分布计算概率质量函数（PMF）
        pmf = ref_dist.pmf(unique)
        # 将概率质量函数标准化为总和为 1.0，以便进行卡方检验
        pmf = pmf / pmf.sum()
        # 创建掩码，标记出现次数大于5且期望频率大于5的值
        msk = (counts > 5) & ((pmf * num_samples) > 5)
        # 断言满足条件的概率质量函数的总和大于0.9
        self.assertGreater(
            pmf[msk].sum(),
            0.9,
            "Distribution is too sparse for test; try increasing num_samples",
        )
        # 如果存在未满足条件的值（即掩码有 False 条目），则合并它们到一个余下的桶中
        if not msk.all():
            counts = np.concatenate([counts[msk], np.sum(counts[~msk], keepdims=True)])
            pmf = np.concatenate([pmf[msk], np.sum(pmf[~msk], keepdims=True)])
        # 执行卡方检验，计算卡方值（chisq）和 p 值
        chisq, p = scipy.stats.chisquare(counts, pmf * num_samples)
        # 断言 p 值大于失败率，输出相应消息
        self.assertGreater(p, failure_rate, message)
    # 测试函数，验证分布对象的 rsample 方法对参数的梯度要求
    def test_rsample_requires_grad(self):
        # 遍历所有分布类和参数组合的示例
        for Dist, params in _get_examples():
            # 遍历每个示例的参数
            for i, param in enumerate(params):
                # 检查参数中是否有任何一个需要梯度的张量
                if not any(getattr(p, "requires_grad", False) for p in param.values()):
                    continue  # 若没有需要梯度的参数，跳过当前循环
                # 根据参数创建分布对象
                dist = Dist(**param)
                # 如果分布对象不支持 rsample 方法，则跳过当前循环
                if not dist.has_rsample:
                    continue
                # 调用 rsample 方法生成样本
                sample = dist.rsample()
                # 断言样本是否需要梯度，若不需要则报错
                self.assertTrue(
                    sample.requires_grad,
                    msg=f"{Dist.__name__} example {i + 1}/{len(params)}, .rsample() does not require grad",
                )

    # 测试函数，验证分布对象的 sample 方法与 enumerate_support 方法的返回类型匹配
    def test_enumerate_support_type(self):
        # 遍历所有分布类和参数组合的示例
        for Dist, params in _get_examples():
            # 遍历每个示例的参数
            for i, param in enumerate(params):
                # 根据参数创建分布对象
                dist = Dist(**param)
                try:
                    # 断言分布对象的 sample 方法与 enumerate_support 方法的返回类型是否匹配
                    self.assertTrue(
                        type(dist.sample()) is type(dist.enumerate_support()),
                        msg=(
                            "{} example {}/{}, return type mismatch between "
                            + "sample and enumerate_support."
                        ).format(Dist.__name__, i + 1, len(params)),
                    )
                except NotImplementedError:
                    pass

    # 测试函数，验证 lazy_property 装饰器对梯度计算的影响
    def test_lazy_property_grad(self):
        # 创建一个需要梯度的张量 x
        x = torch.randn(1, requires_grad=True)

        # 定义一个 Dummy 类，使用 lazy_property 装饰器计算属性 y
        class Dummy:
            @lazy_property
            def y(self):
                return x + 1

        # 定义测试函数 test
        def test():
            x.grad = None  # 清空张量 x 的梯度
            Dummy().y.backward()  # 计算属性 y 的梯度
            # 断言张量 x 的梯度是否为全 1 向量
            self.assertEqual(x.grad, torch.ones(1))

        # 调用测试函数 test，在有梯度和无梯度的情况下分别执行
        test()
        with torch.no_grad():
            test()

        # 创建均值和协方差张量，其中协方差张量需要梯度
        mean = torch.randn(2)
        cov = torch.eye(2, requires_grad=True)
        # 创建多元正态分布对象 distn
        distn = MultivariateNormal(mean, cov)
        with torch.no_grad():
            distn.scale_tril  # 获取 scale_tril 属性
        # 对 scale_tril 属性的元素求和并反向传播梯度
        distn.scale_tril.sum().backward()
        # 断言协方差张量的梯度不为空
        self.assertIsNotNone(cov.grad)

    # 测试函数，验证各分布类是否在 _get_examples 函数中有示例
    def test_has_examples(self):
        # 获取包含示例的分布类集合
        distributions_with_examples = {e.Dist for e in _get_examples()}
        # 遍历全局变量中的所有对象
        for Dist in globals().values():
            # 检查对象是否为类且继承自 Distribution，且不是 Distribution 或 ExponentialFamily 自身
            if (
                isinstance(Dist, type)
                and issubclass(Dist, Distribution)
                and Dist is not Distribution
                and Dist is not ExponentialFamily
            ):
                # 断言该分布类在示例集合中存在
                self.assertIn(
                    Dist,
                    distributions_with_examples,
                    f"Please add {Dist.__name__} to the _get_examples list in test_distributions.py",
                )
    # 测试支持属性方法
    def test_support_attributes(self):
        # 对于每一个分布 Dist 和其参数 params 的组合，执行以下操作
        for Dist, params in _get_examples():
            # 对于每一个参数 param
            for param in params:
                # 使用参数 param 创建分布对象 d
                d = Dist(**param)
                # 获取事件维度
                event_dim = len(d.event_shape)
                # 断言分布对象的支持属性中的事件维度与计算得到的相等
                self.assertEqual(d.support.event_dim, event_dim)
                # 尝试断言 Dist 类型的支持属性中的事件维度与计算得到的相等
                try:
                    self.assertEqual(Dist.support.event_dim, event_dim)
                # 如果抛出 NotImplementedError 异常则跳过
                except NotImplementedError:
                    pass
                # 获取是否离散属性
                is_discrete = d.support.is_discrete
                # 尝试断言分布对象的支持属性中的是否离散与计算得到的相等
                try:
                    self.assertEqual(Dist.support.is_discrete, is_discrete)
                # 如果抛出 NotImplementedError 异常则跳过
                except NotImplementedError:
                    pass

    # 测试分布扩展方法
    def test_distribution_expand(self):
        # 定义不同的形状
        shapes = [torch.Size(), torch.Size((2,)), torch.Size((2, 1))]
        # 对于每一个分布 Dist 和其参数 params 的组合，执行以下操作
        for Dist, params in _get_examples():
            # 对于每一个参数 param
            for param in params:
                # 对于每一个形状 shape
                for shape in shapes:
                    # 使用参数 param 创建分布对象 d
                    d = Dist(**param)
                    # 计算扩展后的形状
                    expanded_shape = shape + d.batch_shape
                    # 计算原始形状
                    original_shape = d.batch_shape + d.event_shape
                    # 计算预期的形状
                    expected_shape = shape + original_shape
                    # 执行分布对象的扩展方法
                    expanded = d.expand(batch_shape=list(expanded_shape))
                    # 生成样本
                    sample = expanded.sample()
                    # 获取实际形状
                    actual_shape = expanded.sample().shape
                    # 断言扩展后的对象类型与原始对象类型相等
                    self.assertEqual(expanded.__class__, d.__class__)
                    # 断言原始对象样本的形状与计算得到的原始形状相等
                    self.assertEqual(d.sample().shape, original_shape)
                    # 断言扩展对象的对数概率与原始对象的对数概率相等
                    self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                    # 断言实际形状与预期形状相等
                    self.assertEqual(actual_shape, expected_shape)
                    # 断言扩展对象的批处理形状与计算得到的扩展形状相等
                    self.assertEqual(expanded.batch_shape, expanded_shape)
                    # 尝试断言扩展对象的均值与原始对象的均值扩展形状相等
                    try:
                        self.assertEqual(
                            expanded.mean, d.mean.expand(expanded_shape + d.event_shape)
                        )
                        # 尝试断言扩展对象的方差与原始对象的方差扩展形状相等
                        self.assertEqual(
                            expanded.variance,
                            d.variance.expand(expanded_shape + d.event_shape),
                        )
                    # 如果抛出 NotImplementedError 异常则跳过
                    except NotImplementedError:
                        pass

    # 测试分布子类的扩展方法
    def test_distribution_subclass_expand(self):
        # 定义扩展形状
        expand_by = torch.Size((2,))
        # 对于每一个分布 Dist 和其参数 params 的组合，执行以下操作
        for Dist, params in _get_examples():

            # 创建分布的子类
            class SubClass(Dist):
                pass

            # 对于每一个参数 param
            for param in params:
                # 使用参数 param 创建分布子类对象 d
                d = SubClass(**param)
                # 计算扩展后的形状
                expanded_shape = expand_by + d.batch_shape
                # 计算原始形状
                original_shape = d.batch_shape + d.event_shape
                # 计算预期的形状
                expected_shape = expand_by + original_shape
                # 执行分布子类对象的扩展方法
                expanded = d.expand(batch_shape=expanded_shape)
                # 生成样本
                sample = expanded.sample()
                # 获取实际形状
                actual_shape = expanded.sample().shape
                # 断言扩展后的对象类型与原始对象类型相等
                self.assertEqual(expanded.__class__, d.__class__)
                # 断言原始对象样本的形状与计算得到的原始形状相等
                self.assertEqual(d.sample().shape, original_shape)
                # 断言扩展对象的对数概率与原始对象的对数概率相等
                self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                # 断言实际形状与预期形状相等
                self.assertEqual(actual_shape, expected_shape)
    # 设置默认张量数据类型为双精度浮点型
    @set_default_dtype(torch.double)
    # 测试伯努利分布的样本生成和梯度检查等功能
    def test_bernoulli(self):
        # 定义概率张量 p，并要求其梯度计算
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        # 定义伯努利分布参数 r，并要求其梯度计算
        r = torch.tensor(0.3, requires_grad=True)
        # 定义标量 s
        s = 0.3
        # 测试 Bernoulli 分布生成样本的形状是否符合预期
        self.assertEqual(Bernoulli(p).sample((8,)).size(), (8, 3))
        # 检查生成的 Bernoulli 样本是否不需要梯度
        self.assertFalse(Bernoulli(p).sample().requires_grad)
        # 测试 Bernoulli 分布生成多维样本的形状是否符合预期
        self.assertEqual(Bernoulli(r).sample((8,)).size(), (8,))
        # 测试生成标量 Bernoulli 样本的形状是否符合预期
        self.assertEqual(Bernoulli(r).sample().size(), ())
        # 测试生成二维 Bernoulli 样本的形状是否符合预期
        self.assertEqual(
            Bernoulli(r).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        # 测试生成标量 s 的 Bernoulli 样本的形状是否符合预期
        self.assertEqual(Bernoulli(s).sample().size(), ())
        # 对 Bernoulli 分布的 log_prob 方法进行梯度检查
        self._gradcheck_log_prob(Bernoulli, (p,))
        
        # 定义用于参考的 log_prob 函数，用于检查 Bernoulli 分布的对数概率计算是否正确
        def ref_log_prob(idx, val, log_prob):
            prob = p[idx]
            self.assertEqual(log_prob, math.log(prob if val else 1 - prob))
        
        # 检查 Bernoulli 分布的 log_prob 方法的实现是否正确
        self._check_log_prob(Bernoulli(p), ref_log_prob)
        # 检查对 logits 转换后的 Bernoulli 分布的 log_prob 方法是否正确
        self._check_log_prob(Bernoulli(logits=p.log() - (-p).log1p()), ref_log_prob)
        # 测试 Bernoulli 分布的 rsample 方法是否会抛出 NotImplementedError 异常
        self.assertRaises(NotImplementedError, Bernoulli(r).rsample)

        # 检查熵的计算是否正确
        self.assertEqual(
            Bernoulli(p).entropy(),
            torch.tensor([0.6108, 0.5004, 0.6730]),
            atol=1e-4,
            rtol=0,
        )
        # 测试当概率为零时 Bernoulli 分布的熵是否为零
        self.assertEqual(Bernoulli(torch.tensor([0.0])).entropy(), torch.tensor([0.0]))
        # 测试生成标量 s 的 Bernoulli 分布的熵是否符合预期
        self.assertEqual(
            Bernoulli(s).entropy(), torch.tensor(0.6108), atol=1e-4, rtol=0
        )

        # 检查使用自动微分进行前向传播是否正确
        self._check_forward_ad(torch.bernoulli)
        self._check_forward_ad(lambda x: x.bernoulli_())
        self._check_forward_ad(lambda x: x.bernoulli_(x.clone().detach()))
        self._check_forward_ad(lambda x: x.bernoulli_(x))

    # 测试 Bernoulli 分布的 enumerate_support 方法
    def test_bernoulli_enumerate_support(self):
        # 定义不同示例，每个示例包含概率分布的描述和预期的支持集合
        examples = [
            ({"probs": [0.1]}, [[0], [1]]),
            ({"probs": [0.1, 0.9]}, [[0], [1]]),
            ({"probs": [[0.1, 0.2], [0.3, 0.4]]}, [[[0]], [[1]]]),
        ]
        # 检查 enumerate_support 方法是否正确枚举支持集合
        self._check_enumerate_support(Bernoulli, examples)

    # 测试 Bernoulli 分布在三维张量上的样本生成功能
    def test_bernoulli_3d(self):
        # 定义形状为 (2, 3, 5) 的概率张量 p，并要求其梯度计算
        p = torch.full((2, 3, 5), 0.5).requires_grad_()
        # 测试生成三维 Bernoulli 样本的形状是否符合预期
        self.assertEqual(Bernoulli(p).sample().size(), (2, 3, 5))
        # 测试生成形状为 (2, 5) 的多维 Bernoulli 样本的形状是否符合预期
        self.assertEqual(
            Bernoulli(p).sample(sample_shape=(2, 5)).size(), (2, 5, 2, 3, 5)
        )
        # 测试生成形状为 (2,) 的 Bernoulli 样本的形状是否符合预期
        self.assertEqual(Bernoulli(p).sample((2,)).size(), (2, 2, 3, 5))

    # 设置默认张量数据类型为双精度浮点型
    @set_default_dtype(torch.double)
    # 定义测试函数 test_geometric，测试几何分布相关功能
    def test_geometric(self):
        # 创建一个张量 p，表示几何分布的概率参数，需要计算梯度
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        # 创建一个张量 r，表示几何分布的概率参数，需要计算梯度
        r = torch.tensor(0.3, requires_grad=True)
        # 创建一个标量 s，表示几何分布的概率参数，不需要计算梯度
        s = 0.3

        # 测试样本采样函数 sample 的输出形状是否为 (8, 3)
        self.assertEqual(Geometric(p).sample((8,)).size(), (8, 3))
        # 测试当参数为 1 时，sample 函数的输出是否为 0
        self.assertEqual(Geometric(1).sample(), 0)
        # 测试当参数为 1 时，log_prob 函数输入参数为 1.0 时的输出是否为负无穷 -inf
        self.assertEqual(Geometric(1).log_prob(torch.tensor(1.0)), -inf)
        # 测试当参数为 1 时，log_prob 函数输入参数为 0.0 时的输出是否为 0
        self.assertEqual(Geometric(1).log_prob(torch.tensor(0.0)), 0)
        # 测试 sample 函数是否不需要计算梯度
        self.assertFalse(Geometric(p).sample().requires_grad)
        # 测试样本采样函数 sample 的输出形状是否为 (8,)
        self.assertEqual(Geometric(r).sample((8,)).size(), (8,))
        # 测试样本采样函数 sample 的输出形状是否为 ()
        self.assertEqual(Geometric(r).sample().size(), ())
        # 测试样本采样函数 sample 的输出形状是否为 (3, 2)
        self.assertEqual(Geometric(r).sample((3, 2)).size(), (3, 2))
        # 测试样本采样函数 sample 的输出形状是否为 ()
        self.assertEqual(Geometric(s).sample().size(), ())

        # 使用梯度检查函数 _gradcheck_log_prob 检查 log_prob 函数对参数 p 的梯度计算是否正确
        self._gradcheck_log_prob(Geometric, (p,))
        # 测试当参数为 0 时，是否会引发 ValueError 异常
        self.assertRaises(ValueError, lambda: Geometric(0))
        # 测试当使用 rsample 函数时是否会引发 NotImplementedError 异常
        self.assertRaises(NotImplementedError, Geometric(r).rsample)

        # 使用 _check_forward_ad 函数检查 forward_autograd 函数对参数为 0.2 的几何分布的行为是否正确
        self._check_forward_ad(lambda x: x.geometric_(0.2))

    # 如果未安装 NumPy，跳过测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 设置默认数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    # 测试几何分布的 log_prob 和 entropy 函数
    def test_geometric_log_prob_and_entropy(self):
        # 创建一个张量 p，表示几何分布的概率参数，需要计算梯度
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        # 创建一个标量 s，表示几何分布的概率参数，不需要计算梯度
        s = 0.3

        # 定义参考 log_prob 函数，用于验证 log_prob 函数的正确性
        def ref_log_prob(idx, val, log_prob):
            # 从 p 中分离出概率值，进行概率质量函数的比较
            prob = p[idx].detach()
            self.assertEqual(log_prob, scipy.stats.geom(prob, loc=-1).logpmf(val))

        # 使用 _check_log_prob 函数检查 log_prob 函数的正确性
        self._check_log_prob(Geometric(p), ref_log_prob)
        # 使用 _check_log_prob 函数检查对 logits 转换后的 log_prob 函数的正确性
        self._check_log_prob(Geometric(logits=p.log() - (-p).log1p()), ref_log_prob)

        # 检查 entropy 函数的计算结果是否与 scipy 的几何分布熵值一致，允许的误差为 1e-3
        self.assertEqual(
            Geometric(p).entropy(),
            scipy.stats.geom(p.detach().numpy(), loc=-1).entropy(),
            atol=1e-3,
            rtol=0,
        )
        # 检查 entropy 函数的计算结果是否与 scipy 的几何分布熵值一致，允许的误差为 1e-3
        self.assertEqual(
            float(Geometric(s).entropy()),
            scipy.stats.geom(s, loc=-1).entropy().item(),
            atol=1e-3,
            rtol=0,
        )

    # 如果未安装 NumPy，跳过测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 测试几何分布的样本采样函数
    def test_geometric_sample(self):
        # 设置随机数生成器种子，用于随机统计测试
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        # 遍历不同的概率值，进行离散采样的一致性检查
        for prob in [0.01, 0.18, 0.8]:
            # 使用 _check_sampler_discrete 函数检查离散采样的正确性
            self._check_sampler_discrete(
                Geometric(prob),
                scipy.stats.geom(p=prob, loc=-1),
                f"Geometric(prob={prob})",
            )

    # 设置默认数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    # 测试二项分布相关功能
    def test_binomial(self):
        # 创建一个张量 p，表示二项分布的概率参数，需要计算梯度
        p = torch.arange(0.05, 1, 0.1).requires_grad_()
        # 遍历不同的试验次数，进行 log_prob 函数的梯度检查
        for total_count in [1, 2, 10]:
            # 使用 _gradcheck_log_prob 函数检查 log_prob 函数对参数 p 的梯度计算是否正确
            self._gradcheck_log_prob(lambda p: Binomial(total_count, p), [p])
            # 使用 _gradcheck_log_prob 函数检查 log_prob 函数对 logits 转换后的参数 p 的梯度计算是否正确
            self._gradcheck_log_prob(
                lambda p: Binomial(total_count, None, p.log()), [p]
            )
        # 测试当使用 rsample 函数时是否会引发 NotImplementedError 异常
        self.assertRaises(NotImplementedError, Binomial(10, p).rsample)

    # 设置默认数据类型为半精度浮点数，测试二项分布相关功能
    test_binomial_half = set_default_dtype(torch.float16)(test_binomial)
    # 设置默认数据类型为 bfloat16，测试二项分布相关功能
    test_binomial_bfloat16 = set_default_dtype(torch.bfloat16)(test_binomial)

    # 如果未安装 NumPy，跳过测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_binomial_sample(self):
        # 设置随机数生成种子为0，参见注释[随机化统计测试]
        set_rng_seed(0)
        # 针对不同的概率和次数进行测试
        for prob in [0.01, 0.1, 0.5, 0.8, 0.9]:
            for count in [2, 10, 100, 500]:
                # 调用 _check_sampler_discrete 方法进行离散采样器的检查
                self._check_sampler_discrete(
                    Binomial(total_count=count, probs=prob),
                    scipy.stats.binom(count, prob),
                    f"Binomial(total_count={count}, probs={prob})",
                )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype(torch.double)
    def test_binomial_log_prob_and_entropy(self):
        # 使用 torch.arange 定义概率范围
        probs = torch.arange(0.05, 1, 0.1)
        # 对每个总数进行循环
        for total_count in [1, 2, 10]:

            def ref_log_prob(idx, x, log_prob):
                # 获取给定索引的概率并计算预期值
                p = probs.view(-1)[idx].item()
                expected = scipy.stats.binom(total_count, p).logpmf(x)
                # 使用断言检查对数概率是否与预期值相等
                self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

            # 调用 _check_log_prob 方法检查对数概率
            self._check_log_prob(Binomial(total_count, probs), ref_log_prob)
            # 将概率转换为 logits 并调用 _check_log_prob 方法检查对数概率
            logits = probs_to_logits(probs, is_binary=True)
            self._check_log_prob(Binomial(total_count, logits=logits), ref_log_prob)

            # 创建 Binomial 对象并验证其熵是否与预期值相等
            bin = Binomial(total_count, logits=logits)
            self.assertEqual(
                bin.entropy(),
                scipy.stats.binom(
                    total_count, bin.probs.detach().numpy(), loc=-1
                ).entropy(),
                atol=1e-3,
                rtol=0,
            )

    def test_binomial_stable(self):
        # 定义 logits、total_count 和 x
        logits = torch.tensor([-100.0, 100.0], dtype=torch.float)
        total_count = 1.0
        x = torch.tensor([0.0, 0.0], dtype=torch.float)
        # 计算 Binomial 分布的对数概率并确保其是有限的
        log_prob = Binomial(total_count, logits=logits).log_prob(x)
        self.assertTrue(torch.isfinite(log_prob).all())

        # 确保在 logits=0 且值=0 时的梯度为 0.5
        x = torch.tensor(0.0, requires_grad=True)
        y = Binomial(total_count, logits=x).log_prob(torch.tensor(0.0))
        self.assertEqual(grad(y, x)[0], torch.tensor(-0.5))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype(torch.double)
    def test_binomial_log_prob_vectorized_count(self):
        # 定义概率和总数的组合
        probs = torch.tensor([0.2, 0.7, 0.9])
        for total_count, sample in [
            (torch.tensor([10]), torch.tensor([7.0, 3.0, 9.0])),
            (torch.tensor([1, 2, 10]), torch.tensor([0.0, 1.0, 9.0])),
        ]:
            # 计算 Binomial 分布的对数概率并验证是否与预期值相等
            log_prob = Binomial(total_count, probs).log_prob(sample)
            expected = scipy.stats.binom(
                total_count.cpu().numpy(), probs.cpu().numpy()
            ).logpmf(sample)
            self.assertEqual(log_prob, expected, atol=1e-4, rtol=0)
    # 定义一个测试方法，用于测试二项分布的支持枚举功能
    def test_binomial_enumerate_support(self):
        # 定义测试示例，每个示例包括输入参数和预期输出
        examples = [
            ({"probs": [0.1], "total_count": 2}, [[0], [1], [2]]),
            ({"probs": [0.1, 0.9], "total_count": 2}, [[0], [1], [2]]),
            (
                {"probs": [[0.1, 0.2], [0.3, 0.4]], "total_count": 3},
                [[[0]], [[1]], [[2]], [[3]]],
            ),
        ]
        # 调用内部方法检查枚举支持函数的输出是否符合预期
        self._check_enumerate_support(Binomial, examples)

    # 设置默认数据类型为双精度浮点数的装饰器，用于测试极端值情况下的二项分布
    @set_default_dtype(torch.double)
    def test_binomial_extreme_vals(self):
        # 定义总数为100的测试用例，测试当成功概率为0或1时的二项分布
        total_count = 100
        bin0 = Binomial(total_count, 0)
        # 断言样本为0
        self.assertEqual(bin0.sample(), 0)
        # 断言当样本为0时的对数概率
        self.assertEqual(bin0.log_prob(torch.tensor([0.0]))[0], 0, atol=1e-3, rtol=0)
        # 断言当样本为1时的指数化的对数概率为0
        self.assertEqual(float(bin0.log_prob(torch.tensor([1.0])).exp()), 0)
        bin1 = Binomial(total_count, 1)
        # 断言样本为总数
        self.assertEqual(bin1.sample(), total_count)
        # 断言当样本为总数时的对数概率
        self.assertEqual(
            bin1.log_prob(torch.tensor([float(total_count)]))[0], 0, atol=1e-3, rtol=0
        )
        # 断言当样本为总数减1时的指数化的对数概率为0
        self.assertEqual(
            float(bin1.log_prob(torch.tensor([float(total_count - 1)])).exp()), 0
        )
        # 创建一个全零张量作为输入，测试零计数的二项分布
        zero_counts = torch.zeros(torch.Size((2, 2)))
        bin2 = Binomial(zero_counts, 1)
        # 断言样本为全零张量
        self.assertEqual(bin2.sample(), zero_counts)
        # 断言零计数输入时的对数概率为全零张量
        self.assertEqual(bin2.log_prob(zero_counts), zero_counts)

    # 设置默认数据类型为双精度浮点数的装饰器，测试向量化计数的二项分布
    @set_default_dtype(torch.double)
    def test_binomial_vectorized_count(self):
        # 设置随机数种子以进行随机统计测试
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        # 定义总数张量，测试向量化计数的二项分布
        total_count = torch.tensor([[4, 7], [3, 8]], dtype=torch.float64)
        bin0 = Binomial(total_count, torch.tensor(1.0))
        # 断言样本为总数张量
        self.assertEqual(bin0.sample(), total_count)
        bin1 = Binomial(total_count, torch.tensor(0.5))
        # 生成大量样本并测试其分布特性
        samples = bin1.sample(torch.Size((100000,)))
        # 断言所有样本小于等于总数张量的对应元素
        self.assertTrue((samples <= total_count.type_as(samples)).all())
        # 断言样本的均值等于二项分布的均值，允许误差范围在0.02以内
        self.assertEqual(samples.mean(dim=0), bin1.mean, atol=0.02, rtol=0)
        # 断言样本的方差等于二项分布的方差，允许误差范围在0.02以内
        self.assertEqual(samples.var(dim=0), bin1.variance, atol=0.02, rtol=0)

    # 测试负二项分布的方法
    @set_default_dtype(torch.double)
    def test_negative_binomial(self):
        # 创建一个梯度需求的概率张量
        p = torch.arange(0.05, 1, 0.1).requires_grad_()
        # 对于每个指定的总数，检查负二项分布的对数概率的梯度
        for total_count in [1, 2, 10]:
            self._gradcheck_log_prob(lambda p: NegativeBinomial(total_count, p), [p])
            # 对数概率通过对数概率值的方式检查负二项分布
            self._gradcheck_log_prob(
                lambda p: NegativeBinomial(total_count, None, p.log()), [p]
            )
        # 断言抛出未实现错误，因为负二项分布的抽样方法未实现
        self.assertRaises(NotImplementedError, NegativeBinomial(10, p).rsample)
        # 断言抛出未实现错误，因为负二项分布的熵方法未实现
        self.assertRaises(NotImplementedError, NegativeBinomial(10, p).entropy)

    # 如果未找到 NumPy，则跳过以下测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义测试函数 test_negative_binomial_log_prob，测试负二项分布的对数概率计算
    def test_negative_binomial_log_prob(self):
        # 创建概率序列，从0.05到0.95，步长为0.1
        probs = torch.arange(0.05, 1, 0.1)
        # 对于不同的总数目，执行以下操作
        for total_count in [1, 2, 10]:

            # 定义参考对数概率计算函数 ref_log_prob，用于比较计算结果与预期值
            def ref_log_prob(idx, x, log_prob):
                # 获取对应索引处的概率值
                p = probs.view(-1)[idx].item()
                # 计算预期的对数概率值，使用 scipy 库中负二项分布的对数概率密度函数
                expected = scipy.stats.nbinom(total_count, 1 - p).logpmf(x)
                # 断言计算得到的对数概率与预期值的接近程度
                self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

            # 检查负二项分布的对数概率计算是否正确
            self._check_log_prob(NegativeBinomial(total_count, probs), ref_log_prob)
            # 将概率转换为 logits，并检查负二项分布的对数概率计算是否正确
            logits = probs_to_logits(probs, is_binary=True)
            self._check_log_prob(
                NegativeBinomial(total_count, logits=logits), ref_log_prob
            )

    # 根据测试条件判断是否跳过该单元测试（如果没有安装 NumPy）
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 设置默认数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    # 测试负二项分布在向量化总数情况下的对数概率计算
    def test_negative_binomial_log_prob_vectorized_count(self):
        # 指定概率序列
        probs = torch.tensor([0.2, 0.7, 0.9])
        # 对于不同的总数目和样本，执行以下操作
        for total_count, sample in [
            (torch.tensor([10]), torch.tensor([7.0, 3.0, 9.0])),
            (torch.tensor([1, 2, 10]), torch.tensor([0.0, 1.0, 9.0])),
        ]:
            # 计算负二项分布给定样本的对数概率
            log_prob = NegativeBinomial(total_count, probs).log_prob(sample)
            # 计算预期的对数概率值，使用 scipy 库中负二项分布的对数概率密度函数
            expected = scipy.stats.nbinom(
                total_count.cpu().numpy(), 1 - probs.cpu().numpy()
            ).logpmf(sample)
            # 断言计算得到的对数概率与预期值的接近程度
            self.assertEqual(log_prob, expected, atol=1e-4, rtol=0)

    # 根据测试条件判断是否跳过该单元测试（如果没有安装 CUDA）
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    # 测试排除零值的二项分布（使用 CUDA）
    def test_zero_excluded_binomial(self):
        # 使用二项分布生成大量样本，确保所有值都大于等于零
        vals = Binomial(
            total_count=torch.tensor(1.0).cuda(), probs=torch.tensor(0.9).cuda()
        ).sample(torch.Size((100000000,)))
        self.assertTrue((vals >= 0).all())
        # 使用二项分布生成大量样本，确保所有值都小于2
        vals = Binomial(
            total_count=torch.tensor(1.0).cuda(), probs=torch.tensor(0.1).cuda()
        ).sample(torch.Size((100000000,)))
        self.assertTrue((vals < 2).all())
        # 使用二项分布生成较少样本，验证是否接近一半为零，一半为一
        vals = Binomial(
            total_count=torch.tensor(1.0).cuda(), probs=torch.tensor(0.5).cuda()
        ).sample(torch.Size((10000,)))
        # 断言生成的样本中零值数量大于4000
        assert (vals == 0.0).sum() > 4000
        # 断言生成的样本中一值数量大于4000
        assert (vals == 1.0).sum() > 4000

    # 设置默认数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    # 测试一维多项分布
    def test_multinomial_1d(self):
        # 指定总数目
        total_count = 10
        # 指定概率分布，要求计算梯度
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        # 断言多项分布生成样本的形状为 (3,)
        self.assertEqual(Multinomial(total_count, p).sample().size(), (3,))
        # 断言多项分布生成样本的形状为 (2, 2, 3)
        self.assertEqual(Multinomial(total_count, p).sample((2, 2)).size(), (2, 2, 3))
        # 断言多项分布生成样本的形状为 (1, 3)
        self.assertEqual(Multinomial(total_count, p).sample((1,)).size(), (1, 3))
        # 梯度检查多项分布的对数概率计算
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        # 梯度检查多项分布的对数概率计算（使用对数概率）
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])
        # 断言多项分布不支持随机采样（预期抛出 NotImplementedError 异常）
        self.assertRaises(NotImplementedError, Multinomial(10, p).rsample)

    # 根据测试条件判断是否跳过该单元测试（如果没有安装 NumPy）
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 设置默认数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    # 测试单维多项分布的对数概率和熵计算函数
    def test_multinomial_1d_log_prob_and_entropy(self):
        total_count = 10
        # 定义概率分布的概率向量，并标记其需要梯度
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        # 创建一个多项分布对象，给定总数和概率向量
        dist = Multinomial(total_count, probs=p)
        # 从多项分布中抽样一个样本
        x = dist.sample()
        # 计算抽样结果的对数概率
        log_prob = dist.log_prob(x)
        # 使用 SciPy 计算多项分布的对数概率的期望
        expected = torch.tensor(
            scipy.stats.multinomial.logpmf(
                x.numpy(), n=total_count, p=dist.probs.detach().numpy()
            )
        )
        # 断言计算的对数概率与期望值相等
        self.assertEqual(log_prob, expected)

        # 使用 logits 形式的概率向量创建另一个多项分布对象
        dist = Multinomial(total_count, logits=p.log())
        # 从新的多项分布中抽样一个样本
        x = dist.sample()
        # 计算抽样结果的对数概率
        log_prob = dist.log_prob(x)
        # 使用 SciPy 计算多项分布的对数概率的期望
        expected = torch.tensor(
            scipy.stats.multinomial.logpmf(
                x.numpy(), n=total_count, p=dist.probs.detach().numpy()
            )
        )
        # 断言计算的对数概率与期望值相等
        self.assertEqual(log_prob, expected)

        # 使用 SciPy 计算多项分布的熵的期望值
        expected = scipy.stats.multinomial.entropy(
            total_count, dist.probs.detach().numpy()
        )
        # 断言计算的熵与期望值在给定的误差范围内相等
        self.assertEqual(dist.entropy(), expected, atol=1e-3, rtol=0)

    # 为测试双维多项分布设置默认的 Torch 数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    def test_multinomial_2d(self):
        total_count = 10
        # 定义一个二维概率矩阵
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        # 定义另一个二维概率矩阵
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        # 根据给定的概率矩阵创建张量，并标记其需要梯度
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        # 断言从多项分布中抽样得到的结果的形状是否正确
        self.assertEqual(Multinomial(total_count, p).sample().size(), (2, 3))
        # 断言从多项分布中抽样得到的结果的形状是否正确，给定了样本形状参数
        self.assertEqual(
            Multinomial(total_count, p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3)
        )
        # 断言从多项分布中抽样得到的结果的形状是否正确，给定了样本数参数
        self.assertEqual(Multinomial(total_count, p).sample((6,)).size(), (6, 2, 3))
        # 设置随机数生成器的种子为 0
        set_rng_seed(0)
        # 对 log_prob 函数进行梯度检查
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        # 对 log_prob 函数进行梯度检查，使用 logits 形式的概率向量
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])

        # 对于极端概率值的样本抽样检查
        self.assertEqual(
            Multinomial(total_count, s).sample(),
            torch.tensor([[total_count, 0], [0, total_count]], dtype=torch.float64),
        )

    # 设置默认的 Torch 数据类型为双精度浮点数，用于测试单维分类分布
    @set_default_dtype(torch.double)
    def test_categorical_1d(self):
        # 定义概率分布的概率向量，并标记其需要梯度
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        # 断言分类分布的均值是否全为 NaN
        self.assertTrue(is_all_nan(Categorical(p).mean))
        # 断言分类分布的方差是否全为 NaN
        self.assertTrue(is_all_nan(Categorical(p).variance))
        # 断言从分类分布中抽样得到的结果的形状是否正确
        self.assertEqual(Categorical(p).sample().size(), ())
        # 断言从分类分布中抽样得到的结果是否不需要梯度
        self.assertFalse(Categorical(p).sample().requires_grad)
        # 断言从分类分布中抽样得到的结果的形状是否正确，给定了样本形状参数
        self.assertEqual(Categorical(p).sample((2, 2)).size(), (2, 2))
        # 断言从分类分布中抽样得到的结果的形状是否正确，给定了样本数参数
        self.assertEqual(Categorical(p).sample((1,)).size(), (1,))
        # 对 log_prob 函数进行梯度检查
        self._gradcheck_log_prob(Categorical, (p,))
        # 断言在尝试使用 rsample 函数时是否会抛出 NotImplementedError 异常
        self.assertRaises(NotImplementedError, Categorical(p).rsample)
    # 定义测试方法，测试二维分类分布的相关功能
    def test_categorical_2d(self):
        # 定义概率数组
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        # 创建张量并标记需要计算梯度
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        
        # 断言均值的大小为 (2,)
        self.assertEqual(Categorical(p).mean.size(), (2,))
        # 断言方差的大小为 (2,)
        self.assertEqual(Categorical(p).variance.size(), (2,))
        # 断言均值全部为 NaN
        self.assertTrue(is_all_nan(Categorical(p).mean))
        # 断言方差全部为 NaN
        self.assertTrue(is_all_nan(Categorical(p).variance))
        # 断言从分类分布中抽样的大小为 (2,)
        self.assertEqual(Categorical(p).sample().size(), (2,))
        # 断言从分类分布中抽样形状为 (3, 4) 的大小为 (3, 4, 2)
        self.assertEqual(Categorical(p).sample(sample_shape=(3, 4)).size(), (3, 4, 2))
        # 断言从分类分布中抽样形状为 (6,) 的大小为 (6, 2)
        self.assertEqual(Categorical(p).sample((6,)).size(), (6, 2))
        # 对 log_prob 方法进行梯度检查
        self._gradcheck_log_prob(Categorical, (p,))
        
        # 对概率值极端情况下的抽样进行检查
        set_rng_seed(0)
        self.assertEqual(
            Categorical(s).sample(sample_shape=(2,)), torch.tensor([[0, 1], [0, 1]])
        )
        
        # 定义参考的 log_prob 函数
        def ref_log_prob(idx, val, log_prob):
            sample_prob = p[idx][val] / p[idx].sum()
            self.assertEqual(log_prob, math.log(sample_prob))
        
        # 对 log_prob 方法进行检查
        self._check_log_prob(Categorical(p), ref_log_prob)
        self._check_log_prob(Categorical(logits=p.log()), ref_log_prob)
        
        # 检查熵的计算结果
        self.assertEqual(
            Categorical(p).entropy(), torch.tensor([1.0114, 1.0297]), atol=1e-4, rtol=0
        )
        self.assertEqual(Categorical(s).entropy(), torch.tensor([0.0, 0.0]))
        # issue gh-40553
        # 检查概率值中包含无穷值时熵的计算结果
        logits = p.log()
        logits[1, 1] = logits[0, 2] = float("-inf")
        e = Categorical(logits=logits).entropy()
        self.assertEqual(e, torch.tensor([0.6365, 0.5983]), atol=1e-4, rtol=0)

    # 定义测试方法，测试枚举支持功能
    def test_categorical_enumerate_support(self):
        # 定义测试示例
        examples = [
            ({"probs": [0.1, 0.2, 0.7]}, [0, 1, 2]),
            ({"probs": [[0.1, 0.9], [0.3, 0.7]]}, [[0], [1]]),
        ]
        # 调用 _check_enumerate_support 方法进行检查
        self._check_enumerate_support(Categorical, examples)

    # 设置默认数据类型为双精度浮点数，定义测试方法，测试一维独热分类分布
    @set_default_dtype(torch.double)
    def test_one_hot_categorical_1d(self):
        # 创建张量并标记需要计算梯度
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        # 断言从一维独热分类分布中抽样的大小为 (3,)
        self.assertEqual(OneHotCategorical(p).sample().size(), (3,))
        # 断言从一维独热分类分布中抽样不需要计算梯度
        self.assertFalse(OneHotCategorical(p).sample().requires_grad)
        # 断言从一维独热分类分布中抽样形状为 (2, 2) 的大小为 (2, 2, 3)
        self.assertEqual(OneHotCategorical(p).sample((2, 2)).size(), (2, 2, 3))
        # 断言从一维独热分类分布中抽样形状为 (1,) 的大小为 (1, 3)
        self.assertEqual(OneHotCategorical(p).sample((1,)).size(), (1, 3))
        # 对 log_prob 方法进行梯度检查
        self._gradcheck_log_prob(OneHotCategorical, (p,))
        # 断言调用 rsample 方法会引发 NotImplementedError 异常
        self.assertRaises(NotImplementedError, OneHotCategorical(p).rsample)
    def test_one_hot_categorical_2d(self):
        # 定义一个二维概率矩阵
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        # 另一个二维概率矩阵
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        # 创建张量，设置 requires_grad=True 以支持梯度计算
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        # 断言采样结果的形状为 (2, 3)
        self.assertEqual(OneHotCategorical(p).sample().size(), (2, 3))
        # 断言采样形状为 (3, 4) 的样本的结果形状为 (3, 4, 2, 3)
        self.assertEqual(
            OneHotCategorical(p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3)
        )
        # 断言采样形状为 (6,) 的样本的结果形状为 (6, 2, 3)
        self.assertEqual(OneHotCategorical(p).sample((6,)).size(), (6, 2, 3))
        # 梯度检查，验证对数概率计算的正确性
        self._gradcheck_log_prob(OneHotCategorical, (p,))

        # 创建 OneHotCategorical 分布对象
        dist = OneHotCategorical(p)
        # 从分布中采样
        x = dist.sample()
        # 断言 OneHotCategorical 分布的对数概率等于对应的 Categorical 分布的对数概率
        self.assertEqual(dist.log_prob(x), Categorical(p).log_prob(x.max(-1)[1]))

    def test_one_hot_categorical_enumerate_support(self):
        # 不同的例子和期望输出
        examples = [
            ({"probs": [0.1, 0.2, 0.7]}, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ({"probs": [[0.1, 0.9], [0.3, 0.7]]}, [[[1, 0]], [[0, 1]]]),
        ]
        # 检查 enumerate_support 方法的输出是否符合预期
        self._check_enumerate_support(OneHotCategorical, examples)

    def test_poisson_forward_ad(self):
        # 检查 torch.poisson 的自动求导功能
        self._check_forward_ad(torch.poisson)

    def test_poisson_shape(self):
        # 创建具有随机速率的 Poisson 分布对象，设置 requires_grad=True 以支持梯度计算
        rate = torch.randn(2, 3).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        # 断言从 Poisson 分布中采样结果的形状为 (2, 3)
        self.assertEqual(Poisson(rate).sample().size(), (2, 3))
        # 断言从 Poisson 分布中采样形状为 (7,) 的样本结果的形状为 (7, 2, 3)
        self.assertEqual(Poisson(rate).sample((7,)).size(), (7, 2, 3))
        # 断言从 Poisson 分布中采样结果的形状为 (1,)，使用形状 (1,) 进行采样的结果的形状为 (1, 1)
        self.assertEqual(Poisson(rate_1d).sample().size(), (1,))
        self.assertEqual(Poisson(rate_1d).sample((1,)).size(), (1, 1))
        # 断言从 Poisson 分布中采样结果的形状为 (2,)
        self.assertEqual(Poisson(2.0).sample((2,)).size(), (2,))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @set_default_dtype(torch.double)
    def test_poisson_log_prob(self):
        # 创建具有随机速率的 Poisson 分布对象，设置 requires_grad=True 以支持梯度计算
        rate = torch.randn(2, 3).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        rate_zero = torch.zeros([], requires_grad=True)

        # 参考实现的对数概率函数
        def ref_log_prob(ref_rate, idx, x, log_prob):
            l = ref_rate.view(-1)[idx].detach()
            expected = scipy.stats.poisson.logpmf(x, l)
            # 断言计算得到的对数概率与预期值在一定容差范围内相等
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 设置随机数种子以确保可重复性
        set_rng_seed(0)
        # 检查 Poisson 分布的对数概率计算是否正确
        self._check_log_prob(Poisson(rate), lambda *args: ref_log_prob(rate, *args))
        self._check_log_prob(
            Poisson(rate_zero), lambda *args: ref_log_prob(rate_zero, *args)
        )
        # 梯度检查
        self._gradcheck_log_prob(Poisson, (rate,))
        self._gradcheck_log_prob(Poisson, (rate_1d,))

        # 由于零速率处进入了禁止的参数空间，无法自动检查梯度，因此与理论结果进行比较
        dist = Poisson(rate_zero)
        dist.log_prob(torch.ones_like(rate_zero)).backward()
        # 断言在零速率处的梯度是无穷大
        self.assertEqual(rate_zero.grad, torch.inf)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    # 定义用于测试 Poisson 分布采样的方法
    def test_poisson_sample(self):
        # 设置随机数生成器种子为1，参见“随机统计测试说明”
        set_rng_seed(1)
        # 保存当前默认的张量数据类型
        saved_dtype = torch.get_default_dtype()
        # 遍历不同的数据类型：float、double、bfloat16、half
        for dtype in [torch.float, torch.double, torch.bfloat16, torch.half]:
            # 设置当前默认的张量数据类型为循环变量 dtype
            torch.set_default_dtype(dtype)
            # 遍历不同的泊松分布的参数 lambda：0.1, 1.0, 5.0
            for rate in [0.1, 1.0, 5.0]:
                # 调用 _check_sampler_discrete 方法，验证泊松分布采样的一致性
                self._check_sampler_discrete(
                    Poisson(rate),  # 创建泊松分布对象
                    scipy.stats.poisson(rate),  # 使用 SciPy 的泊松分布作为对比
                    f"Poisson(lambda={rate})",  # 打印的泊松分布的参数信息
                    failure_rate=1e-3,  # 容许的失败率阈值
                )
        # 恢复默认的张量数据类型
        torch.set_default_dtype(saved_dtype)

    # 如果 CUDA 不可用则跳过测试
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_poisson_gpu_sample(self):
        # 设置随机数生成器种子为1
        set_rng_seed(1)
        # 遍历不同的泊松分布的参数 lambda：0.12, 0.9, 4.0
        for rate in [0.12, 0.9, 4.0]:
            # 调用 _check_sampler_discrete 方法，验证 CUDA 下泊松分布采样的一致性
            self._check_sampler_discrete(
                Poisson(torch.tensor([rate]).cuda()),  # 创建 CUDA 下的泊松分布对象
                scipy.stats.poisson(rate),  # 使用 SciPy 的泊松分布作为对比
                f"Poisson(lambda={rate}, cuda)",  # 打印的 CUDA 下泊松分布的参数信息
                failure_rate=1e-3,  # 容许的失败率阈值
            )

    # 使用双精度浮点数作为默认的张量数据类型进行测试
    @set_default_dtype(torch.double)
    def test_relaxed_bernoulli(self):
        # 创建张量 p，并标记为需要梯度计算
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        # 创建张量 r，并标记为需要梯度计算
        r = torch.tensor(0.3, requires_grad=True)
        # 创建标量 s
        s = 0.3
        # 创建张量 temp，并标记为需要梯度计算
        temp = torch.tensor(0.67, requires_grad=True)
        # 断言 RelaxedBernoulli(temp, p) 采样的形状为 (8, 3)
        self.assertEqual(RelaxedBernoulli(temp, p).sample((8,)).size(), (8, 3))
        # 断言 RelaxedBernoulli(temp, p) 单个样本不需要梯度计算
        self.assertFalse(RelaxedBernoulli(temp, p).sample().requires_grad)
        # 断言 RelaxedBernoulli(temp, r) 采样的形状为 (8,)
        self.assertEqual(RelaxedBernoulli(temp, r).sample((8,)).size(), (8,))
        # 断言 RelaxedBernoulli(temp, r) 单个样本的形状为 ()
        self.assertEqual(RelaxedBernoulli(temp, r).sample().size(), ())
        # 断言 RelaxedBernoulli(temp, r) 采样的形状为 (3, 2)
        self.assertEqual(
            RelaxedBernoulli(temp, r).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        # 断言 RelaxedBernoulli(temp, s) 单个样本的形状为 ()
        self.assertEqual(RelaxedBernoulli(temp, s).sample().size(), ())
        # 对 RelaxedBernoulli(temp, p) 和 RelaxedBernoulli(temp, r) 的 log_prob 方法进行梯度检查
        self._gradcheck_log_prob(RelaxedBernoulli, (temp, p))
        self._gradcheck_log_prob(RelaxedBernoulli, (temp, r))

        # 测试 rsample 方法不会失败
        # 调用 rsample 方法并进行反向传播
        s = RelaxedBernoulli(temp, p).rsample()
        s.backward(torch.ones_like(s))

    # 如果 Numpy 不可用则跳过测试
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_rounded_relaxed_bernoulli(self):
        set_rng_seed(0)  # 设置随机数种子为0，参见注释[随机化统计测试]

        class Rounded:
            def __init__(self, dist):
                self.dist = dist

            def sample(self, *args, **kwargs):
                # 对给定分布采样后取整，返回取整后的结果
                return torch.round(self.dist.sample(*args, **kwargs))

        # 遍历概率和温度的组合，进行离散采样器的检查
        for probs, temp in product([0.1, 0.2, 0.8], [0.1, 1.0, 10.0]):
            self._check_sampler_discrete(
                Rounded(RelaxedBernoulli(temp, probs)),
                scipy.stats.bernoulli(probs),
                f"Rounded(RelaxedBernoulli(temp={temp}, probs={probs}))",
                failure_rate=1e-3,
            )

        # 对于不同的概率值，验证等概率样本是否与期望的相等
        for probs in [0.001, 0.2, 0.999]:
            equal_probs = torch.tensor(0.5)
            dist = RelaxedBernoulli(1e10, probs)
            s = dist.rsample()
            self.assertEqual(equal_probs, s)

    @set_default_dtype(torch.double)
    def test_relaxed_one_hot_categorical_1d(self):
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        temp = torch.tensor(0.67, requires_grad=True)
        # 验证样本的形状是否为(3,)
        self.assertEqual(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample().size(), (3,)
        )
        # 验证样本是否不需要梯度
        self.assertFalse(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample().requires_grad
        )
        # 验证不同sample_shape下的样本形状是否正确
        self.assertEqual(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample((2, 2)).size(),
            (2, 2, 3),
        )
        # 验证在指定sample_shape=(1,)下的样本形状是否正确
        self.assertEqual(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample((1,)).size(),
            (1, 3),
        )
        # 对log_prob进行梯度检查
        self._gradcheck_log_prob(
            lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp, p)
        )

    @set_default_dtype(torch.double)
    def test_relaxed_one_hot_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        temp = torch.tensor([3.0], requires_grad=True)
        # 通过温度值的不同，注释控制对样本的log_prob梯度检查
        # 当温度值低于0.25时，相对于样本，log_prob的梯度检查更加不稳定
        temp_2 = torch.tensor([0.25], requires_grad=True)
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        # 验证样本的形状是否为(2, 3)
        self.assertEqual(RelaxedOneHotCategorical(temp, p).sample().size(), (2, 3))
        # 验证在指定sample_shape=(3, 4)下的样本形状是否正确
        self.assertEqual(
            RelaxedOneHotCategorical(temp, p).sample(sample_shape=(3, 4)).size(),
            (3, 4, 2, 3),
        )
        # 验证在指定sample_shape=(6,)下的样本形状是否正确
        self.assertEqual(
            RelaxedOneHotCategorical(temp, p).sample((6,)).size(), (6, 2, 3)
        )
        # 对两个不同温度值下的log_prob进行梯度检查
        self._gradcheck_log_prob(
            lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp, p)
        )
        self._gradcheck_log_prob(
            lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False),
            (temp_2, p),
        )
    # 如果没有安装 Numpy，跳过这个测试
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    # 定义测试函数 test_argmax_relaxed_categorical
    def test_argmax_relaxed_categorical(self):
        # 设置随机数生成器种子为 0，参见“随机统计测试注意事项”
        set_rng_seed(0)  # see Note [Randomized statistical tests]

        # 定义 ArgMax 类
        class ArgMax:
            def __init__(self, dist):
                self.dist = dist

            # 从分布中抽样的方法
            def sample(self, *args, **kwargs):
                # 从分布中抽样并返回最大值的索引
                s = self.dist.sample(*args, **kwargs)
                _, idx = torch.max(s, -1)
                return idx

        # 定义 ScipyCategorical 类
        class ScipyCategorical:
            def __init__(self, dist):
                self.dist = dist

            # 概率质量函数方法
            def pmf(self, samples):
                # 创建新的样本矩阵，对应概率分布的概率位置设置为 1
                new_samples = np.zeros(samples.shape + self.dist.p.shape)
                new_samples[np.arange(samples.shape[0]), samples] = 1
                return self.dist.pmf(new_samples)

        # 遍历概率和温度的组合
        for probs, temp in product(
            [torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2, 0.6])], [0.1, 1.0, 10.0]
        ):
            # 对 _check_sampler_discrete 方法进行测试，使用 ArgMax 和 ScipyCategorical 类
            self._check_sampler_discrete(
                ArgMax(RelaxedOneHotCategorical(temp, probs)),
                ScipyCategorical(scipy.stats.multinomial(1, probs)),
                f"Rounded(RelaxedOneHotCategorical(temp={temp}, probs={probs}))",
                failure_rate=1e-3,
            )

        # 遍历概率的组合
        for probs in [torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2, 0.6])]:
            # 创建所有概率相等的 tensor
            equal_probs = torch.ones(probs.size()) / probs.size()[0]
            # 使用 RelaxedOneHotCategorical 分布创建 dist 对象
            dist = RelaxedOneHotCategorical(1e10, probs)
            # 从分布中抽样
            s = dist.rsample()
            # 断言抽样结果与均匀分布的期望相等
            self.assertEqual(equal_probs, s)

    # 将默认的 torch 数据类型设置为 double
    @set_default_dtype(torch.double)
    # 定义单元测试方法，测试 Uniform 分布的采样功能

    def test_uniform(self):
        # 创建一个形状为 5x5 的张量 low，所有元素为 0，并开启梯度跟踪
        low = torch.zeros(5, 5, requires_grad=True)
        # 创建一个形状为 5x5 的张量 high，所有元素为 3，并开启梯度跟踪
        high = (torch.ones(5, 5) * 3).requires_grad_()
        # 创建一个形状为 1 的张量 low_1d，元素为 0，并开启梯度跟踪
        low_1d = torch.zeros(1, requires_grad=True)
        # 创建一个形状为 1 的张量 high_1d，元素为 3，并开启梯度跟踪
        high_1d = (torch.ones(1) * 3).requires_grad_()

        # 测试 Uniform 分布的采样方法，期望返回形状为 (5, 5)
        self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
        # 测试 Uniform 分布的采样方法，期望返回形状为 (7, 5, 5)
        self.assertEqual(Uniform(low, high).sample((7,)).size(), (7, 5, 5))
        # 测试 Uniform 分布的采样方法，期望返回形状为 (1,)
        self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
        # 测试 Uniform 分布的采样方法，期望返回形状为 (1, 1)
        self.assertEqual(Uniform(low_1d, high_1d).sample((1,)).size(), (1, 1))
        # 测试 Uniform 分布的采样方法，期望返回形状为 (1,)
        self.assertEqual(Uniform(0.0, 1.0).sample((1,)).size(), (1,))

        # 检查在 validate_args=False 的情况下，当采样值超出范围时的对数概率计算
        uniform = Uniform(low_1d, high_1d, validate_args=False)
        above_high = torch.tensor([4.0])
        below_low = torch.tensor([-1.0])
        # 检查当采样值为 above_high 时的对数概率，期望为 -inf
        self.assertEqual(uniform.log_prob(above_high).item(), -inf)
        # 检查当采样值为 below_low 时的对数概率，期望为 -inf
        self.assertEqual(uniform.log_prob(below_low).item(), -inf)

        # 检查当采样值超出范围时的累积分布函数计算
        # 检查当采样值为 below_low 时的累积分布函数，期望为 0
        self.assertEqual(uniform.cdf(below_low).item(), 0)
        # 检查当采样值为 above_high 时的累积分布函数，期望为 1
        self.assertEqual(uniform.cdf(above_high).item(), 1)

        # 设置随机数种子为 1，检查 Uniform 分布对数概率的梯度检查
        set_rng_seed(1)
        self._gradcheck_log_prob(Uniform, (low, high))
        self._gradcheck_log_prob(Uniform, (low, 1.0))
        self._gradcheck_log_prob(Uniform, (0.0, high))

        # 获取当前随机数生成器状态，生成一个与 low 相同形状的随机数 rand
        state = torch.get_rng_state()
        rand = low.new(low.size()).uniform_()
        # 恢复随机数生成器状态为之前的状态
        torch.set_rng_state(state)
        # 从 Uniform 分布中采样一个样本 u
        u = Uniform(low, high).rsample()
        # 计算 u 的梯度
        u.backward(torch.ones_like(u))
        # 检查 low 的梯度是否为 1 - rand
        self.assertEqual(low.grad, 1 - rand)
        # 检查 high 的梯度是否为 rand
        self.assertEqual(high.grad, rand)
        # 将 low 和 high 的梯度清零
        low.grad.zero_()
        high.grad.zero_()

        # 检查调用 _check_forward_ad 方法时 Uniform 分布的前向自动求导功能
        self._check_forward_ad(lambda x: x.uniform_())

    # 如果没有安装 NumPy，则跳过此测试用例
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_vonmises_sample(self):
        # 对不同的 loc 和 concentration 参数进行 VonMises 分布的采样测试
        for loc in [0.0, math.pi / 2.0]:
            for concentration in [0.03, 0.3, 1.0, 10.0, 100.0]:
                # 调用 _check_sampler_sampler 方法，检查采样结果与 NumPy 中的 vonmises 分布采样结果的一致性
                self._check_sampler_sampler(
                    VonMises(loc, concentration),
                    scipy.stats.vonmises(loc=loc, kappa=concentration),
                    f"VonMises(loc={loc}, concentration={concentration})",
                    num_samples=int(1e5),
                    circular=True,
                )

    # 测试 VonMises 分布的对数概率计算功能
    def test_vonmises_logprob(self):
        # 对不同的 concentration 参数进行 VonMises 分布的对数概率测试
        concentrations = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
        for concentration in concentrations:
            # 创建一个从 0 到 2π 的网格
            grid = torch.arange(0.0, 2 * math.pi, 1e-4)
            # 计算 VonMises 分布在网格上的对数概率，并转换为概率值
            prob = VonMises(0.0, concentration).log_prob(grid).exp()
            # 计算概率的平均值乘以 2π，应接近 1
            norm = prob.mean().item() * 2 * math.pi
            # 断言计算得到的概率均值与 1 的差异小于 1e-3
            self.assertLess(abs(norm - 1), 1e-3)

    # 设置默认张量类型为双精度浮点数类型
    @set_default_dtype(torch.double)
    # 定义一个测试方法，用于测试 Cauchy 分布的功能
    def test_cauchy(self):
        # 创建一个形状为 (5, 5) 的零张量，并开启梯度追踪
        loc = torch.zeros(5, 5, requires_grad=True)
        # 创建一个形状为 (5, 5) 的单位张量，并开启梯度追踪
        scale = torch.ones(5, 5, requires_grad=True)
        # 创建一个形状为 (1,) 的零张量，并开启梯度追踪
        loc_1d = torch.zeros(1, requires_grad=True)
        # 创建一个形状为 (1,) 的单位张量，并开启梯度追踪
        scale_1d = torch.ones(1, requires_grad=True)
        
        # 断言 Cauchy 分布的均值是否全为 NaN
        self.assertTrue(is_all_nan(Cauchy(loc_1d, scale_1d).mean))
        # 断言 Cauchy 分布的方差是否为无穷大
        self.assertEqual(Cauchy(loc_1d, scale_1d).variance, inf)
        # 断言从 Cauchy 分布中采样的张量形状为 (5, 5)
        self.assertEqual(Cauchy(loc, scale).sample().size(), (5, 5))
        # 断言从 Cauchy 分布中采样的张量形状为 (7, 5, 5)
        self.assertEqual(Cauchy(loc, scale).sample((7,)).size(), (7, 5, 5))
        # 断言从 Cauchy 分布中采样的张量形状为 (1,)
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample().size(), (1,))
        # 断言从 Cauchy 分布中采样的张量形状为 (1, 1)
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        # 断言从 Cauchy 分布中采样的张量形状为 (1,)
        self.assertEqual(Cauchy(0.0, 1.0).sample((1,)).size(), (1,))

        # 设置随机数种子为 1
        set_rng_seed(1)
        # 对 Cauchy 分布的 log_prob 方法进行梯度检查
        self._gradcheck_log_prob(Cauchy, (loc, scale))
        self._gradcheck_log_prob(Cauchy, (loc, 1.0))
        self._gradcheck_log_prob(Cauchy, (0.0, scale))

        # 保存当前随机数生成器的状态
        state = torch.get_rng_state()
        # 在 loc 的形状上生成服从 Cauchy 分布的随机数
        eps = loc.new(loc.size()).cauchy_()
        # 恢复之前保存的随机数生成器的状态
        torch.set_rng_state(state)
        # 从 Cauchy 分布中采样一个值
        c = Cauchy(loc, scale).rsample()
        # 对采样值进行反向传播
        c.backward(torch.ones_like(c))
        # 断言 loc 的梯度是否与 scale 的全一张量相同
        self.assertEqual(loc.grad, torch.ones_like(scale))
        # 断言 scale 的梯度是否与 eps 相同
        self.assertEqual(scale.grad, eps)
        # 清空 loc 和 scale 的梯度
        loc.grad.zero_()
        scale.grad.zero_()

        # 对 _check_forward_ad 方法进行前向自动微分检查

    # 将默认数据类型设置为 double 类型，并定义测试方法 test_halfcauchy
    @set_default_dtype(torch.double)
    def test_halfcauchy(self):
        # 创建一个形状为 (5, 5) 的单位张量，并开启梯度追踪
        scale = torch.ones(5, 5, requires_grad=True)
        # 创建一个形状为 (1,) 的单位张量，并开启梯度追踪
        scale_1d = torch.ones(1, requires_grad=True)
        
        # 断言 HalfCauchy 分布的均值是否全为无穷大
        self.assertTrue(torch.isinf(HalfCauchy(scale_1d).mean).all())
        # 断言 HalfCauchy 分布的方差是否为无穷大
        self.assertEqual(HalfCauchy(scale_1d).variance, inf)
        # 断言从 HalfCauchy 分布中采样的张量形状为 (5, 5)
        self.assertEqual(HalfCauchy(scale).sample().size(), (5, 5))
        # 断言从 HalfCauchy 分布中采样的张量形状为 (7, 5, 5)
        self.assertEqual(HalfCauchy(scale).sample((7,)).size(), (7, 5, 5))
        # 断言从 HalfCauchy 分布中采样的张量形状为 (1,)
        self.assertEqual(HalfCauchy(scale_1d).sample().size(), (1,))
        # 断言从 HalfCauchy 分布中采样的张量形状为 (1, 1)
        self.assertEqual(HalfCauchy(scale_1d).sample((1,)).size(), (1, 1))
        # 断言从 HalfCauchy 分布中采样的张量形状为 (1,)
        self.assertEqual(HalfCauchy(1.0).sample((1,)).size(), (1,))

        # 设置随机数种子为 1
        set_rng_seed(1)
        # 对 HalfCauchy 分布的 log_prob 方法进行梯度检查
        self._gradcheck_log_prob(HalfCauchy, (scale,))
        self._gradcheck_log_prob(HalfCauchy, (1.0,))

        # 保存当前随机数生成器的状态
        state = torch.get_rng_state()
        # 在 scale 的形状上生成服从 Cauchy 分布的非负随机数
        eps = scale.new(scale.size()).cauchy_().abs_()
        # 恢复之前保存的随机数生成器的状态
        torch.set_rng_state(state)
        # 从 HalfCauchy 分布中采样一个值
        c = HalfCauchy(scale).rsample()
        # 对采样值进行反向传播
        c.backward(torch.ones_like(c))
        # 断言 scale 的梯度是否与 eps 相同
        self.assertEqual(scale.grad, eps)
        # 清空 scale 的梯度
        scale.grad.zero_()
    # 定义一个名为 test_halfnormal 的测试方法
    def test_halfnormal(self):
        # 创建一个 5x5 的张量 std，其值为标准正态分布随机数的绝对值，并标记需要计算梯度
        std = torch.randn(5, 5).abs().requires_grad_()
        # 创建一个标量 std_1d，其值为标准正态分布随机数的绝对值，并标记需要计算梯度
        std_1d = torch.randn(1).abs().requires_grad_()
        # 创建一个张量 std_delta，其值为 [1e-5, 1e-5]
        std_delta = torch.tensor([1e-5, 1e-5])
        # 断言 HalfNormal(std) 采样结果的尺寸为 (5, 5)
        self.assertEqual(HalfNormal(std).sample().size(), (5, 5))
        # 断言 HalfNormal(std) 采样结果的尺寸为 (7, 5, 5)
        self.assertEqual(HalfNormal(std).sample((7,)).size(), (7, 5, 5))
        # 断言 HalfNormal(std_1d) 采样结果的尺寸为 (1, 1)
        self.assertEqual(HalfNormal(std_1d).sample((1,)).size(), (1, 1))
        # 断言 HalfNormal(std_1d) 采样结果的尺寸为 (1,)
        self.assertEqual(HalfNormal(std_1d).sample().size(), (1,))
        # 断言 HalfNormal(0.6) 采样结果的尺寸为 (1,)
        self.assertEqual(HalfNormal(0.6).sample((1,)).size(), (1,))
        # 断言 HalfNormal(50.0) 采样结果的尺寸为 (1,)
        self.assertEqual(HalfNormal(50.0).sample((1,)).size(), (1,))

        # 对极端 std 值进行采样检查
        set_rng_seed(1)
        self.assertEqual(
            HalfNormal(std_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
            atol=1e-4,
            rtol=0,
        )

        # 对 HalfNormal 分布的 log_prob 进行梯度检查
        self._gradcheck_log_prob(HalfNormal, (std,))
        self._gradcheck_log_prob(HalfNormal, (1.0,))

        # 检查 .log_prob() 方法的广播性质
        # 创建一个 HalfNormal 分布对象 dist，其尺度为 torch.ones(2, 1, 4)
        dist = HalfNormal(torch.ones(2, 1, 4))
        # 计算给定样本的 log_prob，并断言其形状为 (2, 3, 4)
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))

    # 若未安装 NumPy，则跳过以下测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_logprob(self):
        # 创建一个尺寸为 (5, 1) 的张量 std，其值为标准正态分布随机数的绝对值，并标记需要计算梯度
        std = torch.randn(5, 1).abs().requires_grad_()

        # 定义一个用于参考 log_prob 的函数 ref_log_prob
        def ref_log_prob(idx, x, log_prob):
            # 从 std 中提取索引为 idx 的视图，分离出其值并将其称为 s
            s = std.view(-1)[idx].detach()
            # 使用 scipy.stats.halfnorm(scale=s) 计算 x 的对数概率密度预期值
            expected = scipy.stats.halfnorm(scale=s).logpdf(x)
            # 断言 log_prob 等于预期值 expected，允许的绝对误差为 1e-3，相对误差为 0
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 使用 _check_log_prob 方法验证 HalfNormal(std) 分布的 log_prob 方法
        self._check_log_prob(HalfNormal(std), ref_log_prob)

    # 若未安装 NumPy，则跳过以下测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_sample(self):
        # 设置随机数生成种子为 0
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        # 遍历不同的 std 值进行采样检查
        for std in [0.1, 1.0, 10.0]:
            # 使用 _check_sampler_sampler 方法验证 HalfNormal(std) 采样结果的正确性
            self._check_sampler_sampler(
                HalfNormal(std),
                scipy.stats.halfnorm(scale=std),
                f"HalfNormal(scale={std})",
            )

    # 将默认数据类型设置为 torch.double 后，执行测试方法 test_inversegamma
    @set_default_dtype(torch.double)
    def test_inversegamma(self):
        # 创建尺寸为 (2, 3) 的张量 alpha，其值为标准正态分布随机数的指数值，并标记需要计算梯度
        alpha = torch.randn(2, 3).exp().requires_grad_()
        # 创建尺寸为 (2, 3) 的张量 beta，其值为标准正态分布随机数的指数值，并标记需要计算梯度
        beta = torch.randn(2, 3).exp().requires_grad_()
        # 断言 InverseGamma(alpha, beta) 采样结果的尺寸为 (2, 3)
        self.assertEqual(InverseGamma(alpha, beta).sample().size(), (2, 3))
        # 断言 InverseGamma(alpha, beta) 采样结果的尺寸为 (5, 2, 3)
        self.assertEqual(InverseGamma(alpha, beta).sample((5,)).size(), (5, 2, 3))
        # 断言 InverseGamma(alpha_1d, beta_1d) 采样结果的尺寸为 (1, 1)
        self.assertEqual(InverseGamma(alpha_1d, beta_1d).sample((1,)).size(), (1, 1))
        # 断言 InverseGamma(alpha_1d, beta_1d) 采样结果的尺寸为 (1,)
        self.assertEqual(InverseGamma(alpha_1d, beta_1d).sample().size(), (1,))
        # 断言 InverseGamma(0.5, 0.5) 采样结果的尺寸为 ()
        self.assertEqual(InverseGamma(0.5, 0.5).sample().size(), ())
        # 断言 InverseGamma(0.5, 0.5) 采样结果的尺寸为 (1,)
        self.assertEqual(InverseGamma(0.5, 0.5).sample((1,)).size(), (1,))

        # 对 InverseGamma 分布的 log_prob 进行梯度检查
        self._gradcheck_log_prob(InverseGamma, (alpha, beta))

        # 创建一个 InverseGamma 分布对象 dist，其 alpha 为 torch.ones(4)，beta 为 torch.ones(2, 1, 1)
        dist = InverseGamma(torch.ones(4), torch.ones(2, 1, 1))
        # 计算给定样本的 log_prob，并断言其形状为 (2, 3, 4)
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))
    # 如果没有安装 NumPy，则跳过该测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_inversegamma_sample(self):
        # 设置随机数种子为0，用于随机化统计测试
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        # 对于集中度和速率的组合进行迭代测试
        for concentration, rate in product([2, 5], [0.1, 1.0, 10.0]):
            # 调用 _check_sampler_sampler 方法，对 InverseGamma 分布进行采样测试
            self._check_sampler_sampler(
                InverseGamma(concentration, rate),
                # 使用 SciPy 中的 Inverse Gamma 分布进行对比
                scipy.stats.invgamma(concentration, scale=rate),
                "InverseGamma()",
            )
    
    # 设置默认的 Torch 数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    def test_lognormal(self):
        # 创建具有随机正态分布值的张量，并标记为需要梯度
        mean = torch.randn(5, 5, requires_grad=True)
        # 创建具有随机正态分布绝对值的张量，并标记为需要梯度
        std = torch.randn(5, 5).abs().requires_grad_()
        # 创建具有随机正态分布值的一维张量，并标记为需要梯度
        mean_1d = torch.randn(1, requires_grad=True)
        # 创建具有随机正态分布绝对值的一维张量，并标记为需要梯度
        std_1d = torch.randn(1).abs().requires_grad_()
        # 创建固定的平均值变量和标准差变量
        mean_delta = torch.tensor([1.0, 0.0])
        std_delta = torch.tensor([1e-5, 1e-5])
        # 对 LogNormal 分布进行采样并验证样本大小
        self.assertEqual(LogNormal(mean, std).sample().size(), (5, 5))
        # 对 LogNormal 分布进行多次采样并验证样本大小
        self.assertEqual(LogNormal(mean, std).sample((7,)).size(), (7, 5, 5))
        # 对具有一维参数的 LogNormal 分布进行采样并验证样本大小
        self.assertEqual(LogNormal(mean_1d, std_1d).sample((1,)).size(), (1, 1))
        # 对具有一维参数的 LogNormal 分布进行采样并验证样本大小
        self.assertEqual(LogNormal(mean_1d, std_1d).sample().size(), (1,))
        # 对具有固定参数的 LogNormal 分布进行采样并验证样本大小
        self.assertEqual(LogNormal(0.2, 0.6).sample((1,)).size(), (1,))
        # 对具有固定参数的 LogNormal 分布进行采样并验证样本大小
        self.assertEqual(LogNormal(-0.7, 50.0).sample((1,)).size(), (1,))
    
        # 对极端均值和标准差进行采样检查
        set_rng_seed(1)
        self.assertEqual(
            LogNormal(mean_delta, std_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[math.exp(1), 1.0], [math.exp(1), 1.0]]]),
            atol=1e-4,
            rtol=0,
        )
    
        # 使用梯度检查方法检查 LogNormal 分布的对数概率密度函数
        self._gradcheck_log_prob(LogNormal, (mean, std))
        self._gradcheck_log_prob(LogNormal, (mean, 1.0))
        self._gradcheck_log_prob(LogNormal, (0.0, std))
    
        # 检查 .log_prob() 方法是否可以进行广播
        dist = LogNormal(torch.zeros(4), torch.ones(2, 1, 1))
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))
    
        # 使用 _check_forward_ad 方法检查 LogNormal 分布的自动微分计算
        self._check_forward_ad(lambda x: x.log_normal_())
    
    # 如果没有安装 NumPy，则跳过该测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_logprob(self):
        # 创建具有随机正态分布值的张量，并标记为需要梯度
        mean = torch.randn(5, 1, requires_grad=True)
        # 创建具有随机正态分布绝对值的张量，并标记为需要梯度
        std = torch.randn(5, 1).abs().requires_grad_()
    
        # 定义用于参考对数概率密度函数的函数
        def ref_log_prob(idx, x, log_prob):
            # 获取均值和标准差的视图并分离梯度
            m = mean.view(-1)[idx].detach()
            s = std.view(-1)[idx].detach()
            # 使用 SciPy 的 lognorm 分布计算预期的对数概率密度函数值
            expected = scipy.stats.lognorm(s=s, scale=math.exp(m)).logpdf(x)
            # 断言计算出的对数概率密度函数值与预期值相等，容忍度为 1e-3
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)
    
        # 使用 _check_log_prob 方法检查 LogNormal 分布的对数概率密度函数
        self._check_log_prob(LogNormal(mean, std), ref_log_prob)
    
    # 如果没有安装 NumPy，则跳过该测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_sample(self):
        # 设置随机数种子为0，用于随机化统计测试
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        # 对于给定的均值和标准差的组合进行迭代测试
        for mean, std in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            # 调用 _check_sampler_sampler 方法，对 LogNormal 分布进行采样测试
            self._check_sampler_sampler(
                LogNormal(mean, std),
                # 使用 SciPy 中的 LogNormal 分布进行对比
                scipy.stats.lognorm(scale=math.exp(mean), s=std),
                f"LogNormal(loc={mean}, scale={std})",
            )
    # 设置默认的张量数据类型为双精度浮点型
    @set_default_dtype(torch.double)
    # 定义测试函数 test_logisticnormal
    def test_logisticnormal(self):
        # 设定随机数生成器种子为1，见注释 [Randomized statistical tests]
        set_rng_seed(1)
        # 创建一个形状为(5, 5)的随机张量 mean，并设置其需要梯度计算
        mean = torch.randn(5, 5).requires_grad_()
        # 创建一个形状为(5, 5)的随机张量 std，并取其绝对值后设置需要梯度计算
        std = torch.randn(5, 5).abs().requires_grad_()
        # 创建一个形状为(1,)的随机张量 mean_1d，并设置其需要梯度计算
        mean_1d = torch.randn(1).requires_grad_()
        # 创建一个形状为(1,)的随机张量 std_1d，并取其绝对值后设置需要梯度计算
        std_1d = torch.randn(1).abs().requires_grad_()
        # 创建一个张量 mean_delta，值为[1.0, 0.0]
        mean_delta = torch.tensor([1.0, 0.0])
        # 创建一个张量 std_delta，值为[1e-5, 1e-5]
        std_delta = torch.tensor([1e-5, 1e-5])

        # 断言生成的 LogisticNormal(mean, std) 样本的形状为 (5, 6)
        self.assertEqual(LogisticNormal(mean, std).sample().size(), (5, 6))
        # 断言生成的 LogisticNormal(mean, std) 样本的形状为 (7, 5, 6)
        self.assertEqual(LogisticNormal(mean, std).sample((7,)).size(), (7, 5, 6))
        # 断言生成的 LogisticNormal(mean_1d, std_1d) 样本的形状为 (1, 2)
        self.assertEqual(LogisticNormal(mean_1d, std_1d).sample((1,)).size(), (1, 2))
        # 断言生成的 LogisticNormal(mean_1d, std_1d) 样本的形状为 (2,)
        self.assertEqual(LogisticNormal(mean_1d, std_1d).sample().size(), (2,))
        # 断言生成的 LogisticNormal(0.2, 0.6) 样本的形状为 (2,)
        self.assertEqual(LogisticNormal(0.2, 0.6).sample().size(), (2,))
        # 断言生成的 LogisticNormal(-0.7, 50.0) 样本的形状为 (2,)
        self.assertEqual(LogisticNormal(-0.7, 50.0).sample().size(), (2,))

        # 对于均值和标准差为极端值时的样本检查
        set_rng_seed(1)
        # 断言生成的 LogisticNormal(mean_delta, std_delta) 样本与预期张量一致
        self.assertEqual(
            LogisticNormal(mean_delta, std_delta).sample(),
            torch.tensor(
                [
                    math.exp(1) / (1.0 + 1.0 + math.exp(1)),
                    1.0 / (1.0 + 1.0 + math.exp(1)),
                    1.0 / (1.0 + 1.0 + math.exp(1)),
                ]
            ),
            atol=1e-4,  # 允许的绝对误差
            rtol=0,     # 允许的相对误差
        )

        # TODO: gradcheck 似乎会修改样本值，导致单纯限制条件稍有不足
        # 使用 _gradcheck_log_prob 方法检查对数概率梯度
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (mean, std)
        )
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (mean, 1.0)
        )
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (0.0, std)
        )

    # 如果未安装 NumPy 则跳过此测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义测试函数 test_logisticnormal_logprob
    def test_logisticnormal_logprob(self):
        # 创建一个形状为 (5, 7) 的随机张量 mean，并设置其需要梯度计算
        mean = torch.randn(5, 7).requires_grad_()
        # 创建一个形状为 (5, 7) 的随机张量 std，并取其绝对值后设置需要梯度计算
        std = torch.randn(5, 7).abs().requires_grad_()

        # 简单测试，目前只是烟雾测试
        # TODO: 一旦 _check_log_prob 能处理多维分布，添加对概率对数的正确测试
        # 创建 LogisticNormal 分布对象
        dist = LogisticNormal(mean, std)
        # 断言对数概率的形状为 (5,)
        assert dist.log_prob(dist.sample()).detach().cpu().numpy().shape == (5,)

    # 返回一个基于基础分布的逻辑正态分布参考采样器
    def _get_logistic_normal_ref_sampler(self, base_dist):
        # 定义内部函数 _sampler(num_samples)
        def _sampler(num_samples):
            # 从 base_dist 中生成 num_samples 个样本 x
            x = base_dist.rvs(num_samples)
            # 计算偏移量 offset
            offset = np.log((x.shape[-1] + 1) - np.ones_like(x).cumsum(-1))
            # 计算逻辑正态分布 z
            z = 1.0 / (1.0 + np.exp(offset - x))
            # 计算累积乘积 z_cumprod
            z_cumprod = np.cumprod(1 - z, axis=-1)
            # 构建 y1 和 y2
            y1 = np.pad(z, ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
            y2 = np.pad(
                z_cumprod, ((0, 0), (1, 0)), mode="constant", constant_values=1.0
            )
            # 返回 y1 * y2
            return y1 * y2

        # 返回内部定义的 _sampler 函数
        return _sampler
    # 如果未安装 NumPy，则跳过该单元测试；用于条件跳过测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义 logisticnormal_sample 测试方法
    def test_logisticnormal_sample(self):
        # 设置随机数种子为 0，参见注释 [Randomized statistical tests]
        set_rng_seed(0)
        
        # 定义均值和协方差的生成器
        means = map(np.asarray, [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)])
        covs = map(np.diag, [(0.1, 0.1), (1.0, 1.0), (10.0, 10.0)])
        
        # 对每一组均值和协方差进行迭代
        for mean, cov in product(means, covs):
            # 创建基础多变量正态分布对象
            base_dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            # 创建参考多变量正态分布对象
            ref_dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            # 设置参考分布的随机采样器为自定义的 logistic normal 采样器
            ref_dist.rvs = self._get_logistic_normal_ref_sampler(base_dist)
            
            # 转换均值和协方差为 PyTorch 张量
            mean_th = torch.tensor(mean)
            std_th = torch.tensor(np.sqrt(np.diag(cov)))
            
            # 执行检查样本生成器的方法
            self._check_sampler_sampler(
                LogisticNormal(mean_th, std_th),  # 测试 LogisticNormal 类
                ref_dist,  # 参考分布
                f"LogisticNormal(loc={mean_th}, scale={std_th})",  # 日志消息
                multivariate=True,  # 多变量设置为真
            )
    # 定义测试函数，用于测试混合同一家族分布的形状
    def test_mixture_same_family_shape(self):
        # 创建一个一维正态分布的混合分布，分布系数为随机生成的概率分布，分布数据为随机生成的正态分布
        normal_case_1d = MixtureSameFamily(
            Categorical(torch.rand(5)), Normal(torch.randn(5), torch.rand(5))
        )
        # 创建一个批次的一维正态分布混合分布，分布系数为随机生成的概率分布（批次维度为3），分布数据为随机生成的正态分布（批次维度为3）
        normal_case_1d_batch = MixtureSameFamily(
            Categorical(torch.rand(3, 5)), Normal(torch.randn(3, 5), torch.rand(3, 5))
        )
        # 创建一个多批次的一维正态分布混合分布，分布系数为随机生成的概率分布（批次维度为4x3x5），分布数据为随机生成的正态分布（批次维度为4x3x5）
        normal_case_1d_multi_batch = MixtureSameFamily(
            Categorical(torch.rand(4, 3, 5)),
            Normal(torch.randn(4, 3, 5), torch.rand(4, 3, 5)),
        )
        # 创建一个二维正态分布的混合分布，分布系数为随机生成的概率分布，分布数据为独立的二维正态分布（均值和标准差均为随机生成）
        normal_case_2d = MixtureSameFamily(
            Categorical(torch.rand(5)),
            Independent(Normal(torch.randn(5, 2), torch.rand(5, 2)), 1),
        )
        # 创建一个批次的二维正态分布混合分布，分布系数为随机生成的概率分布（批次维度为3），分布数据为独立的二维正态分布（批次维度为3x5x2）
        normal_case_2d_batch = MixtureSameFamily(
            Categorical(torch.rand(3, 5)),
            Independent(Normal(torch.randn(3, 5, 2), torch.rand(3, 5, 2)), 1),
        )
        # 创建一个多批次的二维正态分布混合分布，分布系数为随机生成的概率分布（批次维度为4x3x5），分布数据为独立的二维正态分布（批次维度为4x3x5x2）
        normal_case_2d_multi_batch = MixtureSameFamily(
            Categorical(torch.rand(4, 3, 5)),
            Independent(Normal(torch.randn(4, 3, 5, 2), torch.rand(4, 3, 5, 2)), 1),
        )

        # 断言，验证从各个分布中采样后得到的数据形状是否符合预期
        self.assertEqual(normal_case_1d.sample().size(), ())  # 一维正态分布的采样结果应该是标量
        self.assertEqual(normal_case_1d.sample((2,)).size(), (2,))  # 一维正态分布批次采样结果应该是形状为(2,)的向量
        self.assertEqual(normal_case_1d.sample((2, 7)).size(), (2, 7))  # 一维正态分布批次采样结果应该是形状为(2, 7)的矩阵
        self.assertEqual(normal_case_1d_batch.sample().size(), (3,))  # 批次一维正态分布采样结果应该是形状为(3,)的向量
        self.assertEqual(normal_case_1d_batch.sample((2,)).size(), (2, 3))  # 批次一维正态分布批次采样结果应该是形状为(2, 3)的矩阵
        self.assertEqual(normal_case_1d_batch.sample((2, 7)).size(), (2, 7, 3))  # 批次一维正态分布批次采样结果应该是形状为(2, 7, 3)的张量
        self.assertEqual(normal_case_1d_multi_batch.sample().size(), (4, 3))  # 多批次一维正态分布采样结果应该是形状为(4, 3)的矩阵
        self.assertEqual(normal_case_1d_multi_batch.sample((2,)).size(), (2, 4, 3))  # 多批次一维正态分布批次采样结果应该是形状为(2, 4, 3)的张量
        self.assertEqual(normal_case_1d_multi_batch.sample((2, 7)).size(), (2, 7, 4, 3))  # 多批次一维正态分布批次采样结果应该是形状为(2, 7, 4, 3)的张量

        # 断言，验证从各个二维分布中采样后得到的数据形状是否符合预期
        self.assertEqual(normal_case_2d.sample().size(), (2,))  # 二维正态分布的采样结果应该是形状为(2,)的向量
        self.assertEqual(normal_case_2d.sample((2,)).size(), (2, 2))  # 二维正态分布批次采样结果应该是形状为(2, 2)的矩阵
        self.assertEqual(normal_case_2d.sample((2, 7)).size(), (2, 7, 2))  # 二维正态分布批次采样结果应该是形状为(2, 7, 2)的张量
        self.assertEqual(normal_case_2d_batch.sample().size(), (3, 2))  # 批次二维正态分布采样结果应该是形状为(3, 2)的矩阵
        self.assertEqual(normal_case_2d_batch.sample((2,)).size(), (2, 3, 2))  # 批次二维正态分布批次采样结果应该是形状为(2, 3, 2)的张量
        self.assertEqual(normal_case_2d_batch.sample((2, 7)).size(), (2, 7, 3, 2))  # 批次二维正态分布批次采样结果应该是形状为(2, 7, 3, 2)的张量
        self.assertEqual(normal_case_2d_multi_batch.sample().size(), (4, 3, 2))  # 多批次二维正态分布采样结果应该是形状为(4, 3, 2)的张量
        self.assertEqual(normal_case_2d_multi_batch.sample((2,)).size(), (2, 4, 3, 2))  # 多批次二维正态分布批次采样结果应该是形状为(2, 4, 3, 2)的张量
        self.assertEqual(
            normal_case_2d_multi_batch.sample((2, 7)).size(), (2, 7, 4, 3, 2)
        )  # 多批次二维正态分布批次采样结果应该是形状为(2, 7, 4, 3, 2)的张量

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    # 定义一个测试函数，用于测试混合分布相同家族的对数概率计算
    def test_mixture_same_family_log_prob(self):
        # 生成一个形状为(5, 5)的随机概率张量，并对最后一维进行softmax归一化
        probs = torch.rand(5, 5).softmax(dim=-1)
        # 生成一个形状为(5, 5)的随机均值张量
        loc = torch.randn(5, 5)
        # 生成一个形状为(5, 5)的随机标准差张量
        scale = torch.rand(5, 5)

        # 定义一个参考对数概率计算函数，接受索引idx、数据x和对数概率log_prob作为参数
        def ref_log_prob(idx, x, log_prob):
            # 将概率张量转换为NumPy数组
            p = probs[idx].numpy()
            # 将均值张量转换为NumPy数组
            m = loc[idx].numpy()
            # 将标准差张量转换为NumPy数组
            s = scale[idx].numpy()
            # 创建一个多项式分布对象，用给定的概率p
            mix = scipy.stats.multinomial(1, p)
            # 创建一个正态分布对象，以给定的均值m和标准差s
            comp = scipy.stats.norm(m, s)
            # 计算混合分布的对数概率的期望值
            expected = scipy.special.logsumexp(comp.logpdf(x) + np.log(mix.p))
            # 使用断言检查计算得到的对数概率是否与期望值一致，允许的绝对误差为1e-3，相对误差为0
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 调用私有方法_check_log_prob，验证MixtureSameFamily对象的对数概率计算
        self._check_log_prob(
            MixtureSameFamily(Categorical(probs=probs), Normal(loc, scale)),
            ref_log_prob,
        )

    # 如果未安装NumPy，则跳过此测试
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    # 定义一个测试函数，用于测试混合分布相同家族的采样功能
    def test_mixture_same_family_sample(self):
        # 生成一个长度为5的随机概率张量，并对其进行softmax归一化
        probs = torch.rand(5).softmax(dim=-1)
        # 生成一个长度为5的随机均值张量
        loc = torch.randn(5)
        # 生成一个长度为5的随机标准差张量
        scale = torch.rand(5)

        # 定义一个模拟Scipy混合正态分布的类
        class ScipyMixtureNormal:
            def __init__(self, probs, mu, std):
                self.probs = probs
                self.mu = mu
                self.std = std

            # 定义一个采样方法，返回n_sample个样本
            def rvs(self, n_sample):
                # 对每个分量使用正态分布生成n_sample个样本
                comp_samples = [
                    scipy.stats.norm(m, s).rvs(n_sample)
                    for m, s in zip(self.mu, self.std)
                ]
                # 使用多项式分布生成n_sample个混合分量的选择样本
                mix_samples = scipy.stats.multinomial(1, self.probs).rvs(n_sample)
                samples = []
                # 将对应于混合分量选择样本的正态分量样本汇总为最终采样结果
                for i in range(n_sample):
                    samples.append(comp_samples[mix_samples[i].argmax()][i])
                return np.asarray(samples)

        # 调用私有方法_check_sampler_sampler，验证MixtureSameFamily对象的采样功能
        self._check_sampler_sampler(
            MixtureSameFamily(Categorical(probs=probs), Normal(loc, scale)),
            ScipyMixtureNormal(probs.numpy(), loc.numpy(), scale.numpy()),
            # 创建一个描述MixtureSameFamily对象的字符串
            f"""MixtureSameFamily(Categorical(probs={probs}),
            Normal(loc={loc}, scale={scale}))""",
        )

    # 将默认张量类型设置为双精度浮点型
    @set_default_dtype(torch.double)
    # 定义测试方法，用于测试正常情况下的正态分布抽样操作
    def test_normal(self):
        # 生成一个5x5的张量loc，其元素服从正态分布，并且允许计算梯度
        loc = torch.randn(5, 5, requires_grad=True)
        # 生成一个5x5的张量scale，其元素服从正态分布取绝对值后并允许计算梯度
        scale = torch.randn(5, 5).abs().requires_grad_()
        # 生成一个1维张量loc_1d，元素服从正态分布并允许计算梯度
        loc_1d = torch.randn(1, requires_grad=True)
        # 生成一个1维张量scale_1d，元素服从正态分布取绝对值后并允许计算梯度
        scale_1d = torch.randn(1).abs().requires_grad_()
        # 定义一个张量loc_delta，其元素为[1.0, 0.0]
        loc_delta = torch.tensor([1.0, 0.0])
        # 定义一个张量scale_delta，其元素为[1e-5, 1e-5]
        scale_delta = torch.tensor([1e-5, 1e-5])
        
        # 断言正态分布对象Normal生成的样本大小为(5, 5)
        self.assertEqual(Normal(loc, scale).sample().size(), (5, 5))
        # 断言正态分布对象Normal生成的样本大小为(7, 5, 5)
        self.assertEqual(Normal(loc, scale).sample((7,)).size(), (7, 5, 5))
        # 断言正态分布对象Normal生成的样本大小为(1, 1)
        self.assertEqual(Normal(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        # 断言正态分布对象Normal生成的样本大小为(1,)
        self.assertEqual(Normal(loc_1d, scale_1d).sample().size(), (1,))
        # 断言均值为0.2，标准差为0.6的正态分布对象生成的样本大小为(1,)
        self.assertEqual(Normal(0.2, 0.6).sample((1,)).size(), (1,))
        # 断言均值为-0.7，标准差为50.0的正态分布对象生成的样本大小为(1,)
        self.assertEqual(Normal(-0.7, 50.0).sample((1,)).size(), (1,))

        # 设置随机数种子为1，检查正态分布对象Normal在极端均值和标准差情况下的样本生成
        set_rng_seed(1)
        self.assertEqual(
            Normal(loc_delta, scale_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            atol=1e-4,
            rtol=0,
        )

        # 对Normal分布的对数概率进行梯度检查
        self._gradcheck_log_prob(Normal, (loc, scale))
        self._gradcheck_log_prob(Normal, (loc, 1.0))
        self._gradcheck_log_prob(Normal, (0.0, scale))

        # 保存当前随机数生成器状态
        state = torch.get_rng_state()
        # 生成服从标准正态分布的扰动项eps
        eps = torch.normal(torch.zeros_like(loc), torch.ones_like(scale))
        # 恢复随机数生成器状态
        torch.set_rng_state(state)
        # 从正态分布对象Normal中抽取样本z
        z = Normal(loc, scale).rsample()
        # 计算z关于自身的梯度
        z.backward(torch.ones_like(z))
        # 断言loc的梯度为全1张量
        self.assertEqual(loc.grad, torch.ones_like(loc))
        # 断言scale的梯度等于eps
        self.assertEqual(scale.grad, eps)
        # 将loc和scale的梯度清零
        loc.grad.zero_()
        scale.grad.zero_()
        # 断言z的大小为(5, 5)
        self.assertEqual(z.size(), (5, 5))

        # 定义函数ref_log_prob，用于检查给定索引处的对数概率值是否正确
        def ref_log_prob(idx, x, log_prob):
            m = loc.view(-1)[idx]
            s = scale.view(-1)[idx]
            expected = math.exp(-((x - m) ** 2) / (2 * s**2)) / math.sqrt(
                2 * math.pi * s**2
            )
            # 断言计算得到的对数概率值与预期值相符
            self.assertEqual(log_prob, math.log(expected), atol=1e-3, rtol=0)

        # 使用_check_log_prob方法检查正态分布对象Normal的对数概率计算
        self._check_log_prob(Normal(loc, scale), ref_log_prob)
        # 使用_check_forward_ad方法检查torch.normal函数的自动微分性质
        self._check_forward_ad(torch.normal)
        # 使用_check_forward_ad方法检查带有固定标准差0.5的torch.normal函数的自动微分性质
        self._check_forward_ad(lambda x: torch.normal(x, 0.5))
        # 使用_check_forward_ad方法检查带有固定均值0.2的torch.normal函数的自动微分性质
        self._check_forward_ad(lambda x: torch.normal(0.2, x))
        # 使用_check_forward_ad方法检查torch.normal方法自身的自动微分性质
        self._check_forward_ad(lambda x: x.normal_())
    # 定义一个单元测试方法，用于测试低秩多变量正态分布的对数概率计算
    def test_lowrank_multivariate_normal_log_prob(self):
        # 生成一个随机的均值张量，需要计算梯度
        mean = torch.randn(3, requires_grad=True)
        # 生成一个随机的低秩因子张量，需要计算梯度
        cov_factor = torch.randn(3, 1, requires_grad=True)
        # 生成一个随机的对角协方差张量，取绝对值后需要计算梯度
        cov_diag = torch.randn(3).abs().requires_grad_()
        # 计算完整的协方差矩阵，由低秩因子乘以其转置加上对角协方差得到
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()

        # 检查对数概率值与 scipy 的 logpdf 是否匹配，
        # 同时检查协方差和 scale_tril 参数是否等效
        dist1 = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        ref_dist = scipy.stats.multivariate_normal(
            mean.detach().numpy(), cov.detach().numpy()
        )

        # 生成 dist1 分布下的样本 x
        x = dist1.sample((10,))
        # 计算预期的 logpdf 值
        expected = ref_dist.logpdf(x.numpy())

        # 断言计算得到的对数概率值与预期值的均方差的平均值接近于零
        self.assertEqual(
            0.0,
            np.mean((dist1.log_prob(x).detach().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )

        # 再次验证批处理版本的行为与非批处理版本相同
        mean = torch.randn(5, 3, requires_grad=True)
        cov_factor = torch.randn(5, 3, 2, requires_grad=True)
        cov_diag = torch.randn(5, 3).abs().requires_grad_()

        # 创建批处理版本的分布对象
        dist_batched = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        # 创建非批处理版本的分布对象列表
        dist_unbatched = [
            LowRankMultivariateNormal(mean[i], cov_factor[i], cov_diag[i])
            for i in range(mean.size(0))
        ]

        # 生成 dist_batched 分布下的样本 x
        x = dist_batched.sample((10,))
        # 计算批处理版本和非批处理版本的对数概率值
        batched_prob = dist_batched.log_prob(x)
        unbatched_prob = torch.stack(
            [dist_unbatched[i].log_prob(x[:, i]) for i in range(5)]
        ).t()

        # 断言批处理版本和非批处理版本的形状相同，并且内容在给定的公差范围内相等
        self.assertEqual(batched_prob.shape, unbatched_prob.shape)
        self.assertEqual(batched_prob, unbatched_prob, atol=1e-3, rtol=0)

    # 如果没有安装 NumPy，跳过该测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lowrank_multivariate_normal_sample(self):
        # 设置随机数种子，见注释 [随机化统计测试]
        set_rng_seed(0)
        # 生成一个随机的均值张量，需要计算梯度
        mean = torch.randn(5, requires_grad=True)
        # 生成一个随机的低秩因子张量，需要计算梯度
        cov_factor = torch.randn(5, 1, requires_grad=True)
        # 生成一个随机的对角协方差张量，取绝对值后需要计算梯度
        cov_diag = torch.randn(5).abs().requires_grad_()
        # 计算完整的协方差矩阵，由低秩因子乘以其转置加上对角协方差得到
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()

        # 使用 _check_sampler_sampler 方法检查采样器的行为
        self._check_sampler_sampler(
            LowRankMultivariateNormal(mean, cov_factor, cov_diag),
            scipy.stats.multivariate_normal(
                mean.detach().numpy(), cov.detach().numpy()
            ),
            f"LowRankMultivariateNormal(loc={mean}, cov_factor={cov_factor}, cov_diag={cov_diag})",
            multivariate=True,
        )
    # 测试低秩多变量正态分布的性质
    def test_lowrank_multivariate_normal_properties(self):
        # 生成随机均值
        loc = torch.randn(5)
        # 生成随机的协方差因子
        cov_factor = torch.randn(5, 2)
        # 生成随机的协方差对角线，取绝对值
        cov_diag = torch.randn(5).abs()
        # 计算协方差矩阵
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()
        # 创建低秩多变量正态分布对象 m1
        m1 = LowRankMultivariateNormal(loc, cov_factor, cov_diag)
        # 创建多元正态分布对象 m2，使用计算得到的均值和协方差
        m2 = MultivariateNormal(loc=loc, covariance_matrix=cov)
        # 检查两个分布对象的均值是否相等
        self.assertEqual(m1.mean, m2.mean)
        # 检查两个分布对象的方差是否相等
        self.assertEqual(m1.variance, m2.variance)
        # 检查两个分布对象的协方差矩阵是否相等
        self.assertEqual(m1.covariance_matrix, m2.covariance_matrix)
        # 检查两个分布对象的 scale_tril 属性是否相等
        self.assertEqual(m1.scale_tril, m2.scale_tril)
        # 检查两个分布对象的 precision_matrix 属性是否相等
        self.assertEqual(m1.precision_matrix, m2.precision_matrix)
        # 检查两个分布对象的熵是否相等
        self.assertEqual(m1.entropy(), m2.entropy())

    # 测试低秩多变量正态分布的矩（moments）
    def test_lowrank_multivariate_normal_moments(self):
        # 设置随机数生成器种子，用于随机统计测试
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        # 生成随机均值
        mean = torch.randn(5)
        # 生成随机的协方差因子
        cov_factor = torch.randn(5, 2)
        # 生成随机的协方差对角线，取绝对值
        cov_diag = torch.randn(5).abs()
        # 创建低秩多变量正态分布对象 d
        d = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        # 生成 d 的样本，用于计算经验均值
        samples = d.rsample((100000,))
        # 计算样本的经验均值
        empirical_mean = samples.mean(0)
        # 检查低秩多变量正态分布对象的均值是否与经验均值接近，给定绝对误差和相对误差
        self.assertEqual(d.mean, empirical_mean, atol=0.01, rtol=0)
        # 计算样本的经验方差
        empirical_var = samples.var(0)
        # 检查低秩多变量正态分布对象的方差是否与经验方差接近，给定绝对误差和相对误差
        self.assertEqual(d.variance, empirical_var, atol=0.02, rtol=0)

    # 使用双精度浮点数测试多元正态分布，精度矩阵保持稳定性
    @set_default_dtype(torch.double)
    @set_default_dtype(torch.double)
    def test_multivariate_normal_stable_with_precision_matrix(self):
        # 生成随机向量 x
        x = torch.randn(10)
        # 计算精度矩阵 P，采用 RBF 核函数
        P = torch.exp(-((x - x.unsqueeze(-1)) ** 2))  # RBF kernel
        # 创建多元正态分布对象，使用零向量均值和计算得到的精度矩阵 P
        MultivariateNormal(x.new_zeros(10), precision_matrix=P)

    # 如果没有安装 NumPy，则跳过测试
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    # 定义一个测试方法，用于测试多元正态分布的对数概率计算
    def test_multivariate_normal_log_prob(self):
        # 生成一个具有梯度的随机均值向量
        mean = torch.randn(3, requires_grad=True)
        # 生成一个3x10的随机张量
        tmp = torch.randn(3, 10)
        # 计算协方差矩阵，要求其具有梯度
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        # 计算精度矩阵（协方差矩阵的逆），要求其具有梯度
        prec = cov.inverse().requires_grad_()
        # 计算 Cholesky 分解的下三角矩阵，要求其具有梯度
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # 检查 logprob 值与 scipy 的 logpdf 是否匹配，
        # 同时检查协方差和 scale_tril 参数是否等效
        dist1 = MultivariateNormal(mean, cov)
        dist2 = MultivariateNormal(mean, precision_matrix=prec)
        dist3 = MultivariateNormal(mean, scale_tril=scale_tril)
        ref_dist = scipy.stats.multivariate_normal(
            mean.detach().numpy(), cov.detach().numpy()
        )

        # 从分布中采样一组样本
        x = dist1.sample((10,))
        # 计算预期的对数概率密度函数值
        expected = ref_dist.logpdf(x.numpy())

        # 断言：检查生成的对数概率与预期值的均方差是否接近0
        self.assertEqual(
            0.0,
            np.mean((dist1.log_prob(x).detach().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )
        self.assertEqual(
            0.0,
            np.mean((dist2.log_prob(x).detach().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )
        self.assertEqual(
            0.0,
            np.mean((dist3.log_prob(x).detach().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )

        # 再次确认批量版本的行为与非批量版本相同
        mean = torch.randn(5, 3, requires_grad=True)
        tmp = torch.randn(5, 3, 10)
        # 计算批量的协方差矩阵，要求其具有梯度
        cov = (tmp.unsqueeze(-2) * tmp.unsqueeze(-3)).mean(-1).requires_grad_()

        # 创建批量版本的多元正态分布对象
        dist_batched = MultivariateNormal(mean, cov)
        # 创建非批量版本的多元正态分布对象列表
        dist_unbatched = [
            MultivariateNormal(mean[i], cov[i]) for i in range(mean.size(0))
        ]

        # 从批量分布中采样一组样本
        x = dist_batched.sample((10,))
        # 计算批量的对数概率密度函数值
        batched_prob = dist_batched.log_prob(x)
        # 计算非批量的对数概率密度函数值，并转置结果
        unbatched_prob = torch.stack(
            [dist_unbatched[i].log_prob(x[:, i]) for i in range(5)]
        ).t()

        # 断言：检查批量和非批量版本的形状和数值是否一致
        self.assertEqual(batched_prob.shape, unbatched_prob.shape)
        self.assertEqual(batched_prob, unbatched_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义一个测试方法，用于测试多变量正态分布的采样功能
    def test_multivariate_normal_sample(self):
        # 设置随机数生成种子为0，用于统计测试的随机化处理
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        # 生成一个随机的均值向量，形状为(3,)，并且要求梯度计算
        mean = torch.randn(3, requires_grad=True)
        # 生成一个形状为(3, 10)的随机张量
        tmp = torch.randn(3, 10)
        # 计算并要求梯度计算的协方差矩阵
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        # 计算并要求梯度计算的精度矩阵，是协方差矩阵的逆矩阵
        prec = cov.inverse().requires_grad_()
        # 计算并要求梯度计算的下三角矩阵，是协方差矩阵的 Cholesky 分解
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # 对 MultivariateNormal 对象进行采样，并进行与 SciPy 多变量正态分布的比较
        self._check_sampler_sampler(
            MultivariateNormal(mean, cov),
            scipy.stats.multivariate_normal(
                mean.detach().numpy(), cov.detach().numpy()
            ),
            f"MultivariateNormal(loc={mean}, cov={cov})",
            multivariate=True,
        )
        # 对 MultivariateNormal 对象进行采样，使用精度矩阵进行构造，并与 SciPy 的比较
        self._check_sampler_sampler(
            MultivariateNormal(mean, precision_matrix=prec),
            scipy.stats.multivariate_normal(
                mean.detach().numpy(), cov.detach().numpy()
            ),
            f"MultivariateNormal(loc={mean}, atol={prec})",
            multivariate=True,
        )
        # 对 MultivariateNormal 对象进行采样，使用下三角矩阵进行构造，并与 SciPy 的比较
        self._check_sampler_sampler(
            MultivariateNormal(mean, scale_tril=scale_tril),
            scipy.stats.multivariate_normal(
                mean.detach().numpy(), cov.detach().numpy()
            ),
            f"MultivariateNormal(loc={mean}, scale_tril={scale_tril})",
            multivariate=True,
        )

    # 将默认数据类型设置为双精度类型，并定义多变量正态分布的属性测试方法
    @set_default_dtype(torch.double)
    def test_multivariate_normal_properties(self):
        # 生成一个随机的均值向量，形状为(5,)
        loc = torch.randn(5)
        # 生成一个随机的下三角矩阵，形状为(5, 5)，用于构造多变量正态分布对象
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(5, 5))
        # 创建多变量正态分布对象 m
        m = MultivariateNormal(loc=loc, scale_tril=scale_tril)
        # 检查多变量正态分布对象的协方差矩阵与其下三角矩阵乘以转置的结果是否一致
        self.assertEqual(m.covariance_matrix, m.scale_tril.mm(m.scale_tril.t()))
        # 检查多变量正态分布对象的协方差矩阵与其精度矩阵乘积结果是否接近单位矩阵
        self.assertEqual(
            m.covariance_matrix.mm(m.precision_matrix), torch.eye(m.event_shape[0])
        )
        # 检查多变量正态分布对象的下三角矩阵是否与其协方差矩阵的 Cholesky 分解一致
        self.assertEqual(m.scale_tril, torch.linalg.cholesky(m.covariance_matrix))

    # 将默认数据类型设置为双精度类型，并定义多变量正态分布的矩测试方法
    @set_default_dtype(torch.double)
    def test_multivariate_normal_moments(self):
        # 设置随机数生成种子为0，用于统计测试的随机化处理
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        # 生成一个随机的均值向量，形状为(5,)
        mean = torch.randn(5)
        # 生成一个随机的下三角矩阵，形状为(5, 5)，用于构造多变量正态分布对象
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(5, 5))
        # 创建多变量正态分布对象 d
        d = MultivariateNormal(mean, scale_tril=scale_tril)
        # 生成 d 的随机样本，样本数为 100000
        samples = d.rsample((100000,))
        # 计算样本的均值向量
        empirical_mean = samples.mean(0)
        # 检查多变量正态分布对象的均值向量是否与样本均值接近，容忍度为 0.01
        self.assertEqual(d.mean, empirical_mean, atol=0.01, rtol=0)
        # 计算样本的方差向量
        empirical_var = samples.var(0)
        # 检查多变量正态分布对象的方差向量是否与样本方差接近，容忍度为 0.05
        self.assertEqual(d.variance, empirical_var, atol=0.05, rtol=0)

    # 使用与多变量正态分布相同的测试方法来测试 Wishart 分布，使用精度矩阵作为参数
    @set_default_dtype(torch.double)
    def test_wishart_stable_with_precision_matrix(self):
        # 设置随机数生成种子为0，用于统计测试的随机化处理
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        # 设置 Wishart 分布的维度为 10
        ndim = 10
        # 生成一个随机向量 x，形状为(ndim,)
        x = torch.randn(ndim)
        # 计算并生成精度矩阵 P，采用 RBF 核函数
        P = torch.exp(-((x - x.unsqueeze(-1)) ** 2))  # RBF kernel
        # 创建 Wishart 分布对象，使用精度矩阵 P
        Wishart(torch.tensor(ndim), precision_matrix=P)

    # 如果未安装 NumPy，则跳过这个测试
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    # 将默认数据类型设置为双精度类型
    @set_default_dtype(torch.double)
    # 定义测试 Wishart 分布的对数概率函数
    def test_wishart_log_prob(self):
        # 设置随机数种子为0，用于随机化的统计测试，参见注释 [Randomized statistical tests]
        set_rng_seed(0)
        # 定义 Wishart 分布的维度为3
        ndim = 3
        # 创建一个随机数作为 Wishart 分布的自由度，并要求其梯度信息
        df = torch.rand([], requires_grad=True) + ndim - 1
        # 对于 Wishart 分布，SciPy 在版本 1.7.0 之后允许 ndim - 1 < df < ndim
        if version.parse(scipy.__version__) < version.parse("1.7.0"):
            df += 1.0
        # 生成一个形状为 (ndim, 10) 的随机张量
        tmp = torch.randn(ndim, 10)
        # 计算协方差矩阵，并要求其梯度信息
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        # 计算精度矩阵的逆，并要求其梯度信息
        prec = cov.inverse().requires_grad_()
        # 计算协方差矩阵的 Cholesky 分解，并要求其梯度信息
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # 检查 logprob 值是否与 scipy 的 logpdf 相匹配，
        # 并且协方差矩阵和 scale_tril 参数是否等价
        dist1 = Wishart(df, cov)
        dist2 = Wishart(df, precision_matrix=prec)
        dist3 = Wishart(df, scale_tril=scale_tril)
        # 使用 scipy 的 Wishart 分布作为参考分布
        ref_dist = scipy.stats.wishart(df.item(), cov.detach().numpy())

        # 从 dist1 中采样 1000 次，计算其对应的 logpdf
        x = dist1.sample((1000,))
        expected = ref_dist.logpdf(x.transpose(0, 2).numpy())

        # 断言检查 dist1 的 log_prob 是否与预期值 expected 相符
        self.assertEqual(
            0.0,
            np.mean((dist1.log_prob(x).detach().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )
        # 断言检查 dist2 的 log_prob 是否与预期值 expected 相符
        self.assertEqual(
            0.0,
            np.mean((dist2.log_prob(x).detach().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )
        # 断言检查 dist3 的 log_prob 是否与预期值 expected 相符
        self.assertEqual(
            0.0,
            np.mean((dist3.log_prob(x).detach().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )

        # 再次检查批处理版本是否与非批处理版本行为相同
        # 创建一个形状为 (5,) 的随机数，作为 Wishart 分布的自由度，并要求其梯度信息
        df = torch.rand(5, requires_grad=True) + ndim - 1
        # 对于 Wishart 分布，SciPy 在版本 1.7.0 之后允许 ndim - 1 < df < ndim
        if version.parse(scipy.__version__) < version.parse("1.7.0"):
            df += 1.0
        # 生成一个形状为 (5, ndim, 10) 的随机张量
        tmp = torch.randn(5, ndim, 10)
        # 计算协方差矩阵，并要求其梯度信息
        cov = (tmp.unsqueeze(-2) * tmp.unsqueeze(-3)).mean(-1).requires_grad_()

        # 创建批处理 Wishart 分布对象
        dist_batched = Wishart(df, cov)
        # 创建非批处理 Wishart 分布对象的列表
        dist_unbatched = [Wishart(df[i], cov[i]) for i in range(df.size(0))]

        # 从 dist_batched 中采样 1000 次，并计算其对数概率
        x = dist_batched.sample((1000,))
        # 计算批处理的 log_prob
        batched_prob = dist_batched.log_prob(x)
        # 计算非批处理的 log_prob，并进行转置
        unbatched_prob = torch.stack(
            [dist_unbatched[i].log_prob(x[:, i]) for i in range(5)]
        ).t()

        # 断言检查批处理和非批处理版本的形状和数值是否一致
        self.assertEqual(batched_prob.shape, unbatched_prob.shape)
        self.assertEqual(batched_prob, unbatched_prob, atol=1e-3, rtol=0)

    # 如果未找到 NumPy，则跳过测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 设置默认的张量类型为双精度浮点数
    @set_default_dtype(torch.double)
    # 定义一个测试方法，用于测试 Wishart 分布的采样功能
    def test_wishart_sample(self):
        set_rng_seed(0)  # 设置随机数种子，见注释 [Randomized statistical tests]
        ndim = 3
        # 生成一个随机数作为 Wishart 分布的自由度参数 df，要求梯度计算
        df = torch.rand([], requires_grad=True) + ndim - 1
        # 如果 SciPy 版本低于 1.7.0，调整 df 的值使得 ndim - 1 < df < ndim
        if version.parse(scipy.__version__) < version.parse("1.7.0"):
            df += 1.0
        # 生成一个 ndim x 10 的随机张量 tmp
        tmp = torch.randn(ndim, 10)
        # 计算 tmp 和它的转置的乘积，并求其均值，作为协方差矩阵 cov，并要求梯度计算
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        # 计算 cov 的逆矩阵作为精度矩阵 prec，并要求梯度计算
        prec = cov.inverse().requires_grad_()
        # 对 cov 进行 Cholesky 分解，得到下三角矩阵 scale_tril，并要求梯度计算
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # 使用 df 和 cov 创建 SciPy 的 Wishart 分布的参考分布 ref_dist
        ref_dist = scipy.stats.wishart(df.item(), cov.detach().numpy())

        # 使用自定义方法 _check_sampler_sampler 检查 Wishart 分布的采样器
        self._check_sampler_sampler(
            Wishart(df, cov),
            ref_dist,
            f"Wishart(df={df}, covariance_matrix={cov})",
            multivariate=True,
        )
        self._check_sampler_sampler(
            Wishart(df, precision_matrix=prec),
            ref_dist,
            f"Wishart(df={df}, precision_matrix={prec})",
            multivariate=True,
        )
        self._check_sampler_sampler(
            Wishart(df, scale_tril=scale_tril),
            ref_dist,
            f"Wishart(df={df}, scale_tril={scale_tril})",
            multivariate=True,
        )

    # 定义一个测试方法，用于验证 Wishart 分布的性质
    def test_wishart_properties(self):
        set_rng_seed(0)  # 设置随机数种子，见注释 [Randomized statistical tests]
        ndim = 5
        # 生成一个随机数作为 Wishart 分布的自由度参数 df
        df = torch.rand([]) + ndim - 1
        # 生成一个下三角矩阵 scale_tril，使用 lower_cholesky 转换器对随机生成的 ndim x ndim 张量进行处理
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(ndim, ndim))
        # 创建 Wishart 分布对象 m
        m = Wishart(df=df, scale_tril=scale_tril)
        # 验证协方差矩阵是否等于 scale_tril 乘以其转置
        self.assertEqual(m.covariance_matrix, m.scale_tril.mm(m.scale_tril.t()))
        # 验证协方差矩阵乘以精度矩阵是否等于单位矩阵
        self.assertEqual(
            m.covariance_matrix.mm(m.precision_matrix), torch.eye(m.event_shape[0])
        )
        # 验证 scale_tril 是否等于协方差矩阵的 Cholesky 分解
        self.assertEqual(m.scale_tril, torch.linalg.cholesky(m.covariance_matrix))

    # 定义一个测试方法，用于验证 Wishart 分布的矩估计
    def test_wishart_moments(self):
        set_rng_seed(0)  # 设置随机数种子，见注释 [Randomized statistical tests]
        ndim = 3
        # 生成一个随机数作为 Wishart 分布的自由度参数 df
        df = torch.rand([]) + ndim - 1
        # 生成一个下三角矩阵 scale_tril，使用 lower_cholesky 转换器对随机生成的 ndim x ndim 张量进行处理
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(ndim, ndim))
        # 创建 Wishart 分布对象 d
        d = Wishart(df=df, scale_tril=scale_tril)
        # 生成大量样本进行抽样
        samples = d.rsample((ndim * ndim * 100000,))
        # 计算样本的均值作为经验均值 empirical_mean
        empirical_mean = samples.mean(0)
        # 验证 Wishart 分布的理论均值是否与经验均值 empirical_mean 在容许误差范围内相等
        self.assertEqual(d.mean, empirical_mean, atol=0.5, rtol=0)
        # 计算样本的方差作为经验方差 empirical_var
        empirical_var = samples.var(0)
        # 验证 Wishart 分布的理论方差是否与经验方差 empirical_var 在容许误差范围内相等
        self.assertEqual(d.variance, empirical_var, atol=0.5, rtol=0)

    @set_default_dtype(torch.double)
    # 定义测试指数分布的函数
    def test_exponential(self):
        # 创建一个随机的绝对值张量，标记为需要梯度
        rate = torch.randn(5, 5).abs().requires_grad_()
        # 创建一个随机的绝对值张量（1维），标记为需要梯度
        rate_1d = torch.randn(1).abs().requires_grad_()
        # 断言从指数分布中抽样的张量形状为 (5, 5)
        self.assertEqual(Exponential(rate).sample().size(), (5, 5))
        # 断言从指数分布中抽样的张量形状为 (7, 5, 5)
        self.assertEqual(Exponential(rate).sample((7,)).size(), (7, 5, 5))
        # 断言从指数分布中抽样的张量形状为 (1, 1)
        self.assertEqual(Exponential(rate_1d).sample((1,)).size(), (1, 1))
        # 断言从指数分布中抽样的张量形状为 (1,)
        self.assertEqual(Exponential(rate_1d).sample().size(), (1,))
        # 断言从指数分布中抽样的张量形状为 (1,)
        self.assertEqual(Exponential(0.2).sample((1,)).size(), (1,))
        # 断言从指数分布中抽样的张量形状为 (1,)
        self.assertEqual(Exponential(50.0).sample((1,)).size(), (1,))

        # 对指数分布的 log_prob 方法进行梯度检查
        self._gradcheck_log_prob(Exponential, (rate,))
        # 保存当前随机数生成器状态
        state = torch.get_rng_state()
        # 生成一个指数分布样本
        eps = rate.new(rate.size()).exponential_()
        # 恢复之前保存的随机数生成器状态
        torch.set_rng_state(state)
        # 从指数分布中抽样一个样本，并计算反向传播
        z = Exponential(rate).rsample()
        z.backward(torch.ones_like(z))
        # 断言梯度与预期值相符
        self.assertEqual(rate.grad, -eps / rate**2)
        # 清零梯度
        rate.grad.zero_()
        # 断言抽样的张量形状为 (5, 5)
        self.assertEqual(z.size(), (5, 5))

        # 定义参考 log_prob 方法的函数
        def ref_log_prob(idx, x, log_prob):
            # 从 rate 中取出索引为 idx 的元素作为参数 m
            m = rate.view(-1)[idx]
            # 计算预期的 log_prob 值
            expected = math.log(m) - m * x
            # 断言 log_prob 值与预期值在给定的误差范围内相等
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 使用 ref_log_prob 函数检查 log_prob 方法的输出
        self._check_log_prob(Exponential(rate), ref_log_prob)
        # 使用 _check_forward_ad 方法检查自动求导是否正确
        self._check_forward_ad(lambda x: x.exponential_())

        # 定义计算均值和方差的函数
        def mean_var(lambd, sample):
            # 从指数分布中抽样数据
            sample.exponential_(lambd)
            # 计算抽样数据的均值
            mean = sample.float().mean()
            # 计算抽样数据的方差
            var = sample.float().var()
            # 断言均值与理论值在误差范围内相等
            self.assertEqual((1.0 / lambd), mean, atol=2e-2, rtol=2e-2)
            # 断言方差与理论值在误差范围内相等
            self.assertEqual((1.0 / lambd) ** 2, var, atol=2e-2, rtol=2e-2)

        # 遍历不同的数据类型和 lambda 值进行均值和方差的检查
        for dtype in [torch.float, torch.double, torch.bfloat16, torch.float16]:
            for lambd in [0.2, 0.5, 1.0, 1.5, 2.0, 5.0]:
                # 设置样本长度
                sample_len = 50000
                # 使用 mean_var 函数进行均值和方差的检查
                mean_var(lambd, torch.rand(sample_len, dtype=dtype))

    # 跳过测试，如果未安装 NumPy
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义测试指数分布的抽样函数
    def test_exponential_sample(self):
        # 设置随机数种子
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        # 遍历不同的 rate 值进行抽样器测试
        for rate in [1e-5, 1.0, 10.0]:
            # 使用 _check_sampler_sampler 方法检查抽样器的抽样效果
            self._check_sampler_sampler(
                Exponential(rate),
                scipy.stats.expon(scale=1.0 / rate),
                f"Exponential(rate={rate})",
            )

    # 将默认数据类型设置为双精度
    @set_default_dtype(torch.double)
    # Laplace分布的单元测试方法
    def test_laplace(self):
        # 生成一个5x5大小的随机张量loc，需要计算梯度
        loc = torch.randn(5, 5, requires_grad=True)
        # 生成一个5x5大小的随机张量scale，取绝对值并需要计算梯度
        scale = torch.randn(5, 5).abs().requires_grad_()
        # 生成一个1维随机张量loc_1d，需要计算梯度
        loc_1d = torch.randn(1, requires_grad=True)
        # 生成一个1维随机张量scale_1d，需要计算梯度
        scale_1d = torch.randn(1, requires_grad=True)
        # 创建一个长度为2的张量loc_delta
        loc_delta = torch.tensor([1.0, 0.0])
        # 创建一个长度为2的张量scale_delta
        scale_delta = torch.tensor([1e-5, 1e-5])
        
        # 检验从Laplace分布中抽样的张量大小是否符合预期
        self.assertEqual(Laplace(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Laplace(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Laplace(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Laplace(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Laplace(0.2, 0.6).sample((1,)).size(), (1,))
        self.assertEqual(Laplace(-0.7, 50.0).sample((1,)).size(), (1,))

        # 对于均值和标准差的极端值进行抽样检查
        set_rng_seed(0)
        self.assertEqual(
            Laplace(loc_delta, scale_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            atol=1e-4,
            rtol=0,
        )

        # 检验Laplace分布的对数概率密度函数梯度是否正确
        self._gradcheck_log_prob(Laplace, (loc, scale))
        self._gradcheck_log_prob(Laplace, (loc, 1.0))
        self._gradcheck_log_prob(Laplace, (0.0, scale))

        # 保存随机数生成器状态，生成eps，然后恢复状态
        state = torch.get_rng_state()
        eps = torch.ones_like(loc).uniform_(-0.5, 0.5)
        torch.set_rng_state(state)
        
        # 从Laplace分布中抽样，计算反向传播，检验梯度是否正确
        z = Laplace(loc, scale).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(loc.grad, torch.ones_like(loc))
        self.assertEqual(scale.grad, -eps.sign() * torch.log1p(-2 * eps.abs()))
        loc.grad.zero_()
        scale.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        # 定义用于检验对数概率的参考函数
        def ref_log_prob(idx, x, log_prob):
            m = loc.view(-1)[idx]
            s = scale.view(-1)[idx]
            expected = -math.log(2 * s) - abs(x - m) / s
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 使用参考函数检验Laplace分布的对数概率
        self._check_log_prob(Laplace(loc, scale), ref_log_prob)

    # 如果没有安装NumPy，则跳过该测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 使用双精度浮点数类型进行默认设置
    @set_default_dtype(torch.double)
    # Laplace分布的抽样测试方法
    def test_laplace_sample(self):
        set_rng_seed(1)  # 见注释[随机化统计测试]
        # 遍历loc和scale的组合，检验抽样器的一致性
        for loc, scale in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                Laplace(loc, scale),
                scipy.stats.laplace(loc=loc, scale=scale),
                f"Laplace(loc={loc}, scale={scale})",
            )
    # 定义 Gamma 分布的形状测试函数
    def test_gamma_shape(self):
        # 创建随机数张量 alpha 和 beta，应用指数函数并要求梯度
        alpha = torch.randn(2, 3).exp().requires_grad_()
        beta = torch.randn(2, 3).exp().requires_grad_()
        alpha_1d = torch.randn(1).exp().requires_grad_()
        beta_1d = torch.randn(1).exp().requires_grad_()
        # 断言 Gamma 分布的样本形状为 (2, 3)
        self.assertEqual(Gamma(alpha, beta).sample().size(), (2, 3))
        # 断言 Gamma 分布的样本形状为 (5, 2, 3)
        self.assertEqual(Gamma(alpha, beta).sample((5,)).size(), (5, 2, 3))
        # 断言 Gamma 分布的样本形状为 (1, 1)
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample((1,)).size(), (1, 1))
        # 断言 Gamma 分布的样本形状为 (1,)
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample().size(), (1,))
        # 断言 Gamma 分布的样本形状为 ()
        self.assertEqual(Gamma(0.5, 0.5).sample().size(), ())
        # 断言 Gamma 分布的样本形状为 (1,)
        self.assertEqual(Gamma(0.5, 0.5).sample((1,)).size(), (1,))

        # 定义参考对数概率函数，计算预期值
        def ref_log_prob(idx, x, log_prob):
            # 从 alpha 和 beta 中选择对应索引处的值，并分离出来
            a = alpha.view(-1)[idx].detach()
            b = beta.view(-1)[idx].detach()
            # 使用 SciPy 计算 Gamma 分布的对数概率密度函数的预期值
            expected = scipy.stats.gamma.logpdf(x, a, scale=1 / b)
            # 断言计算得到的对数概率与预期值相等，允许的绝对误差为 1e-3，相对误差为 0
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 调用内部方法 _check_log_prob，验证 Gamma 分布的对数概率计算
        self._check_log_prob(Gamma(alpha, beta), ref_log_prob)

    # 跳过 CUDA 未找到的情况，跳过 NumPy 未找到的情况
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义 Gamma 分布 GPU 形状测试函数
    def test_gamma_gpu_shape(self):
        # 创建随机数张量 alpha 和 beta，将其移到 CUDA 设备上，应用指数函数并要求梯度
        alpha = torch.randn(2, 3).cuda().exp().requires_grad_()
        beta = torch.randn(2, 3).cuda().exp().requires_grad_()
        alpha_1d = torch.randn(1).cuda().exp().requires_grad_()
        beta_1d = torch.randn(1).cuda().exp().requires_grad_()
        # 断言 Gamma 分布的样本形状为 (2, 3)
        self.assertEqual(Gamma(alpha, beta).sample().size(), (2, 3))
        # 断言 Gamma 分布的样本形状为 (5, 2, 3)
        self.assertEqual(Gamma(alpha, beta).sample((5,)).size(), (5, 2, 3))
        # 断言 Gamma 分布的样本形状为 (1, 1)
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample((1,)).size(), (1, 1))
        # 断言 Gamma 分布的样本形状为 (1,)
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample().size(), (1,))
        
        # 定义参考对数概率函数，计算预期值（在 CPU 上）
        def ref_log_prob(idx, x, log_prob):
            # 从 alpha 和 beta 中选择对应索引处的值，并分离出来（在 CPU 上）
            a = alpha.view(-1)[idx].detach().cpu()
            b = beta.view(-1)[idx].detach().cpu()
            # 使用 SciPy 计算 Gamma 分布的对数概率密度函数的预期值（在 CPU 上）
            expected = scipy.stats.gamma.logpdf(x.cpu(), a, scale=1 / b)
            # 断言计算得到的对数概率与预期值相等，允许的绝对误差为 1e-3，相对误差为 0
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 调用内部方法 _check_log_prob，验证 Gamma 分布的对数概率计算
        self._check_log_prob(Gamma(alpha, beta), ref_log_prob)

    # 跳过 CUDA 未找到的情况，跳过 NumPy 未找到的情况
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义 Gamma 分布采样测试函数
    def test_gamma_sample(self):
        # 设置随机数种子为 0，见注释 [Randomized statistical tests]
        set_rng_seed(0)
        # 遍历 alpha 和 beta 的组合，执行以下操作
        for alpha, beta in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            # 调用内部方法 _check_sampler_sampler，验证 Gamma 分布的采样器
            self._check_sampler_sampler(
                Gamma(alpha, beta),
                scipy.stats.gamma(alpha, scale=1.0 / beta),
                f"Gamma(concentration={alpha}, rate={beta})",
            )

    # 跳过 CUDA 未找到的情况，跳过 NumPy 未找到的情况
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_gamma_gpu_sample(self):
        # 设置随机数种子为0，确保结果可复现
        set_rng_seed(0)
        # 遍历 alpha 和 beta 的组合
        for alpha, beta in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            # 将 alpha 和 beta 转换为 CUDA 张量
            a, b = torch.tensor([alpha]).cuda(), torch.tensor([beta]).cuda()
            # 使用 Gamma 分布对象进行采样验证
            self._check_sampler_sampler(
                Gamma(a, b),
                # 使用 SciPy 的 Gamma 分布作为参考
                scipy.stats.gamma(alpha, scale=1.0 / beta),
                # 测试用例名称
                f"Gamma(alpha={alpha}, beta={beta})",
                # 失败率设定为 1e-4
                failure_rate=1e-4,
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_pareto(self):
        # 生成形状为 (2, 3) 的随机正数张量并要求梯度
        scale = torch.randn(2, 3).abs().requires_grad_()
        alpha = torch.randn(2, 3).abs().requires_grad_()
        # 生成形状为 (1,) 的随机正数张量并要求梯度
        scale_1d = torch.randn(1).abs().requires_grad_()
        alpha_1d = torch.randn(1).abs().requires_grad_()
        # 验证 Pareto 分布的均值是否为正无穷
        self.assertEqual(Pareto(scale_1d, 0.5).mean, inf)
        # 验证 Pareto 分布的方差是否为正无穷
        self.assertEqual(Pareto(scale_1d, 0.5).variance, inf)
        # 验证 Pareto 分布的样本形状是否为 (2, 3)
        self.assertEqual(Pareto(scale, alpha).sample().size(), (2, 3))
        # 验证 Pareto 分布的样本形状是否为 (5, 2, 3)
        self.assertEqual(Pareto(scale, alpha).sample((5,)).size(), (5, 2, 3))
        # 验证 Pareto 分布的样本形状是否为 (1, 1)
        self.assertEqual(Pareto(scale_1d, alpha_1d).sample((1,)).size(), (1, 1))
        # 验证 Pareto 分布的样本形状是否为 (1,)
        self.assertEqual(Pareto(scale_1d, alpha_1d).sample().size(), (1,))
        # 验证 Pareto 分布的样本形状是否为空
        self.assertEqual(Pareto(1.0, 1.0).sample().size(), ())
        # 验证 Pareto 分布的样本形状是否为 (1,)
        self.assertEqual(Pareto(1.0, 1.0).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            # 取出指定索引位置的 scale 和 alpha，不包括梯度信息
            s = scale.view(-1)[idx].detach()
            a = alpha.view(-1)[idx].detach()
            # 使用 SciPy 计算 Pareto 分布的对数概率密度函数作为参考
            expected = scipy.stats.pareto.logpdf(x, a, scale=s)
            # 验证计算得到的对数概率密度函数值与预期值在给定精度下相等
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 使用 _check_log_prob 方法验证 Pareto 分布的对数概率计算
        self._check_log_prob(Pareto(scale, alpha), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_pareto_sample(self):
        # 设置随机数种子为1，见 Note [Randomized statistical tests]
        set_rng_seed(1)
        # 遍历 scale 和 alpha 的组合
        for scale, alpha in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            # 使用 Pareto 分布对象进行采样验证
            self._check_sampler_sampler(
                Pareto(scale, alpha),
                # 使用 SciPy 的 Pareto 分布作为参考
                scipy.stats.pareto(alpha, scale=scale),
                # 测试用例名称
                f"Pareto(scale={scale}, alpha={alpha})",
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义单元测试方法，测试 Gumbel 分布的采样功能
    def test_gumbel(self):
        # 创建形状为 (2, 3) 的正态分布随机张量 loc，并要求计算梯度
        loc = torch.randn(2, 3, requires_grad=True)
        # 创建形状为 (2, 3) 的正态分布随机张量 scale，并要求计算梯度
        scale = torch.randn(2, 3).abs().requires_grad_()
        # 创建形状为 (1,) 的正态分布随机张量 loc_1d，并要求计算梯度
        loc_1d = torch.randn(1, requires_grad=True)
        # 创建形状为 (1,) 的正态分布随机张量 scale_1d，并要求计算梯度
        scale_1d = torch.randn(1).abs().requires_grad_()
        
        # 断言采样结果的形状为 (2, 3)
        self.assertEqual(Gumbel(loc, scale).sample().size(), (2, 3))
        # 断言采样结果的形状为 (5, 2, 3)
        self.assertEqual(Gumbel(loc, scale).sample((5,)).size(), (5, 2, 3))
        # 断言采样结果的形状为 (1,)
        self.assertEqual(Gumbel(loc_1d, scale_1d).sample().size(), (1,))
        # 断言采样结果的形状为 (1, 1)
        self.assertEqual(Gumbel(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        # 断言采样结果的形状为空元组 ()
        self.assertEqual(Gumbel(1.0, 1.0).sample().size(), ())
        # 断言采样结果的形状为 (1,)
        self.assertEqual(Gumbel(1.0, 1.0).sample((1,)).size(), (1,))
        
        # 断言对于给定的 Gumbel 分布参数，使用 validate_args=False 时，Gumbel 分布的累积分布函数值接近于 1.0
        self.assertEqual(
            Gumbel(
                torch.tensor(0.0, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
                validate_args=False,
            ).cdf(20.0),
            1.0,
            atol=1e-4,
            rtol=0,
        )
        # 断言对于给定的 Gumbel 分布参数，使用 validate_args=False 时，Gumbel 分布的累积分布函数值接近于 1.0
        self.assertEqual(
            Gumbel(
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
                validate_args=False,
            ).cdf(50.0),
            1.0,
            atol=1e-4,
            rtol=0,
        )
        # 断言对于给定的 Gumbel 分布参数，使用 validate_args=False 时，Gumbel 分布的累积分布函数值接近于 0.0
        self.assertEqual(
            Gumbel(
                torch.tensor(0.0, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
                validate_args=False,
            ).cdf(-5.0),
            0.0,
            atol=1e-4,
            rtol=0,
        )
        # 断言对于给定的 Gumbel 分布参数，使用 validate_args=False 时，Gumbel 分布的累积分布函数值接近于 0.0
        self.assertEqual(
            Gumbel(
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
                validate_args=False,
            ).cdf(-10.0),
            0.0,
            atol=1e-8,
            rtol=0,
        )

        # 定义内部函数 ref_log_prob，用于检查 Gumbel 分布的对数概率密度函数的参考实现
        def ref_log_prob(idx, x, log_prob):
            # 获取 loc 的视图，以及对应索引 idx 处的数值，且不会传播梯度
            l = loc.view(-1)[idx].detach()
            # 获取 scale 的视图，以及对应索引 idx 处的数值，且不会传播梯度
            s = scale.view(-1)[idx].detach()
            # 使用 scipy 库计算 Gumbel 分布的对数概率密度函数的预期值
            expected = scipy.stats.gumbel_r.logpdf(x, loc=l, scale=s)
            # 断言计算得到的对数概率密度函数值与预期值接近，允许的误差为 1e-3
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 调用 _check_log_prob 方法，验证 Gumbel 分布的对数概率密度函数的实现
        self._check_log_prob(Gumbel(loc, scale), ref_log_prob)

    # 装饰器 unittest.skipIf 检查是否存在 NumPy 库，如果不存在则跳过测试
    # 设置默认数据类型为 torch.double
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype(torch.double)
    # 定义单元测试方法，测试 Gumbel 分布的采样功能
    def test_gumbel_sample(self):
        # 设置随机数生成器的种子为 1，用于随机统计测试
        set_rng_seed(1)  # see note [Randomized statistical tests]
        # 遍历 loc 和 scale 的组合，进行 Gumbel 分布的采样测试
        for loc, scale in product([-5.0, -1.0, -0.1, 0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            # 调用 _check_sampler_sampler 方法，验证采样函数的正确性
            self._check_sampler_sampler(
                Gumbel(loc, scale),
                scipy.stats.gumbel_r(loc=loc, scale=scale),
                f"Gumbel(loc={loc}, scale={scale})",
            )
    # 定义测试函数，测试Kumaraswamy分布的形状
    def test_kumaraswamy_shape(self):
        # 创建随机张量，绝对值化并标记需要梯度
        concentration1 = torch.randn(2, 3).abs().requires_grad_()
        concentration0 = torch.randn(2, 3).abs().requires_grad_()
        concentration1_1d = torch.randn(1).abs().requires_grad_()
        concentration0_1d = torch.randn(1).abs().requires_grad_()
        # 断言样本形状为(2, 3)
        self.assertEqual(
            Kumaraswamy(concentration1, concentration0).sample().size(), (2, 3)
        )
        # 断言样本形状为(5, 2, 3)
        self.assertEqual(
            Kumaraswamy(concentration1, concentration0).sample((5,)).size(), (5, 2, 3)
        )
        # 断言样本形状为(1,)
        self.assertEqual(
            Kumaraswamy(concentration1_1d, concentration0_1d).sample().size(), (1,)
        )
        # 断言样本形状为(1, 1)
        self.assertEqual(
            Kumaraswamy(concentration1_1d, concentration0_1d).sample((1,)).size(),
            (1, 1),
        )
        # 断言样本形状为()
        self.assertEqual(Kumaraswamy(1.0, 1.0).sample().size(), ())
        # 断言样本形状为(1,)
        self.assertEqual(Kumaraswamy(1.0, 1.0).sample((1,)).size(), (1,))

    # Kumaraswamy分布在SciPy中未实现
    # 因此这些测试是显式的
    def test_kumaraswamy_mean_variance(self):
        # 创建随机张量，绝对值化并标记需要梯度
        c1_1 = torch.randn(2, 3).abs().requires_grad_()
        c0_1 = torch.randn(2, 3).abs().requires_grad_()
        c1_2 = torch.randn(4).abs().requires_grad_()
        c0_2 = torch.randn(4).abs().requires_grad_()
        # 构建测试用例列表
        cases = [(c1_1, c0_1), (c1_2, c0_2)]
        # 遍历测试用例
        for i, (a, b) in enumerate(cases):
            # 创建Kumaraswamy分布对象
            m = Kumaraswamy(a, b)
            # 生成60000个样本
            samples = m.sample((60000,))
            # 计算样本均值
            expected = samples.mean(0)
            # 获取分布的期望值
            actual = m.mean
            # 计算期望值误差
            error = (expected - actual).abs()
            # 获取最大误差
            max_error = max(error[error == error])
            # 断言期望值误差小于0.01
            self.assertLess(
                max_error,
                0.01,
                f"Kumaraswamy example {i + 1}/{len(cases)}, incorrect .mean",
            )
            # 计算样本方差
            expected = samples.var(0)
            # 获取分布的方差值
            actual = m.variance
            # 计算方差值误差
            error = (expected - actual).abs()
            # 获取最大误差
            max_error = max(error[error == error])
            # 断言方差值误差小于0.01
            self.assertLess(
                max_error,
                0.01,
                f"Kumaraswamy example {i + 1}/{len(cases)}, incorrect .variance",
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义一个单元测试方法，测试 Fisher-Snedecor 分布的不同功能
    def test_fishersnedecor(self):
        # 创建两个大小为 2x3 的随机张量，并要求梯度
        df1 = torch.randn(2, 3).abs().requires_grad_()
        df2 = torch.randn(2, 3).abs().requires_grad_()
        # 创建两个标量随机张量，不要求梯度
        df1_1d = torch.randn(1).abs()
        df2_1d = torch.randn(1).abs()
        # 断言 FisherSnedecor(1, 2).mean 返回的值全为 NaN
        self.assertTrue(is_all_nan(FisherSnedecor(1, 2).mean))
        # 断言 FisherSnedecor(1, 4).variance 返回的值全为 NaN
        self.assertTrue(is_all_nan(FisherSnedecor(1, 4).variance))
        # 断言 FisherSnedecor(df1, df2).sample() 返回的张量形状为 (2, 3)
        self.assertEqual(FisherSnedecor(df1, df2).sample().size(), (2, 3))
        # 断言 FisherSnedecor(df1, df2).sample((5,)) 返回的张量形状为 (5, 2, 3)
        self.assertEqual(FisherSnedecor(df1, df2).sample((5,)).size(), (5, 2, 3))
        # 断言 FisherSnedecor(df1_1d, df2_1d).sample() 返回的张量形状为 (1,)
        self.assertEqual(FisherSnedecor(df1_1d, df2_1d).sample().size(), (1,))
        # 断言 FisherSnedecor(df1_1d, df2_1d).sample((1,)) 返回的张量形状为 (1, 1)
        self.assertEqual(FisherSnedecor(df1_1d, df2_1d).sample((1,)).size(), (1, 1))
        # 断言 FisherSnedecor(1.0, 1.0).sample() 返回的张量形状为 ()
        self.assertEqual(FisherSnedecor(1.0, 1.0).sample().size(), ())
        # 断言 FisherSnedecor(1.0, 1.0).sample((1,)) 返回的张量形状为 (1,)
        self.assertEqual(FisherSnedecor(1.0, 1.0).sample((1,)).size(), (1,))

        # 定义一个用于参考的对数概率方法
        def ref_log_prob(idx, x, log_prob):
            # 从 df1 和 df2 的视图中分离出索引为 idx 的值，并且不再计算梯度
            f1 = df1.view(-1)[idx].detach()
            f2 = df2.view(-1)[idx].detach()
            # 计算使用 scipy 库计算的预期对数概率
            expected = scipy.stats.f.logpdf(x, f1, f2)
            # 断言 log_prob 与 expected 的值相等，允许的绝对误差为 1e-3，相对误差为 0
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 使用 _check_log_prob 方法检查 FisherSnedecor(df1, df2) 分布的对数概率
        self._check_log_prob(FisherSnedecor(df1, df2), ref_log_prob)

    # 如果未安装 NumPy，则跳过测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 测试 Fisher-Snedecor 分布的样本生成
    def test_fishersnedecor_sample(self):
        # 设置随机数生成种子为 1，参见 [随机化统计测试]
        set_rng_seed(1)
        # 对于 df1 和 df2 中的每个组合
        for df1, df2 in product([0.1, 0.5, 1.0, 5.0, 10.0], [0.1, 0.5, 1.0, 5.0, 10.0]):
            # 使用 _check_sampler_sampler 方法检查 FisherSnedecor 分布的样本生成
            self._check_sampler_sampler(
                FisherSnedecor(df1, df2),
                scipy.stats.f(df1, df2),
                f"FisherSnedecor(loc={df1}, scale={df2})",
            )

    # 如果未安装 NumPy，则跳过测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 测试 Chi2 分布的张量形状
    def test_chi2_shape(self):
        # 创建一个大小为 2x3 的随机张量并将其指数化，并要求梯度
        df = torch.randn(2, 3).exp().requires_grad_()
        # 创建一个大小为 1 的随机张量并将其指数化，并要求梯度
        df_1d = torch.randn(1).exp().requires_grad_()
        # 断言 Chi2(df).sample() 返回的张量形状为 (2, 3)
        self.assertEqual(Chi2(df).sample().size(), (2, 3))
        # 断言 Chi2(df).sample((5,)) 返回的张量形状为 (5, 2, 3)
        self.assertEqual(Chi2(df).sample((5,)).size(), (5, 2, 3))
        # 断言 Chi2(df_1d).sample((1,)) 返回的张量形状为 (1, 1)
        self.assertEqual(Chi2(df_1d).sample((1,)).size(), (1, 1))
        # 断言 Chi2(df_1d).sample() 返回的张量形状为 (1,)
        self.assertEqual(Chi2(df_1d).sample().size(), (1,))
        # 断言 Chi2(torch.tensor(0.5, requires_grad=True)).sample() 返回的张量形状为 ()
        self.assertEqual(
            Chi2(torch.tensor(0.5, requires_grad=True)).sample().size(), ()
        )
        # 断言 Chi2(0.5).sample() 返回的张量形状为 ()
        self.assertEqual(Chi2(0.5).sample().size(), ())
        # 断言 Chi2(0.5).sample((1,)) 返回的张量形状为 (1,)
        self.assertEqual(Chi2(0.5).sample((1,)).size(), (1,))

        # 定义一个用于参考的对数概率方法
        def ref_log_prob(idx, x, log_prob):
            # 从 df 的视图中分离出索引为 idx 的值，并且不再计算梯度
            d = df.view(-1)[idx].detach()
            # 计算使用 scipy 库计算的预期对数概率
            expected = scipy.stats.chi2.logpdf(x, d)
            # 断言 log_prob 与 expected 的值相等，允许的绝对误差为 1e-3，相对误差为 0
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 使用 _check_log_prob 方法检查 Chi2(df) 分布的对数概率
        self._check_log_prob(Chi2(df), ref_log_prob)

    # 如果未安装 NumPy，则跳过测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 测试 Chi2 分布的样本生成
    def test_chi2_sample(self):
        # 设置随机数生成种子为 0，参见 [随机化统计测试]
        set_rng_seed(0)
        # 对于每个 df 中的值
        for df in [0.1, 1.0, 5.0]:
            # 使用 _check_sampler_sampler 方法检查 Chi2 分布的样本生成
            self._check_sampler_sampler(
                Chi2(df), scipy.stats.chi2(df), f"Chi2(df={df})"
            )
    # 定义测试函数 test_studentT，用于测试学生 t 分布相关功能
    def test_studentT(self):
        # 创建一个具有梯度的 2x3 大小的张量，元素服从指数分布
        df = torch.randn(2, 3).exp().requires_grad_()
        # 创建一个具有梯度的大小为 1 的张量，元素服从指数分布
        df_1d = torch.randn(1).exp().requires_grad_()
        # 断言 StudentT(1).mean 的所有元素是否都为 NaN
        self.assertTrue(is_all_nan(StudentT(1).mean))
        # 断言 StudentT(1).variance 的所有元素是否都为 NaN
        self.assertTrue(is_all_nan(StudentT(1).variance))
        # 断言 StudentT(2).variance 是否为无穷大
        self.assertEqual(StudentT(2).variance, inf)
        # 断言从 StudentT(df) 中采样的张量大小是否为 (2, 3)
        self.assertEqual(StudentT(df).sample().size(), (2, 3))
        # 断言从 StudentT(df) 中采样的张量大小是否为 (5, 2, 3)
        self.assertEqual(StudentT(df).sample((5,)).size(), (5, 2, 3))
        # 断言从 StudentT(df_1d) 中采样的张量大小是否为 (1, 1)
        self.assertEqual(StudentT(df_1d).sample((1,)).size(), (1, 1))
        # 断言从 StudentT(df_1d) 中采样的张量大小是否为 (1,)
        self.assertEqual(StudentT(df_1d).sample().size(), (1,))
        # 断言从 StudentT(torch.tensor(0.5, requires_grad=True)) 中采样的张量大小是否为 ()
        self.assertEqual(
            StudentT(torch.tensor(0.5, requires_grad=True)).sample().size(), ()
        )
        # 断言从 StudentT(0.5) 中采样的张量大小是否为 ()
        self.assertEqual(StudentT(0.5).sample().size(), ())
        # 断言从 StudentT(0.5) 中采样的张量大小是否为 (1,)
        self.assertEqual(StudentT(0.5).sample((1,)).size(), (1,))

        # 定义引用函数 ref_log_prob，用于验证 log 概率
        def ref_log_prob(idx, x, log_prob):
            # 从 df 中视图中取出索引为 idx 的数据，并且将其分离（不再跟踪梯度）
            d = df.view(-1)[idx].detach()
            # 使用 scipy 计算 t 分布的 log 概率
            expected = scipy.stats.t.logpdf(x, d)
            # 断言计算出的 log_prob 与期望的 expected 在一定的数值误差范围内相等
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        # 调用 _check_log_prob 方法，传入 StudentT(df) 和 ref_log_prob 函数
        self._check_log_prob(StudentT(df), ref_log_prob)

    # 标记为跳过测试，条件是如果 TEST_NUMPY 为 False，则跳过测试
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    # 设置默认数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    # 定义测试函数 test_studentT_sample，用于测试学生 t 分布的采样功能
    def test_studentT_sample(self):
        # 设置随机数种子为 11，参见备注 [Randomized statistical tests]
        set_rng_seed(11)
        # 遍历 df、loc、scale 的所有组合
        for df, loc, scale in product(
            [0.1, 1.0, 5.0, 10.0], [-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]
        ):
            # 调用 _check_sampler_sampler 方法，用于检查采样器
            self._check_sampler_sampler(
                # 创建 StudentT 分布对象，指定 df、loc、scale
                StudentT(df=df, loc=loc, scale=scale),
                # 创建 scipy 中的 t 分布对象，指定 df、loc、scale
                scipy.stats.t(df=df, loc=loc, scale=scale),
                # 创建描述字符串，说明当前测试的参数组合
                f"StudentT(df={df}, loc={loc}, scale={scale})",
            )

    # 标记为跳过测试，条件是如果 TEST_NUMPY 为 False，则跳过测试
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    # 定义测试函数 test_studentT_log_prob，用于测试学生 t 分布的 log 概率计算功能
    def test_studentT_log_prob(self):
        # 设置随机数种子为 0，参见备注 [Randomized statistical tests]
        set_rng_seed(0)
        num_samples = 10
        # 遍历 df、loc、scale 的所有组合
        for df, loc, scale in product(
            [0.1, 1.0, 5.0, 10.0], [-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]
        ):
            # 创建 StudentT 分布对象，指定 df、loc、scale
            dist = StudentT(df=df, loc=loc, scale=scale)
            # 从分布中采样出大小为 (num_samples,) 的样本 x
            x = dist.sample((num_samples,))
            # 计算实际的 log 概率
            actual_log_prob = dist.log_prob(x)
            # 遍历每个样本
            for i in range(num_samples):
                # 使用 scipy 计算期望的 log 概率
                expected_log_prob = scipy.stats.t.logpdf(
                    x[i], df=df, loc=loc, scale=scale
                )
                # 断言计算出的 log 概率与期望的 expected 在一定的数值误差范围内相等
                self.assertEqual(
                    float(actual_log_prob[i]),
                    float(expected_log_prob),
                    atol=1e-3,
                    rtol=0,
                )

    # 定义测试函数 test_dirichlet_shape，用于测试 Dirichlet 分布的形状
    def test_dirichlet_shape(self):
        # 创建一个具有梯度的 2x3 大小的张量，元素服从指数分布
        alpha = torch.randn(2, 3).exp().requires_grad_()
        # 创建一个具有梯度的大小为 4 的张量，元素服从指数分布
        alpha_1d = torch.randn(4).exp().requires_grad_()
        # 断言从 Dirichlet(alpha) 中采样的张量大小是否为 (2, 3)
        self.assertEqual(Dirichlet(alpha).sample().size(), (2, 3))
        # 断言从 Dirichlet(alpha) 中采样的张量大小是否为 (5, 2, 3)
        self.assertEqual(Dirichlet(alpha).sample((5,)).size(), (5, 2, 3))
        # 断言从 Dirichlet(alpha_1d) 中采样的张量大小是否为 (4,)
        self.assertEqual(Dirichlet(alpha_1d).sample().size(), (4,))
        # 断言从 Dirichlet(alpha_1d) 中采样的张量大小是否为 (1, 4)
        self.assertEqual(Dirichlet(alpha_1d).sample((1,)).size(), (1, 4))
    def test_dirichlet_log_prob(self):
        # 设定采样的次数
        num_samples = 10
        # 生成随机的 Dirichlet 分布参数 alpha，并取指数
        alpha = torch.exp(torch.randn(5))
        # 创建 Dirichlet 分布对象
        dist = Dirichlet(alpha)
        # 从 Dirichlet 分布中采样出大小为 (num_samples,) 的样本集 x
        x = dist.sample((num_samples,))
        # 计算样本 x 的对数概率密度值
        actual_log_prob = dist.log_prob(x)
        # 遍历每个样本，计算预期的对数概率密度值
        for i in range(num_samples):
            expected_log_prob = scipy.stats.dirichlet.logpdf(
                x[i].numpy(), alpha.numpy()
            )
            # 断言实际对数概率密度与预期对数概率密度在一定的误差范围内相等
            self.assertEqual(actual_log_prob[i], expected_log_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_dirichlet_log_prob_zero(self):
        # 测试特殊情况，当 x=0 且 alpha=1 时的对数概率密度值
        # PDF 正比于 x**(alpha-1)，在此情况下应为 0**0=1
        # 但是，如果没有考虑 alpha-1 的值，很容易引入 NaN
        alpha = torch.tensor([1, 2])
        dist = Dirichlet(alpha)
        x = torch.tensor([0, 1])
        # 计算实际的对数概率密度值
        actual_log_prob = dist.log_prob(x)
        # 计算预期的对数概率密度值
        expected_log_prob = scipy.stats.dirichlet.logpdf(x.numpy(), alpha.numpy())
        # 断言实际对数概率密度与预期对数概率密度在一定的误差范围内相等
        self.assertEqual(actual_log_prob, expected_log_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_dirichlet_sample(self):
        # 设定随机数种子，见注释 [Randomized statistical tests]
        set_rng_seed(0)
        # 生成随机的 Dirichlet 分布参数 alpha，并取指数
        alpha = torch.exp(torch.randn(3))
        # 使用 _check_sampler_sampler 方法检查采样器的采样结果
        self._check_sampler_sampler(
            Dirichlet(alpha),
            scipy.stats.dirichlet(alpha.numpy()),
            f"Dirichlet(alpha={list(alpha)})",
            multivariate=True,
        )

    def test_dirichlet_mode(self):
        # 测试 Dirichlet 分布的众数，涵盖了一些边缘情况和 beta 分布
        concentrations_and_modes = [
            ([2, 2, 1], [0.5, 0.5, 0.0]),
            ([3, 2, 1], [2 / 3, 1 / 3, 0]),
            ([0.5, 0.2, 0.2], [1.0, 0.0, 0.0]),
            ([1, 1, 1], [nan, nan, nan]),
        ]
        # 遍历每个浓度和其对应的众数
        for concentration, mode in concentrations_and_modes:
            # 创建 Dirichlet 分布对象
            dist = Dirichlet(torch.tensor(concentration))
            # 断言分布的众数与预期众数相等
            self.assertEqual(dist.mode, torch.tensor(mode))

    def test_beta_shape(self):
        # 生成随机的参数 con1, con0，并取指数，同时需要梯度
        con1 = torch.randn(2, 3).exp().requires_grad_()
        con0 = torch.randn(2, 3).exp().requires_grad_()
        con1_1d = torch.randn(4).exp().requires_grad_()
        con0_1d = torch.randn(4).exp().requires_grad_()
        # 断言 Beta 分布采样结果的形状与预期相等
        self.assertEqual(Beta(con1, con0).sample().size(), (2, 3))
        self.assertEqual(Beta(con1, con0).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Beta(con1_1d, con0_1d).sample().size(), (4,))
        self.assertEqual(Beta(con1_1d, con0_1d).sample((1,)).size(), (1, 4))
        # 断言特定参数下的 Beta 分布采样结果形状与预期相等
        self.assertEqual(Beta(0.1, 0.3).sample().size(), ())
        self.assertEqual(Beta(0.1, 0.3).sample((5,)).size(), (5,))
    # 定义测试函数 test_beta_log_prob，用于测试 Beta 分布的对数概率计算
    def test_beta_log_prob(self):
        # 执行100次测试
        for _ in range(100):
            # 从标准正态分布中随机生成 con1 和 con0
            con1 = np.exp(np.random.normal())
            con0 = np.exp(np.random.normal())
            # 创建 Beta 分布对象
            dist = Beta(con1, con0)
            # 从 Beta 分布中采样一个值 x
            x = dist.sample()
            # 计算采样值 x 的实际对数概率
            actual_log_prob = dist.log_prob(x).sum()
            # 使用 scipy 计算 Beta 分布在 x 处的对数概率作为期望值
            expected_log_prob = scipy.stats.beta.logpdf(x, con1, con0)
            # 断言实际对数概率与期望值接近
            self.assertEqual(
                float(actual_log_prob), float(expected_log_prob), atol=1e-3, rtol=0
            )

    # 装饰器，当未安装 NumPy 时跳过此测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 设置默认的 PyTorch 浮点数类型为双精度
    @set_default_dtype(torch.double)
    # 定义测试函数 test_beta_sample，用于测试 Beta 分布的采样
    def test_beta_sample(self):
        set_rng_seed(1)  # 设置随机数种子，参见注释 [Randomized statistical tests]
        # 遍历不同的 con1 和 con0 组合
        for con1, con0 in product([0.1, 1.0, 10.0], [0.1, 1.0, 10.0]):
            # 使用 _check_sampler_sampler 方法验证 Beta 分布的采样器
            self._check_sampler_sampler(
                Beta(con1, con0),
                scipy.stats.beta(con1, con0),
                f"Beta(alpha={con1}, beta={con0})",
            )
        # 检查当 alpha 很小时不会产生 NAN 的情况
        for Tensor in [torch.FloatTensor, torch.DoubleTensor]:
            x = Beta(Tensor([1e-6]), Tensor([1e-6])).sample()[0]
            # 断言采样值 x 是有限的正数
            self.assertTrue(np.isfinite(x) and x > 0, f"Invalid Beta.sample(): {x}")

    # 定义测试函数 test_beta_underflow，用于测试低参数值 (alpha, beta) 下 Beta 分布的样本分布情况
    def test_beta_underflow(self):
        # 对于低值的 (alpha, beta)，Gamma 分布的样本可能会下溢
        # 在 float32 下可能导致虚假模式出现在 0.5 处。为避免这种情况，
        # torch._sample_dirichlet 使用双精度进行中间计算。
        set_rng_seed(1)
        num_samples = 50000
        # 遍历不同的数据类型（float 和 double）
        for dtype in [torch.float, torch.double]:
            # 使用给定的 dtype 创建 tensor conc
            conc = torch.tensor(1e-2, dtype=dtype)
            # 从 Beta 分布中采样 num_samples 个样本
            beta_samples = Beta(conc, conc).sample([num_samples])
            # 断言没有样本值为 0
            self.assertEqual((beta_samples == 0).sum(), 0)
            # 断言没有样本值为 1
            self.assertEqual((beta_samples == 1).sum(), 0)
            # 断言支持集中在 0 和 1 附近
            frac_zeros = float((beta_samples < 0.1).sum()) / num_samples
            frac_ones = float((beta_samples > 0.9).sum()) / num_samples
            # 断言 frac_zeros 约为 0.5，允许误差为 0.05
            self.assertEqual(frac_zeros, 0.5, atol=0.05, rtol=0)
            # 断言 frac_ones 约为 0.5，允许误差为 0.05
            self.assertEqual(frac_ones, 0.5, atol=0.05, rtol=0)

    # 装饰器，当未安装 CUDA 时跳过此测试
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    # 定义测试函数 test_beta_underflow_gpu，用于在 GPU 上测试低参数值 (alpha, beta) 下 Beta 分布的样本分布情况
    def test_beta_underflow_gpu(self):
        set_rng_seed(1)
        num_samples = 50000
        # 使用双精度在 CUDA 上创建 tensor conc
        conc = torch.tensor(1e-2, dtype=torch.float64).cuda()
        # 从 Beta 分布中在 GPU 上采样 num_samples 个样本
        beta_samples = Beta(conc, conc).sample([num_samples])
        # 断言没有样本值为 0
        self.assertEqual((beta_samples == 0).sum(), 0)
        # 断言没有样本值为 1
        self.assertEqual((beta_samples == 1).sum(), 0)
        # 断言支持集中在 0 和 1 附近
        frac_zeros = float((beta_samples < 0.1).sum()) / num_samples
        frac_ones = float((beta_samples > 0.9).sum()) / num_samples
        # TODO: 一旦 GPU 上的不平衡问题修复，增加精度。
        # 断言 frac_zeros 约为 0.5，允许较大误差为 0.12
        self.assertEqual(frac_zeros, 0.5, atol=0.12, rtol=0)
        # 断言 frac_ones 约为 0.5，允许较大误差为 0.12
        self.assertEqual(frac_ones, 0.5, atol=0.12, rtol=0)

    # 设置默认的 PyTorch 浮点数类型为双精度
    @set_default_dtype(torch.double)
    # 定义测试函数 test_continuous_bernoulli，测试 ContinuousBernoulli 分布类的功能
    def test_continuous_bernoulli(self):
        # 创建概率张量 p，要求其梯度
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        # 创建张量 r，要求其梯度
        r = torch.tensor(0.3, requires_grad=True)
        # 创建标量 s
        s = 0.3
        
        # 测试从 ContinuousBernoulli 分布中抽样，检查结果尺寸是否为 (8, 3)
        self.assertEqual(ContinuousBernoulli(p).sample((8,)).size(), (8, 3))
        # 测试从 ContinuousBernoulli 分布中抽样，检查抽样结果是否不要求梯度
        self.assertFalse(ContinuousBernoulli(p).sample().requires_grad)
        # 测试从 ContinuousBernoulli 分布中抽样，检查结果尺寸是否为 (8,)
        self.assertEqual(ContinuousBernoulli(r).sample((8,)).size(), (8,))
        # 测试从 ContinuousBernoulli 分布中抽样，检查结果尺寸是否为 ()
        self.assertEqual(ContinuousBernoulli(r).sample().size(), ())
        # 测试从 ContinuousBernoulli 分布中抽样，检查结果尺寸是否为 (3, 2)
        self.assertEqual(
            ContinuousBernoulli(r).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        # 测试从 ContinuousBernoulli 分布中抽样，检查结果尺寸是否为 ()
        self.assertEqual(ContinuousBernoulli(s).sample().size(), ())
        # 使用 _gradcheck_log_prob 函数检查 ContinuousBernoulli 的 log_prob 方法
        self._gradcheck_log_prob(ContinuousBernoulli, (p,))
        
        # 定义 ref_log_prob 函数，用于检查对数概率的参考实现
        def ref_log_prob(idx, val, log_prob):
            prob = p[idx]
            if prob > 0.499 and prob < 0.501:  # 使用默认的 lim 值
                # 计算对数规范化常数
                log_norm_const = (
                    math.log(2.0)
                    + 4.0 / 3.0 * math.pow(prob - 0.5, 2)
                    + 104.0 / 45.0 * math.pow(prob - 0.5, 4)
                )
            else:
                log_norm_const = math.log(
                    2.0 * math.atanh(1.0 - 2.0 * prob) / (1.0 - 2.0 * prob)
                )
            # 计算结果
            res = (
                val * math.log(prob) + (1.0 - val) * math.log1p(-prob) + log_norm_const
            )
            # 断言 log_prob 是否等于 res
            self.assertEqual(log_prob, res)
        
        # 使用 _check_log_prob 函数检查 ContinuousBernoulli 的 log_prob 方法
        self._check_log_prob(ContinuousBernoulli(p), ref_log_prob)
        # 使用 _check_log_prob 函数检查 ContinuousBernoulli 的 log_prob 方法，
        # 传入 logits=p.log() - (-p).log1p() 作为参数
        self._check_log_prob(
            ContinuousBernoulli(logits=p.log() - (-p).log1p()), ref_log_prob
        )
        
        # 检查熵的计算结果是否符合预期
        self.assertEqual(
            ContinuousBernoulli(p).entropy(),
            torch.tensor([-0.02938, -0.07641, -0.00682]),
            atol=1e-4,
            rtol=0,
        )
        # 当使用 float 64 时，熵的值对应到 prob 的固定值下
        # 当使用 float 32 时，熵的值应为 -1.76898
        self.assertEqual(
            ContinuousBernoulli(torch.tensor([0.0])).entropy(),
            torch.tensor([-2.58473]),
            atol=1e-5,
            rtol=0,
        )
        # 检查熵的计算结果是否符合预期
        self.assertEqual(
            ContinuousBernoulli(s).entropy(), torch.tensor(-0.02938), atol=1e-4, rtol=0
        )

    # 定义测试函数 test_continuous_bernoulli_3d，测试 ContinuousBernoulli 分布类在三维情况下的功能
    def test_continuous_bernoulli_3d(self):
        # 创建形状为 (2, 3, 5) 的全 0.5 张量，并要求其梯度
        p = torch.full((2, 3, 5), 0.5).requires_grad_()
        # 检查从 ContinuousBernoulli 分布中抽样的结果尺寸是否为 (2, 3, 5)
        self.assertEqual(ContinuousBernoulli(p).sample().size(), (2, 3, 5))
        # 检查从 ContinuousBernoulli 分布中抽样的结果尺寸是否为 (2, 5, 2, 3, 5)
        self.assertEqual(
            ContinuousBernoulli(p).sample(sample_shape=(2, 5)).size(), (2, 5, 2, 3, 5)
        )
        # 检查从 ContinuousBernoulli 分布中抽样的结果尺寸是否为 (2, 2, 3, 5)
        self.assertEqual(ContinuousBernoulli(p).sample((2,)).size(), (2, 2, 3, 5))
    # 定义一个测试函数 test_lkj_cholesky_log_prob，用于测试 LKJ 分布的对数概率计算
    def test_lkj_cholesky_log_prob(self):
        # 定义一个内部函数 tril_cholesky_to_tril_corr，将 Cholesky 分解的下三角矩阵转换为相关系数矩阵的下三角矩阵
        def tril_cholesky_to_tril_corr(x):
            # 将向量 x 转换为 Cholesky 分解的下三角矩阵
            x = vec_to_tril_matrix(x, -1)
            # 计算相关系数矩阵的对角线元素，并填充到原先的矩阵中
            diag = (1 - (x * x).sum(-1)).sqrt().diag_embed()
            x = x + diag
            # 将相关系数矩阵的下三角矩阵转换为向量形式
            return tril_matrix_to_vec(x @ x.T, -1)

        # 对维度从 2 到 4 的范围进行循环
        for dim in range(2, 5):
            # 初始化一个空列表，用于存储对数概率值
            log_probs = []
            # 创建一个 LKJCholesky 分布对象 lkj，指定维度 dim、浓度参数为 1.0，并启用参数验证
            lkj = LKJCholesky(dim, concentration=1.0, validate_args=True)
            # 进行两次采样和概率计算
            for i in range(2):
                # 从 LKJ 分布中采样一个样本
                sample = lkj.sample()
                # 将 Cholesky 分解的下三角矩阵转换为向量形式
                sample_tril = tril_matrix_to_vec(sample, diag=-1)
                # 计算样本的对数概率
                log_prob = lkj.log_prob(sample)
                # 计算 Cholesky 分解转换后的相关系数矩阵的 Jacobian 行列式的对数绝对值
                log_abs_det_jacobian = torch.slogdet(
                    jacobian(tril_cholesky_to_tril_corr, sample_tril)
                ).logabsdet
                # 将对数概率值减去 Jacobian 调整因子的对数绝对值，并添加到列表中
                log_probs.append(log_prob - log_abs_det_jacobian)
            
            # 当维度为 2 时，验证对数概率的期望值为 0.5
            if dim == 2:
                self.assertTrue(
                    # 检查是否所有的对数概率值在数值上接近 0.5 的对数值，容差为 1e-10
                    all(
                        torch.allclose(x, torch.tensor(0.5).log(), atol=1e-10)
                        for x in log_probs
                    )
                )
            
            # 检查两次采样的对数概率值是否相等
            self.assertEqual(log_probs[0], log_probs[1])
            
            # 创建一个无效的样本，包含一个超出边界的额外行，并断言应引发 ValueError 异常
            invalid_sample = torch.cat([sample, sample.new_ones(1, dim)], dim=0)
            self.assertRaises(ValueError, lambda: lkj.log_prob(invalid_sample))
    # 定义一个测试方法，用于验证独立分布的形状
    def test_independent_shape(self):
        # 对于每一个分布类和其参数组合，获取示例
        for Dist, params in _get_examples():
            # 遍历每个参数组合
            for param in params:
                # 使用参数创建基础分布对象
                base_dist = Dist(**param)
                # 从基础分布中抽取样本
                x = base_dist.sample()
                # 计算基础分布对数概率的形状
                base_log_prob_shape = base_dist.log_prob(x).shape
                # 遍历重新解释的批次维度范围
                for reinterpreted_batch_ndims in range(len(base_dist.batch_shape) + 1):
                    # 创建独立分布对象
                    indep_dist = Independent(base_dist, reinterpreted_batch_ndims)
                    # 计算独立分布对数概率的形状，去除重新解释的批次维度
                    indep_log_prob_shape = base_log_prob_shape[
                        : len(base_log_prob_shape) - reinterpreted_batch_ndims
                    ]
                    # 验证独立分布对数概率形状是否符合预期
                    self.assertEqual(indep_dist.log_prob(x).shape, indep_log_prob_shape)
                    # 验证独立分布样本的形状是否与基础分布样本相同
                    self.assertEqual(
                        indep_dist.sample().shape, base_dist.sample().shape
                    )
                    # 验证独立分布是否具有 rsample 方法，与基础分布一致
                    self.assertEqual(indep_dist.has_rsample, base_dist.has_rsample)
                    # 如果独立分布具有 rsample 方法，验证其样本形状是否与基础分布相同
                    if indep_dist.has_rsample:
                        self.assertEqual(
                            indep_dist.sample().shape, base_dist.sample().shape
                        )
                    # 尝试验证独立分布枚举支持的形状是否与基础分布相同
                    try:
                        self.assertEqual(
                            indep_dist.enumerate_support().shape,
                            base_dist.enumerate_support().shape,
                        )
                        # 验证独立分布均值的形状是否与基础分布相同
                        self.assertEqual(indep_dist.mean.shape, base_dist.mean.shape)
                    except NotImplementedError:
                        pass
                    # 尝试验证独立分布方差的形状是否与基础分布相同
                    try:
                        self.assertEqual(
                            indep_dist.variance.shape, base_dist.variance.shape
                        )
                    except NotImplementedError:
                        pass
                    # 尝试验证独立分布熵的形状是否与预期的对数概率形状一致
                    try:
                        self.assertEqual(
                            indep_dist.entropy().shape, indep_log_prob_shape
                        )
                    except NotImplementedError:
                        pass
    # 测试独立分布的扩展功能
    def test_independent_expand(self):
        # 遍历所有示例分布和参数
        for Dist, params in _get_examples():
            # 遍历每个参数
            for param in params:
                # 根据参数创建基础分布对象
                base_dist = Dist(**param)
                # 遍历重新解释的批次维度范围
                for reinterpreted_batch_ndims in range(len(base_dist.batch_shape) + 1):
                    # 遍历不同的尺寸情况
                    for s in [torch.Size(), torch.Size((2,)), torch.Size((2, 3))]:
                        # 创建独立分布对象
                        indep_dist = Independent(base_dist, reinterpreted_batch_ndims)
                        # 计算扩展后的形状
                        expanded_shape = s + indep_dist.batch_shape
                        # 扩展独立分布
                        expanded = indep_dist.expand(expanded_shape)
                        # 对扩展后的分布进行采样
                        expanded_sample = expanded.sample()
                        # 计算期望的形状
                        expected_shape = expanded_shape + indep_dist.event_shape
                        # 断言扩展样本的形状与期望形状相同
                        self.assertEqual(expanded_sample.shape, expected_shape)
                        # 断言扩展后的对数概率等于独立分布的对数概率
                        self.assertEqual(
                            expanded.log_prob(expanded_sample),
                            indep_dist.log_prob(expanded_sample),
                        )
                        # 断言扩展后的事件形状与独立分布的事件形状相同
                        self.assertEqual(expanded.event_shape, indep_dist.event_shape)
                        # 断言扩展后的批次形状与期望的扩展形状相同
                        self.assertEqual(expanded.batch_shape, expanded_shape)

    # 设置默认的数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    def test_cdf_icdf_inverse(self):
        # 测试分布的反函数属性
        for Dist, params in _get_examples():
            # 遍历每个分布示例和参数
            for i, param in enumerate(params):
                # 根据参数创建分布对象
                dist = Dist(**param)
                # 从分布中采样一些样本
                samples = dist.sample(sample_shape=(20,))
                try:
                    # 计算样本的累积分布函数值
                    cdf = dist.cdf(samples)
                    # 计算分布的反累积分布函数值
                    actual = dist.icdf(cdf)
                except NotImplementedError:
                    continue
                # 计算相对误差
                rel_error = torch.abs(actual - samples) / (1e-10 + torch.abs(samples))
                # 断言相对误差的最大值小于给定阈值
                self.assertLess(
                    rel_error.max(),
                    1e-4,
                    msg="\n".join(
                        [
                            f"{Dist.__name__} example {i + 1}/{len(params)}, icdf(cdf(x)) != x",
                            f"x = {samples}",
                            f"cdf(x) = {cdf}",
                            f"icdf(cdf(x)) = {actual}",
                        ]
                    ),
                )

    # 如果没有安装 NumPy，则跳过此测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma_log_prob_at_boundary(self):
        # 测试 Gamma 分布在边界处的对数概率
        for concentration, log_prob in [(0.5, inf), (1, 0), (2, -inf)]:
            # 创建 Gamma 分布对象
            dist = Gamma(concentration, 1)
            # 使用 SciPy 创建 Gamma 分布对象
            scipy_dist = scipy.stats.gamma(concentration)
            # 断言 Gamma 分布在 0 处的对数概率与预期值接近
            self.assertAlmostEqual(dist.log_prob(0), log_prob)
            # 断言 Gamma 分布在 0 处的对数概率与 SciPy 中的值接近
            self.assertAlmostEqual(dist.log_prob(0), scipy_dist.logpdf(0))

    # 设置默认的数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    # 定义一个测试函数，用于测试累积分布函数的导数是否等于概率密度函数在给定值处的值
    def test_cdf_log_prob(self):
        # 使用_get_examples()函数获取分布类型和参数的示例
        for Dist, params in _get_examples():
            # 遍历每个参数示例
            for i, param in enumerate(params):
                # 对参数进行处理，如果是张量则去除梯度信息，例如 gamma 分布的形状参数
                param = {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in param.items()
                }
                # 根据参数创建分布对象
                dist = Dist(**param)
                # 从分布中抽取样本
                samples = dist.sample()
                # 如果分布不是离散分布，则需要保留样本的梯度信息
                if not dist.support.is_discrete:
                    samples.requires_grad_()
                try:
                    # 计算样本的累积分布函数（CDF）和对数概率密度函数（PDF）
                    cdfs = dist.cdf(samples)
                    pdfs = dist.log_prob(samples).exp()
                except NotImplementedError:
                    # 如果计算函数未实现则继续下一个参数示例的测试
                    continue
                # 计算累积分布函数的导数
                cdfs_derivative = grad(cdfs.sum(), [samples])[0]  # 这里不应该使用 torch.abs() 包裹
                # 断言累积分布函数的导数等于概率密度函数的值
                self.assertEqual(
                    cdfs_derivative,
                    pdfs,
                    msg="\n".join(
                        [
                            f"{Dist.__name__} example {i + 1}/{len(params)}, d(cdf)/dx != pdf(x)",
                            f"x = {samples}",
                            f"cdf = {cdfs}",
                            f"pdf = {pdfs}",
                            f"grad(cdf) = {cdfs_derivative}",
                        ]
                    ),
                )
    def test_invalid_parameter_broadcasting(self):
        # 定义测试方法，用于测试无效的参数广播情况，预期会抛出错误
        # 示例类型 (分布类, 分布参数)
        invalid_examples = [
            (
                Normal,
                {"loc": torch.tensor([[0, 0]]), "scale": torch.tensor([1, 1, 1, 1])},
            ),
            (
                Normal,
                {
                    "loc": torch.tensor([[[0, 0, 0], [0, 0, 0]]]),
                    "scale": torch.tensor([1, 1]),
                },
            ),
            (
                FisherSnedecor,
                {
                    "df1": torch.tensor([1, 1]),
                    "df2": torch.tensor([1, 1, 1]),
                },
            ),
            (
                Gumbel,
                {"loc": torch.tensor([[0, 0]]), "scale": torch.tensor([1, 1, 1, 1])},
            ),
            (
                Gumbel,
                {
                    "loc": torch.tensor([[[0, 0, 0], [0, 0, 0]]]),
                    "scale": torch.tensor([1, 1]),
                },
            ),
            (
                Gamma,
                {
                    "concentration": torch.tensor([0, 0]),
                    "rate": torch.tensor([1, 1, 1]),
                },
            ),
            (
                Kumaraswamy,
                {
                    "concentration1": torch.tensor([[1, 1]]),
                    "concentration0": torch.tensor([1, 1, 1, 1]),
                },
            ),
            (
                Kumaraswamy,
                {
                    "concentration1": torch.tensor([[[1, 1, 1], [1, 1, 1]]]),
                    "concentration0": torch.tensor([1, 1]),
                },
            ),
            (Laplace, {"loc": torch.tensor([0, 0]), "scale": torch.tensor([1, 1, 1])}),
            (Pareto, {"scale": torch.tensor([1, 1]), "alpha": torch.tensor([1, 1, 1])}),
            (
                StudentT,
                {
                    "df": torch.tensor([1.0, 1.0]),
                    "scale": torch.tensor([1.0, 1.0, 1.0]),
                },
            ),
            (
                StudentT,
                {"df": torch.tensor([1.0, 1.0]), "loc": torch.tensor([1.0, 1.0, 1.0])},
            ),
        ]

        # 遍历无效示例，对每个分布类及其参数执行断言，验证是否抛出 RuntimeError
        for dist, kwargs in invalid_examples:
            self.assertRaises(RuntimeError, dist, **kwargs)
    # 测试离散分布的模式，确保在模式点左右两侧的对数概率比模式点更小
    def _test_discrete_distribution_mode(self, dist, sanitized_mode, batch_isfinite):
        # 循环遍历左右两个步长
        for step in [-1, 1]:
            # 计算模式点的对数概率
            log_prob_mode = dist.log_prob(sanitized_mode)
            # 如果是 OneHotCategorical 分布，计算其下一个可能模式点的独热编码
            if isinstance(dist, OneHotCategorical):
                idx = (dist._categorical.mode + 1) % dist.probs.shape[-1]
                other = torch.nn.functional.one_hot(
                    idx, num_classes=dist.probs.shape[-1]
                ).to(dist.mode)
            else:
                other = dist.mode + step
            # 创建掩码，检查支持集合内其他可能的模式点
            mask = batch_isfinite & dist.support.check(other)
            # 断言：掩码中任意值为真，或者模式点唯一且数量为1
            self.assertTrue(mask.any() or dist.mode.unique().numel() == 1)
            # 如果事件形状不是标量（如 OneHotCategorical），在右侧添加一个维度
            other = torch.where(
                mask[..., None] if mask.ndim < other.ndim else mask,
                other,
                dist.sample(),
            )
            # 计算其他模式点的对数概率
            log_prob_other = dist.log_prob(other)
            # 计算对数概率的差值
            delta = log_prob_mode - log_prob_other
            # 断言：允许最多1e-12的舍入误差
            self.assertTrue(
                (-1e-12 < delta[mask].detach()).all()
            )  # 允许最多1e-12的舍入误差。

    # 测试连续分布的模式，期望在未约束空间中略微扰动模式点，使对数概率降低
    def _test_continuous_distribution_mode(self, dist, sanitized_mode, batch_isfinite):
        # 定义扰动点的数量
        num_points = 10
        # 转换到分布支持的空间
        transform = transform_to(dist.support)
        # 将模式点转换到未约束空间
        unconstrained_mode = transform.inv(sanitized_mode)
        # 生成扰动，稍微在未约束空间中扰动模式点
        perturbation = 1e-5 * (
            torch.rand((num_points,) + unconstrained_mode.shape) - 0.5
        )
        perturbed_mode = transform(perturbation + unconstrained_mode)
        # 计算原始模式点和扰动后模式点的对数概率
        log_prob_mode = dist.log_prob(sanitized_mode)
        log_prob_other = dist.log_prob(perturbed_mode)
        # 计算对数概率的差值
        delta = log_prob_mode - log_prob_other

        # 如果两个对数概率都是无穷且符号相同，则将差值设置为零
        both_infinite_with_same_sign = (log_prob_mode == log_prob_other) & (
            log_prob_mode.abs() == inf
        )
        delta[both_infinite_with_same_sign] = 0.0
        # 检查批次中所有元素的顺序，允许最多-1e-12的差值
        ordering = (delta > -1e-12).all(axis=0)
        self.assertTrue(ordering[batch_isfinite].all())

    # 设置默认的 torch 数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    def test_mode(self):
        # 定义离散分布类和没有模式的分布类
        discrete_distributions = (
            Bernoulli,  # 伯努利分布
            Binomial,  # 二项分布
            Categorical,  # 分类分布
            Geometric,  # 几何分布
            NegativeBinomial,  # 负二项分布
            OneHotCategorical,  # 单热分类分布
            Poisson,  # 泊松分布
        )
        no_mode_available = (
            ContinuousBernoulli,  # 连续伯努利分布
            LKJCholesky,  # LKJ分布
            LogisticNormal,  # 逻辑正态分布
            MixtureSameFamily,  # 相同家族混合分布
            Multinomial,  # 多项分布
            RelaxedBernoulli,  # 松弛伯努利分布
            RelaxedOneHotCategorical,  # 松弛单热分类分布
        )

        # 遍历获取的分布类和参数组合
        for dist_cls, params in _get_examples():
            for param in params:
                # 使用参数创建分布对象
                dist = dist_cls(**param)
                # 如果是没有定义模式的分布类或者是TransformedDistribution类型，则预期抛出NotImplementedError异常
                if (
                    isinstance(dist, no_mode_available)
                    or type(dist) is TransformedDistribution
                ):
                    with self.assertRaises(NotImplementedError):
                        dist.mode
                    continue

                # 检查事件形状中所有元素或无元素是否都为NaN：部分事件无法定义模式
                isfinite = dist.mode.isfinite().reshape(
                    dist.batch_shape + (dist.event_shape.numel(),)
                )
                batch_isfinite = isfinite.all(axis=-1)
                self.assertTrue((batch_isfinite | ~isfinite.any(axis=-1)).all())

                # 通过从分布中采样来消除未定义的模式
                sanitized_mode = torch.where(
                    ~dist.mode.isnan(), dist.mode, dist.sample()
                )
                # 如果是离散分布，则测试离散分布的模式
                if isinstance(dist, discrete_distributions):
                    self._test_discrete_distribution_mode(
                        dist, sanitized_mode, batch_isfinite
                    )
                else:
                    # 否则，测试连续分布的模式
                    self._test_continuous_distribution_mode(
                        dist, sanitized_mode, batch_isfinite
                    )

                # 确保模式的对数概率不包含NaN值
                self.assertFalse(dist.log_prob(sanitized_mode).isnan().any())
# 这些测试仅适用于实现自定义重参数化梯度的少数分布。大多数 .rsample() 实现仅依赖于重参数化技巧，不需要进行准确性测试。

@skipIfTorchDynamo("Not a TorchDynamo suitable test")
# 使用 @skipIfTorchDynamo 装饰器跳过不适合 TorchDynamo 的测试

class TestRsample(DistributionsTestCase):
    # 测试类 TestRsample，继承自 DistributionsTestCase

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 如果未找到 NumPy，则跳过测试

    def test_gamma(self):
        # Gamma 分布的测试方法

        num_samples = 100
        # 生成 100 个样本

        for alpha in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            # 遍历不同的 alpha 值

            alphas = torch.tensor(
                [alpha] * num_samples, dtype=torch.float, requires_grad=True
            )
            # 创建包含 num_samples 个 alpha 值的张量，用于计算梯度，类型为浮点数

            betas = alphas.new_ones(num_samples)
            # 创建一个与 alphas 具有相同数据类型的张量，值均为 1

            x = Gamma(alphas, betas).rsample()
            # 从 Gamma 分布中采样，得到样本 x

            x.sum().backward()
            # 对样本 x 的总和进行反向传播

            x, ind = x.sort()
            # 对样本 x 进行排序，并返回排序后的结果及其索引

            x = x.detach().numpy()
            # 将张量 x 转换为 NumPy 数组

            actual_grad = alphas.grad[ind].numpy()
            # 获取 alpha 的梯度，并转换为 NumPy 数组

            # 比较与期望梯度 dx/dalpha 在常数 cdf(x,alpha) 处的梯度
            cdf = scipy.stats.gamma.cdf
            # 获取 Gamma 分布的累积分布函数
            pdf = scipy.stats.gamma.pdf
            # 获取 Gamma 分布的概率密度函数

            eps = 0.01 * alpha / (1.0 + alpha**0.5)
            # 计算 epsilon，用于数值梯度计算

            cdf_alpha = (cdf(x, alpha + eps) - cdf(x, alpha - eps)) / (2 * eps)
            # 计算在 alpha 处的数值累积分布函数

            cdf_x = pdf(x, alpha)
            # 计算在样本 x 处的概率密度函数值

            expected_grad = -cdf_alpha / cdf_x
            # 计算期望的梯度值

            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            # 计算相对误差

            self.assertLess(
                np.max(rel_error),
                0.0005,
                "\n".join(
                    [
                        f"Bad gradient dx/alpha for x ~ Gamma({alpha}, 1)",
                        f"x {x}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                        f"at alpha={alpha}, x={x[rel_error.argmax()]}",
                    ]
                ),
            )
            # 使用断言验证梯度是否符合预期，并在不满足条件时输出详细信息

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 如果未找到 NumPy，则跳过测试
    # 定义单元测试方法 test_chi2，用于测试 Chi-squared 分布的梯度计算
    def test_chi2(self):
        # 设定样本数目
        num_samples = 100
        # 遍历不同的自由度 df 值
        for df in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            # 创建包含相同 df 值的张量 dfs，要求梯度计算
            dfs = torch.tensor(
                [df] * num_samples, dtype=torch.float, requires_grad=True
            )
            # 从 Chi2 分布中采样生成 x
            x = Chi2(dfs).rsample()
            # 对 x 求和并反向传播梯度
            x.sum().backward()
            # 对 x 进行排序，并保留排序索引
            x, ind = x.sort()
            # 分离出 x 的数据并转换为 NumPy 数组
            x = x.detach().numpy()
            # 获取实际梯度值
            actual_grad = dfs.grad[ind].numpy()
            # 使用 scipy 计算 Chi-squared 分布的累积分布函数和概率密度函数
            cdf = scipy.stats.chi2.cdf
            pdf = scipy.stats.chi2.pdf
            # 计算用于数值梯度近似的小增量
            eps = 0.01 * df / (1.0 + df**0.5)
            # 计算累积分布函数的数值梯度
            cdf_df = (cdf(x, df + eps) - cdf(x, df - eps)) / (2 * eps)
            # 计算概率密度函数值
            cdf_x = pdf(x, df)
            # 计算期望的梯度值
            expected_grad = -cdf_df / cdf_x
            # 计算相对误差
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            # 使用单元测试断言检查相对误差是否小于预设阈值
            self.assertLess(
                np.max(rel_error),
                0.001,
                "\n".join(
                    [
                        f"Bad gradient dx/ddf for x ~ Chi2({df})",
                        f"x {x}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                    ]
                ),
            )

    # 如果未安装 NumPy，则跳过此测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义一个测试函数，用于测试 Dirichlet 分布在对角线上的行为
    def test_dirichlet_on_diagonal(self):
        # 设置每个参数组合下的采样次数
        num_samples = 20
        # 定义一个参数网格，包含三个不同的值
        grid = [1e-1, 1e0, 1e1]
        # 使用 product 函数遍历参数网格中所有的组合
        for a0, a1, a2 in product(grid, grid, grid):
            # 根据当前参数组合创建一个张量 alphas，每个值重复 num_samples 次，并标记需要计算梯度
            alphas = torch.tensor(
                [[a0, a1, a2]] * num_samples, dtype=torch.float, requires_grad=True
            )
            # 从 Dirichlet 分布中采样，然后取每个样本的第一个值
            x = Dirichlet(alphas).rsample()[:, 0]
            # 计算 x 的元素之和的梯度
            x.sum().backward()
            # 将 x 按升序排序，并获取排序后的索引
            x, ind = x.sort()
            # 将 x 转换为 numpy 数组，并且断开与计算图的连接
            x = x.detach().numpy()
            # 获取实际计算出的梯度值，按照排序后的索引获取对应的部分
            actual_grad = alphas.grad[ind].numpy()[:, 0]
            # 使用 Beta 分布的累积分布函数和概率密度函数
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            # 定义 alpha 和 beta 参数，用于计算误差范围
            alpha, beta = a0, a1 + a2
            # 计算用于估计梯度的小量 eps
            eps = 0.01 * alpha / (1.0 + np.sqrt(alpha))
            # 计算 cdf 在 alpha + eps 和 alpha - eps 处的差异，并除以 2 * eps
            cdf_alpha = (cdf(x, alpha + eps, beta) - cdf(x, alpha - eps, beta)) / (2 * eps)
            # 计算概率密度函数在 alpha 和 beta 处的值
            cdf_x = pdf(x, alpha, beta)
            # 计算期望的梯度值
            expected_grad = -cdf_alpha / cdf_x
            # 计算相对误差
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            # 断言相对误差的最大值小于 0.001，否则输出详细信息
            self.assertLess(
                np.max(rel_error),
                0.001,
                "\n".join(
                    [
                        f"Bad gradient dx[0]/dalpha[0] for Dirichlet([{a0}, {a1}, {a2}])",
                        f"x {x}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                        f"at x={x[rel_error.argmax()]}",
                    ]
                ),
            )

    # 如果没有找到 NumPy，跳过测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义测试函数 `test_beta_wrt_alpha`，用于测试 Beta 分布关于参数 con1 的梯度计算
    def test_beta_wrt_alpha(self):
        num_samples = 20  # 设置样本数量为 20
        grid = [1e-2, 1e-1, 1e0, 1e1, 1e2]  # 定义参数网格，用于测试不同的 con1 和 con0 组合
        # 对于 grid 中的每一对 con1, con0 组合，执行以下操作
        for con1, con0 in product(grid, grid):
            # 创建一个包含 num_samples 个 con1 值的张量，要求梯度计算
            con1s = torch.tensor(
                [con1] * num_samples, dtype=torch.float, requires_grad=True
            )
            # 创建一个与 con1s 相同设备的 con0s 张量，其中填充 num_samples 个 con0 值
            con0s = con1s.new_tensor([con0] * num_samples)
            # 从 Beta 分布 Beta(con1s, con0s) 中采样 x
            x = Beta(con1s, con0s).rsample()
            # 计算 x 各元素之和的梯度
            x.sum().backward()
            # 对 x 进行排序，并返回排序后的 x 和对应的索引 ind
            x, ind = x.sort()
            # 将 x 转换为 NumPy 数组，并断开与梯度图的连接
            x = x.detach().numpy()
            # 获取实际梯度值，即 con1s 对应的梯度值，并转换为 NumPy 数组
            actual_grad = con1s.grad[ind].numpy()
            # 定义 Beta 分布的累积分布函数 (CDF) 和概率密度函数 (PDF)
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            # 计算用于数值梯度的微小增量 eps
            eps = 0.01 * con1 / (1.0 + np.sqrt(con1))
            # 计算数值梯度的估计值 cdf_alpha
            cdf_alpha = (cdf(x, con1 + eps, con0) - cdf(x, con1 - eps, con0)) / (2 * eps)
            # 计算 x 处 Beta 分布的 PDF 值 cdf_x
            cdf_x = pdf(x, con1, con0)
            # 计算期望的梯度值，即数值梯度的比值
            expected_grad = -cdf_alpha / cdf_x
            # 计算相对误差 rel_error
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            # 断言相对误差的最大值小于 0.005，否则输出错误信息
            self.assertLess(
                np.max(rel_error),
                0.005,
                "\n".join(
                    [
                        f"Bad gradient dx/dcon1 for x ~ Beta({con1}, {con0})",
                        f"x {x}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                        f"at x = {x[rel_error.argmax()]}",
                    ]
                ),
            )

    # 如果未安装 NumPy，则跳过此测试
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # 定义一个测试方法，用于测试 Beta 分布的梯度计算
    def test_beta_wrt_beta(self):
        # 设置样本数
        num_samples = 20
        # 定义 Beta 分布的参数网格
        grid = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        # 遍历参数网格中的每一对参数组合
        for con1, con0 in product(grid, grid):
            # 创建一个包含 con0 的张量，要求梯度计算
            con0s = torch.tensor(
                [con0] * num_samples, dtype=torch.float, requires_grad=True
            )
            # 使用新值 con1 创建 con0s 的张量
            con1s = con0s.new_tensor([con1] * num_samples)
            # 从 Beta 分布中采样 x
            x = Beta(con1s, con0s).rsample()
            # 对 x 求和并反向传播梯度
            x.sum().backward()
            # 对 x 进行排序
            x, ind = x.sort()
            # 分离 x 的数值部分并转换为 numpy 数组
            x = x.detach().numpy()
            # 获取实际梯度
            actual_grad = con0s.grad[ind].numpy()
            # 比较期望梯度 dx/dcon0 与常数 cdf(x,con1,con0) 之间的梯度
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            eps = 0.01 * con0 / (1.0 + np.sqrt(con0))
            cdf_beta = (cdf(x, con1, con0 + eps) - cdf(x, con1, con0 - eps)) / (2 * eps)
            cdf_x = pdf(x, con1, con0)
            expected_grad = -cdf_beta / cdf_x
            # 计算相对误差
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            # 断言最大相对误差小于阈值，否则输出详细错误信息
            self.assertLess(
                np.max(rel_error),
                0.005,
                "\n".join(
                    [
                        f"Bad gradient dx/dcon0 for x ~ Beta({con1}, {con0})",
                        f"x {x}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                        f"at x = {x[rel_error.argmax()]!r}",
                    ]
                ),
            )

    # 定义一个测试方法，用于测试多元 Dirichlet 分布的梯度计算
    def test_dirichlet_multivariate(self):
        # 计算临界 alpha
        alpha_crit = 0.25 * (5.0**0.5 - 1.0)
        # 设置样本数
        num_samples = 100000
        # 遍历不同的 alpha 偏移量
        for shift in [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10]:
            # 计算当前 alpha
            alpha = alpha_crit + shift
            # 创建一个包含当前 alpha 的张量，要求梯度计算
            alpha = torch.tensor([alpha], dtype=torch.float, requires_grad=True)
            # 构造 alpha 向量
            alpha_vec = torch.cat([alpha, alpha, alpha.new([1])])
            # 从 Dirichlet 分布中采样 z
            z = Dirichlet(alpha_vec.expand(num_samples, 3)).rsample()
            # 计算 z 第三个分量的均值
            mean_z3 = 1.0 / (2.0 * alpha + 1.0)
            # 计算损失函数
            loss = torch.pow(z[:, 2] - mean_z3, 2.0).mean()
            # 计算实际梯度
            actual_grad = grad(loss, [alpha])[0]
            # 手动计算期望梯度
            num = 1.0 - 2.0 * alpha - 4.0 * alpha**2
            den = (1.0 + alpha) ** 2 * (1.0 + 2.0 * alpha) ** 3
            expected_grad = num / den
            # 断言实际梯度与期望梯度的接近程度
            self.assertEqual(
                actual_grad,
                expected_grad,
                atol=0.002,
                rtol=0,
                msg="\n".join(
                    [
                        "alpha = alpha_c + %.2g" % shift,  # noqa: UP031
                        "expected_grad: %.5g" % expected_grad,  # noqa: UP031
                        "actual_grad: %.5g" % actual_grad,  # noqa: UP031
                        "error = %.2g"  # noqa: UP031
                        % torch.abs(expected_grad - actual_grad).max(),  # noqa: UP031
                    ]
                ),
            )
    # 设置默认数据类型为双精度浮点数（torch.double）
    @set_default_dtype(torch.double)
    # 定义一个测试函数 test_dirichlet_tangent_field，该函数用于测试狄利克雷分布的切向场
    def test_dirichlet_tangent_field(self):
        # 定义采样数目
        num_samples = 20
        # 狄利克雷分布的参数组合列表
        alpha_grid = [0.5, 1.0, 2.0]

        # 定义计算切向场的函数 compute_v，其中 x 是采样值，alpha 是狄利克雷分布的参数
        def compute_v(x, alpha):
            # 构建一个张量，每个分量是在对应方向的狄利克雷分布关于 alpha[i] 的梯度
            return torch.stack(
                [
                    # 调用 _Dirichlet_backward 函数，计算关于 alpha[i] 的梯度
                    _Dirichlet_backward(x, alpha, torch.eye(3, 3)[i].expand_as(x))[:, 0]
                    for i in range(3)
                ],
                dim=-1,
            )

        # 对狄利克雷分布参数的各种组合进行排列组合
        for a1, a2, a3 in product(alpha_grid, alpha_grid, alpha_grid):
            # 创建包含重复 alpha 的张量，并要求计算梯度
            alpha = torch.tensor([a1, a2, a3], requires_grad=True).expand(
                num_samples, 3
            )
            # 从狄利克雷分布中采样
            x = Dirichlet(alpha).rsample()
            # 计算关于 alpha 的对数概率的梯度，并保留计算图以供后续使用
            dlogp_da = grad(
                [Dirichlet(alpha).log_prob(x.detach()).sum()],
                [alpha],
                retain_graph=True,
            )[0][:, 0]
            # 计算关于 x 的对数概率的梯度，并保留计算图以供后续使用
            dlogp_dx = grad(
                [Dirichlet(alpha.detach()).log_prob(x).sum()], [x], retain_graph=True
            )[0]
            # 计算切向场 v 的各个分量
            v = torch.stack(
                [
                    # 计算 x[:, i] 关于 alpha 的梯度
                    grad([x[:, i].sum()], [alpha], retain_graph=True)[0][:, 0]
                    for i in range(3)
                ],
                dim=-1,
            )
            # 使用有限差分法计算其余属性
            # 定义一个与单纯形正交的任意基向量 dx
            dx = torch.tensor([[2.0, -1.0, -1.0], [0.0, 1.0, -1.0]])
            dx /= dx.norm(2, -1, True)
            # 避免处于边界处，设置一个小的扰动 eps
            eps = 1e-2 * x.min(-1, True)[0]
            # 计算 dv0 和 dv1
            dv0 = (
                compute_v(x + eps * dx[0], alpha) - compute_v(x - eps * dx[0], alpha)
            ) / (2 * eps)
            dv1 = (
                compute_v(x + eps * dx[1], alpha) - compute_v(x - eps * dx[1], alpha)
            ) / (2 * eps)
            # 计算散度 div_v
            div_v = (dv0 * dx[0] + dv1 * dx[1]).sum(-1)
            # 计算误差 error，这是一个修改版的连续性方程，使用乘积法则将其表达为对数概率的形式，而不是 less numerically stable log_prob.exp()。
            error = dlogp_da + (dlogp_dx * v).sum(-1) + div_v
            # 断言误差在一个非常小的范围内，否则输出错误消息
            self.assertLess(
                torch.abs(error).max(),
                0.005,
                "\n".join(
                    [
                        f"Dirichlet([{a1}, {a2}, {a3}]) gradient violates continuity equation:",
                        f"error = {error}",
                    ]
                ),
            )
class TestDistributionShapes(DistributionsTestCase):
    # 测试分布形状的单元测试类，继承自 DistributionsTestCase
    def setUp(self):
        # 设置测试环境
        super().setUp()
        # 初始化标量样本
        self.scalar_sample = 1
        # 初始化张量样本1
        self.tensor_sample_1 = torch.ones(3, 2)
        # 初始化张量样本2
        self.tensor_sample_2 = torch.ones(3, 2, 3)

    def test_entropy_shape(self):
        # 测试熵的形状
        for Dist, params in _get_examples():
            # 遍历获取的分布类和参数组合
            for i, param in enumerate(params):
                # 创建分布对象
                dist = Dist(validate_args=False, **param)
                try:
                    # 获取实际熵的形状
                    actual_shape = dist.entropy().size()
                    # 期望的形状为批次形状，如果没有则为空的 torch.Size 对象
                    expected_shape = (
                        dist.batch_shape if dist.batch_shape else torch.Size()
                    )
                    # 构建错误消息，用于形状不匹配时的断言
                    message = f"{Dist.__name__} example {i + 1}/{len(params)}, shape mismatch. expected {expected_shape}, actual {actual_shape}"  # noqa: B950
                    # 断言实际形状与期望形状相等
                    self.assertEqual(actual_shape, expected_shape, msg=message)
                except NotImplementedError:
                    continue

    def test_bernoulli_shape_scalar_params(self):
        # 测试伯努利分布的形状，标量参数版本
        bernoulli = Bernoulli(0.3)
        # 断言批次形状为空的 torch.Size 对象
        self.assertEqual(bernoulli._batch_shape, torch.Size())
        # 断言事件形状为空的 torch.Size 对象
        self.assertEqual(bernoulli._event_shape, torch.Size())
        # 断言从分布中抽样的结果形状为空的 torch.Size 对象
        self.assertEqual(bernoulli.sample().size(), torch.Size())
        # 断言从分布中抽样的结果形状为 (3, 2)
        self.assertEqual(bernoulli.sample((3, 2)).size(), torch.Size((3, 2)))
        # 断言对标量样本计算对数概率会抛出 ValueError 异常
        self.assertRaises(ValueError, bernoulli.log_prob, self.scalar_sample)
        # 断言对张量样本1计算对数概率的形状为 (3, 2)
        self.assertEqual(
            bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言对张量样本2计算对数概率的形状为 (3, 2, 3)
        self.assertEqual(
            bernoulli.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_bernoulli_shape_tensor_params(self):
        # 测试伯努利分布的形状，张量参数版本
        bernoulli = Bernoulli(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        # 断言批次形状为 (3, 2)
        self.assertEqual(bernoulli._batch_shape, torch.Size((3, 2)))
        # 断言事件形状为空的 torch.Size 对象
        self.assertEqual(bernoulli._event_shape, torch.Size(()))
        # 断言从分布中抽样的结果形状为 (3, 2)
        self.assertEqual(bernoulli.sample().size(), torch.Size((3, 2)))
        # 断言从分布中抽样的结果形状为 (3, 2, 3, 2)
        self.assertEqual(bernoulli.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        # 断言对张量样本1计算对数概率的形状为 (3, 2)
        self.assertEqual(
            bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言对张量样本2计算对数概率会抛出 ValueError 异常
        self.assertRaises(ValueError, bernoulli.log_prob, self.tensor_sample_2)
        # 断言对形状为 (3, 1, 1) 的张量计算对数概率的形状为 (3, 3, 2)
        self.assertEqual(
            bernoulli.log_prob(torch.ones(3, 1, 1)).size(), torch.Size((3, 3, 2))
        )
    # 定义测试函数，用于测试 Geometric 分布的形状与参数情况
    def test_geometric_shape_scalar_params(self):
        # 创建 Geometric 分布对象，参数为标量 0.3
        geometric = Geometric(0.3)
        # 断言批量形状为 torch.Size()
        self.assertEqual(geometric._batch_shape, torch.Size())
        # 断言事件形状为 torch.Size()
        self.assertEqual(geometric._event_shape, torch.Size())
        # 断言从分布中抽取样本的大小为 torch.Size()
        self.assertEqual(geometric.sample().size(), torch.Size())
        # 断言从分布中抽取样本的指定大小为 torch.Size((3, 2))
        self.assertEqual(geometric.sample((3, 2)).size(), torch.Size((3, 2)))
        # 断言对标量样本 self.scalar_sample 计算对数概率会抛出 ValueError 异常
        self.assertRaises(ValueError, geometric.log_prob, self.scalar_sample)
        # 断言对 tensor 样本 self.tensor_sample_1 计算对数概率的结果大小为 torch.Size((3, 2))
        self.assertEqual(
            geometric.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言对 tensor 样本 self.tensor_sample_2 计算对数概率的结果大小为 torch.Size((3, 2, 3))
        self.assertEqual(
            geometric.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    # 定义测试函数，用于测试 Geometric 分布在给定张量参数情况下的形状与参数
    def test_geometric_shape_tensor_params(self):
        # 创建 Geometric 分布对象，参数为 3x2 大小的张量
        geometric = Geometric(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        # 断言批量形状为 torch.Size((3, 2))
        self.assertEqual(geometric._batch_shape, torch.Size((3, 2)))
        # 断言事件形状为 torch.Size(())
        self.assertEqual(geometric._event_shape, torch.Size(()))
        # 断言从分布中抽取样本的大小为 torch.Size((3, 2))
        self.assertEqual(geometric.sample().size(), torch.Size((3, 2)))
        # 断言从分布中抽取样本的指定大小为 torch.Size((3, 2, 3, 2))
        self.assertEqual(geometric.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        # 断言对 tensor 样本 self.tensor_sample_1 计算对数概率的结果大小为 torch.Size((3, 2))
        self.assertEqual(
            geometric.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言对 tensor 样本 self.tensor_sample_2 计算对数概率时会抛出 ValueError 异常
        self.assertRaises(ValueError, geometric.log_prob, self.tensor_sample_2)
        # 断言对全为1的 3x1x1 大小的 tensor 计算对数概率的结果大小为 torch.Size((3, 3, 2))
        self.assertEqual(
            geometric.log_prob(torch.ones(3, 1, 1)).size(), torch.Size((3, 3, 2))
        )

    # 定义测试函数，用于测试 Beta 分布的形状与参数情况（标量参数）
    def test_beta_shape_scalar_params(self):
        # 创建 Beta 分布对象，参数为 α=0.1, β=0.1
        dist = Beta(0.1, 0.1)
        # 断言批量形状为 torch.Size()
        self.assertEqual(dist._batch_shape, torch.Size())
        # 断言事件形状为 torch.Size()
        self.assertEqual(dist._event_shape, torch.Size())
        # 断言从分布中抽取样本的大小为 torch.Size()
        self.assertEqual(dist.sample().size(), torch.Size())
        # 断言从分布中抽取样本的指定大小为 torch.Size((3, 2))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2)))
        # 断言对标量样本 self.scalar_sample 计算对数概率会抛出 ValueError 异常
        self.assertRaises(ValueError, dist.log_prob, self.scalar_sample)
        # 断言对 tensor 样本 self.tensor_sample_1 计算对数概率的结果大小为 torch.Size((3, 2))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        # 断言对 tensor 样本 self.tensor_sample_2 计算对数概率的结果大小为 torch.Size((3, 2, 3))
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    # 定义测试函数，用于测试 Beta 分布在给定张量参数情况下的形状与参数
    def test_beta_shape_tensor_params(self):
        # 创建 Beta 分布对象，参数为 3x2 大小的张量 α 和 β
        dist = Beta(
            torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        )
        # 断言批量形状为 torch.Size((3, 2))
        self.assertEqual(dist._batch_shape, torch.Size((3, 2)))
        # 断言事件形状为 torch.Size(())
        self.assertEqual(dist._event_shape, torch.Size(()))
        # 断言从分布中抽取样本的大小为 torch.Size((3, 2))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        # 断言从分布中抽取样本的指定大小为 torch.Size((3, 2, 3, 2))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        # 断言对 tensor 样本 self.tensor_sample_1 计算对数概率的结果大小为 torch.Size((3, 2))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        # 断言对 tensor 样本 self.tensor_sample_2 计算对数概率时会抛出 ValueError 异常
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        # 断言对全为1的 3x1x1 大小的 tensor 计算对数概率的结果大小为 torch.Size((3, 3, 2))
        self.assertEqual(
            dist.log_prob(torch.ones(3, 1, 1)).size(), torch.Size((3, 3, 2))
        )
    # 测试二项分布对象的批量形状属性和样本生成
    def test_binomial_shape(self):
        # 创建一个二项分布对象，其中有两个独立分布，每个分布成功概率为 0.6 和 0.3
        dist = Binomial(10, torch.tensor([0.6, 0.3]))
        # 验证批量形状（batch shape）为 (2,)
        self.assertEqual(dist._batch_shape, torch.Size((2,)))
        # 验证事件形状（event shape）为空
        self.assertEqual(dist._event_shape, torch.Size(()))
        # 验证生成单个样本的形状为 (2,)
        self.assertEqual(dist.sample().size(), torch.Size((2,)))
        # 验证生成指定形状样本的形状为 (3, 2, 2)
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        # 验证对给定样本计算对数概率的输出形状为 (3, 2)
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        # 验证当传入不合法样本时，抛出 ValueError 异常
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)

    # 测试具有向量化参数 n 的二项分布对象的批量形状属性和样本生成
    def test_binomial_shape_vectorized_n(self):
        # 创建一个二项分布对象，其中 n 参数为一个 2x3 的张量，每个分布的成功概率为 0.6, 0.3, 0.1
        dist = Binomial(
            torch.tensor([[10, 3, 1], [4, 8, 4]]), torch.tensor([0.6, 0.3, 0.1])
        )
        # 验证批量形状为 (2, 3)
        self.assertEqual(dist._batch_shape, torch.Size((2, 3)))
        # 验证事件形状为空
        self.assertEqual(dist._event_shape, torch.Size(()))
        # 验证生成单个样本的形状为 (2, 3)
        self.assertEqual(dist.sample().size(), torch.Size((2, 3)))
        # 验证生成指定形状样本的形状为 (3, 2, 2, 3)
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 2, 3)))
        # 验证对给定样本计算对数概率的输出形状为 (3, 2, 3)
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )
        # 验证当传入不合法样本时，抛出 ValueError 异常
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)

    # 测试多项分布对象的批量形状属性和样本生成
    def test_multinomial_shape(self):
        # 创建一个多项分布对象，总试验次数为 10，每个类别的概率分别为 [0.6, 0.3] 在三次试验中重复
        dist = Multinomial(10, torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        # 验证批量形状为 (3,)
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        # 验证事件形状为 (2,)
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        # 验证生成单个样本的形状为 (3, 2)
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        # 验证生成指定形状样本的形状为 (3, 2, 3, 2)
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        # 验证对给定样本计算对数概率的输出形状为 (3,)
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3,)))
        # 验证当传入不合法样本时，抛出 ValueError 异常
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        # 验证对全 1 样本计算对数概率的输出形状为 (3, 3)
        self.assertEqual(dist.log_prob(torch.ones(3, 1, 2)).size(), torch.Size((3, 3)))
    # 定义一个测试方法，用于测试 Categorical 分布的形状
    def test_categorical_shape(self):
        # 对于未分批数据的情况
        dist = Categorical(torch.tensor([0.6, 0.3, 0.1]))
        # 断言批次形状为空
        self.assertEqual(dist._batch_shape, torch.Size(()))
        # 断言事件形状为空
        self.assertEqual(dist._event_shape, torch.Size(()))
        # 断言从分布中抽样的结果形状是一个标量
        self.assertEqual(dist.sample().size(), torch.Size())
        # 断言从分布中抽样并指定形状的结果形状正确
        self.assertEqual(
            dist.sample((3, 2)).size(),
            torch.Size(
                (
                    3,
                    2,
                )
            ),
        )
        # 断言使用给定样本计算对数概率的结果形状为 (3, 2)
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        # 断言使用另一个给定样本计算对数概率的结果形状为 (3, 2, 3)
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )
        # 断言使用维度不匹配的样本会引发 ValueError 异常
        self.assertEqual(dist.log_prob(torch.ones(3, 1)).size(), torch.Size((3, 1)))
        
        # 对于已分批数据的情况
        dist = Categorical(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        # 断言批次形状为 (3,)
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        # 断言事件形状为空
        self.assertEqual(dist._event_shape, torch.Size(()))
        # 断言从分布中抽样的结果形状是 (3,)
        self.assertEqual(dist.sample().size(), torch.Size((3,)))
        # 断言从分布中抽样并指定形状的结果形状正确
        self.assertEqual(
            dist.sample((3, 2)).size(),
            torch.Size(
                (
                    3,
                    2,
                    3,
                )
            ),
        )
        # 断言使用给定样本计算对数概率时，对于 tensor_sample_1 应该引发 ValueError 异常
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)
        # 断言使用另一个给定样本计算对数概率的结果形状为 (3, 2, 3)
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )
        # 断言使用形状为 (3, 1) 的样本计算对数概率的结果形状为 (3, 3)
        self.assertEqual(dist.log_prob(torch.ones(3, 1)).size(), torch.Size((3, 3)))
    # 测试 OneHotCategorical 分布的形状和功能，未批处理情况下的测试
    def test_one_hot_categorical_shape(self):
        # 创建一个 OneHotCategorical 分布，概率参数为 [0.6, 0.3, 0.1]
        dist = OneHotCategorical(torch.tensor([0.6, 0.3, 0.1]))
        # 断言批处理形状为空
        self.assertEqual(dist._batch_shape, torch.Size(()))
        # 断言事件形状为 (3,)
        self.assertEqual(dist._event_shape, torch.Size((3,)))
        # 断言从分布中抽样结果的形状为 (3,)
        self.assertEqual(dist.sample().size(), torch.Size((3,)))
        # 断言抽样多次后的形状为 (3, 2, 3)
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3)))
        # 测试 log_prob 方法在给定样本 self.tensor_sample_1 时抛出 ValueError
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)
        # 创建一个扩展了维度的样本 sample，并验证其 log_prob 方法的返回形状为 (3, 2)
        sample = torch.tensor([0.0, 1.0, 0.0]).expand(3, 2, 3)
        self.assertEqual(
            dist.log_prob(sample).size(),
            torch.Size(
                (
                    3,
                    2,
                )
            ),
        )
        # 创建一个对角矩阵样本 sample，并验证其 log_prob 方法的返回形状为 (3,)
        sample = torch.eye(3)
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3,)))

        # 批处理情况下的测试
        # 创建一个批处理的 OneHotCategorical 分布，概率参数为 [[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]
        dist = OneHotCategorical(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        # 断言批处理形状为 (3,)
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        # 断言事件形状为 (2,)
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        # 断言从分布中抽样结果的形状为 (3, 2)
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        # 断言抽样多次后的形状为 (3, 2, 3, 2)
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        # 创建一个样本 sample，并验证其 log_prob 方法的返回形状为 (3,)
        sample = torch.tensor([0.0, 1.0])
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3,)))
        # 测试 log_prob 方法在给定样本 self.tensor_sample_2 时抛出 ValueError
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        # 创建一个扩展了维度的样本 sample，并验证其 log_prob 方法的返回形状为 (3, 3)
        sample = torch.tensor([0.0, 1.0]).expand(3, 1, 2)
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3, 3)))

    # 测试 Cauchy 分布的形状和功能，使用标量参数
    def test_cauchy_shape_scalar_params(self):
        # 创建一个 Cauchy 分布，位置参数为 0，尺度参数为 1
        cauchy = Cauchy(0, 1)
        # 断言批处理形状为空
        self.assertEqual(cauchy._batch_shape, torch.Size())
        # 断言事件形状为空
        self.assertEqual(cauchy._event_shape, torch.Size())
        # 断言从分布中抽样结果的形状为空
        self.assertEqual(cauchy.sample().size(), torch.Size())
        # 断言抽样多次后的形状为 (3, 2)
        self.assertEqual(cauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2)))
        # 测试 log_prob 方法在给定样本 self.scalar_sample 时抛出 ValueError
        self.assertRaises(ValueError, cauchy.log_prob, self.scalar_sample)
        # 创建一个样本 self.tensor_sample_1，并验证其 log_prob 方法的返回形状为 (3, 2)
        self.assertEqual(
            cauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 创建一个样本 self.tensor_sample_2，并验证其 log_prob 方法的返回形状为 (3, 2, 3)
        self.assertEqual(
            cauchy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )
    def test_cauchy_shape_tensor_params(self):
        # 创建一个 Cauchy 分布对象，参数是均值和标准差
        cauchy = Cauchy(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        # 检查批次形状（batch shape）是否为 (2,)
        self.assertEqual(cauchy._batch_shape, torch.Size((2,)))
        # 检查事件形状（event shape）是否为空
        self.assertEqual(cauchy._event_shape, torch.Size(()))
        # 从 Cauchy 分布中抽取一个样本，检查样本的形状是否为 (2,)
        self.assertEqual(cauchy.sample().size(), torch.Size((2,)))
        # 从 Cauchy 分布中抽取一个指定形状的样本，检查样本的形状是否为 (3, 2, 2)
        self.assertEqual(
            cauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2))
        )
        # 计算给定样本的对数概率密度，检查结果的形状是否为 (3, 2)
        self.assertEqual(
            cauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 对于给定的样本，引发 ValueError 异常，因为样本维度不匹配
        self.assertRaises(ValueError, cauchy.log_prob, self.tensor_sample_2)
        # 计算给定样本的对数概率密度，检查结果的形状是否为 (2, 2)
        self.assertEqual(cauchy.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_halfcauchy_shape_scalar_params(self):
        # 创建一个 HalfCauchy 分布对象，参数是标准差（标量）
        halfcauchy = HalfCauchy(1)
        # 检查批次形状是否为空
        self.assertEqual(halfcauchy._batch_shape, torch.Size())
        # 检查事件形状是否为空
        self.assertEqual(halfcauchy._event_shape, torch.Size())
        # 从 HalfCauchy 分布中抽取一个样本，检查样本的形状是否为空
        self.assertEqual(halfcauchy.sample().size(), torch.Size())
        # 从 HalfCauchy 分布中抽取一个指定形状的样本，检查样本的形状是否为 (3, 2)
        self.assertEqual(
            halfcauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2))
        )
        # 对于给定的样本，引发 ValueError 异常，因为该分布不支持标量样本的对数概率计算
        self.assertRaises(ValueError, halfcauchy.log_prob, self.scalar_sample)
        # 计算给定样本的对数概率密度，检查结果的形状是否为 (3, 2)
        self.assertEqual(
            halfcauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 对于给定的样本，引发 ValueError 异常，因为样本维度不匹配
        self.assertEqual(
            halfcauchy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_halfcauchy_shape_tensor_params(self):
        # 创建一个 HalfCauchy 分布对象，参数是标准差（张量，形状为 (2,)）
        halfcauchy = HalfCauchy(torch.tensor([1.0, 1.0]))
        # 检查批次形状是否为 (2,)
        self.assertEqual(halfcauchy._batch_shape, torch.Size((2,)))
        # 检查事件形状是否为空
        self.assertEqual(halfcauchy._event_shape, torch.Size(()))
        # 从 HalfCauchy 分布中抽取一个样本，检查样本的形状是否为 (2,)
        self.assertEqual(halfcauchy.sample().size(), torch.Size((2,)))
        # 从 HalfCauchy 分布中抽取一个指定形状的样本，检查样本的形状是否为 (3, 2, 2)
        self.assertEqual(
            halfcauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2))
        )
        # 计算给定样本的对数概率密度，检查结果的形状是否为 (3, 2)
        self.assertEqual(
            halfcauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 对于给定的样本，引发 ValueError 异常，因为样本维度不匹配
        self.assertRaises(ValueError, halfcauchy.log_prob, self.tensor_sample_2)
        # 计算给定样本的对数概率密度，检查结果的形状是否为 (2, 2)
        self.assertEqual(
            halfcauchy.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2))
        )
    # 定义测试函数，测试 Dirichlet 分布的形状相关方法
    def test_dirichlet_shape(self):
        # 创建 Dirichlet 分布对象，参数是一个张量
        dist = Dirichlet(torch.tensor([[0.6, 0.3], [1.6, 1.3], [2.6, 2.3]]))
        # 断言批次形状为 (3,)
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        # 断言事件形状为 (2,)
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        # 断言抽样一次的结果形状为 (3, 2)
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        # 断言抽样多次的结果形状为 (5, 4, 3, 2)
        self.assertEqual(dist.sample((5, 4)).size(), torch.Size((5, 4, 3, 2)))
        # 计算一个简单的样本，使其成为单位分布，断言对数概率的结果形状为 (3,)
        simplex_sample = self.tensor_sample_1 / self.tensor_sample_1.sum(-1, keepdim=True)
        self.assertEqual(dist.log_prob(simplex_sample).size(), torch.Size((3,)))
        # 断言对非单位分布的样本计算对数概率会引发 ValueError 异常
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        # 创建一个符合要求的简单样本，断言对数概率的结果形状为 (3, 3)
        simplex_sample = torch.ones(3, 1, 2)
        simplex_sample = simplex_sample / simplex_sample.sum(-1).unsqueeze(-1)
        self.assertEqual(dist.log_prob(simplex_sample).size(), torch.Size((3, 3)))

    # 定义测试函数，测试同一分布族的混合分布的形状相关方法
    def test_mixture_same_family_shape(self):
        # 创建混合分布对象，包含一个分类分布和多个正态分布
        dist = MixtureSameFamily(
            Categorical(torch.rand(5)), Normal(torch.randn(5), torch.rand(5))
        )
        # 断言批次形状为空
        self.assertEqual(dist._batch_shape, torch.Size())
        # 断言事件形状为空
        self.assertEqual(dist._event_shape, torch.Size())
        # 断言抽样一次的结果形状为空
        self.assertEqual(dist.sample().size(), torch.Size())
        # 断言抽样多次的结果形状为 (5, 4)
        self.assertEqual(dist.sample((5, 4)).size(), torch.Size((5, 4)))
        # 断言对给定样本计算对数概率的结果形状为 (3, 2)
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        # 断言对给定不符合要求的样本计算对数概率的结果形状为 (3, 2, 3)
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    # 定义测试函数，测试 Gamma 分布使用标量参数时的形状相关方法
    def test_gamma_shape_scalar_params(self):
        # 创建 Gamma 分布对象，参数是标量
        gamma = Gamma(1, 1)
        # 断言批次形状为空
        self.assertEqual(gamma._batch_shape, torch.Size())
        # 断言事件形状为空
        self.assertEqual(gamma._event_shape, torch.Size())
        # 断言抽样一次的结果形状为空
        self.assertEqual(gamma.sample().size(), torch.Size())
        # 断言抽样多次的结果形状为 (3, 2)
        self.assertEqual(gamma.sample((3, 2)).size(), torch.Size((3, 2)))
        # 断言对标量样本计算对数概率的结果形状为空
        self.assertEqual(gamma.log_prob(self.scalar_sample).size(), torch.Size())
        # 断言对给定样本计算对数概率的结果形状为 (3, 2)
        self.assertEqual(
            gamma.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言对给定不符合要求的样本计算对数概率的结果形状为 (3, 2, 3)
        self.assertEqual(
            gamma.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    # 定义测试函数，测试 Gamma 分布使用张量参数时的形状相关方法
    def test_gamma_shape_tensor_params(self):
        # 创建 Gamma 分布对象，参数是张量
        gamma = Gamma(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
        # 断言批次形状为 (2,)
        self.assertEqual(gamma._batch_shape, torch.Size((2,)))
        # 断言事件形状为空
        self.assertEqual(gamma._event_shape, torch.Size(()))
        # 断言抽样一次的结果形状为 (2,)
        self.assertEqual(gamma.sample().size(), torch.Size((2,)))
        # 断言抽样多次的结果形状为 (3, 2, 2)
        self.assertEqual(gamma.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        # 断言对给定样本计算对数概率的结果形状为 (3, 2)
        self.assertEqual(
            gamma.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言对给定不符合要求的样本计算对数概率会引发 ValueError 异常
        self.assertRaises(ValueError, gamma.log_prob, self.tensor_sample_2)
        # 断言对全为 1 的样本计算对数概率的结果形状为 (2, 2)
        self.assertEqual(gamma.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))
    # 测试 Chi2 分布的形状，使用标量参数
    def test_chi2_shape_scalar_params(self):
        # 创建 Chi2 分布对象，自由度为 1
        chi2 = Chi2(1)
        # 断言批处理形状为空
        self.assertEqual(chi2._batch_shape, torch.Size())
        # 断言事件形状为空
        self.assertEqual(chi2._event_shape, torch.Size())
        # 断言从分布中抽取样本的形状为标量
        self.assertEqual(chi2.sample().size(), torch.Size())
        # 断言从分布中抽取多个样本的形状为 (3, 2)
        self.assertEqual(chi2.sample((3, 2)).size(), torch.Size((3, 2)))
        # 断言计算标量样本的对数概率密度的形状为标量
        self.assertEqual(chi2.log_prob(self.scalar_sample).size(), torch.Size())
        # 断言计算 tensor_sample_1 样本的对数概率密度的形状为 (3, 2)
        self.assertEqual(chi2.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        # 断言计算 tensor_sample_2 样本的对数概率密度的形状为 (3, 2, 3)
        self.assertEqual(
            chi2.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    # 测试 Chi2 分布的形状，使用张量参数
    def test_chi2_shape_tensor_params(self):
        # 创建 Chi2 分布对象，自由度为 [1.0, 1.0]
        chi2 = Chi2(torch.tensor([1.0, 1.0]))
        # 断言批处理形状为 (2,)
        self.assertEqual(chi2._batch_shape, torch.Size((2,)))
        # 断言事件形状为空
        self.assertEqual(chi2._event_shape, torch.Size(()))
        # 断言从分布中抽取样本的形状为 (2,)
        self.assertEqual(chi2.sample().size(), torch.Size((2,)))
        # 断言从分布中抽取多个样本的形状为 (3, 2, 2)
        self.assertEqual(chi2.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        # 断言计算 tensor_sample_1 样本的对数概率密度的形状为 (3, 2)
        self.assertEqual(chi2.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        # 检查是否引发 ValueError 异常，因为 tensor_sample_2 的形状不兼容
        self.assertRaises(ValueError, chi2.log_prob, self.tensor_sample_2)
        # 断言计算全为 1 的 tensor 样本的对数概率密度的形状为 (2, 2)
        self.assertEqual(chi2.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    # 测试 StudentT 分布的形状，使用标量参数
    def test_studentT_shape_scalar_params(self):
        # 创建 StudentT 分布对象，自由度为 1
        st = StudentT(1)
        # 断言批处理形状为空
        self.assertEqual(st._batch_shape, torch.Size())
        # 断言事件形状为空
        self.assertEqual(st._event_shape, torch.Size())
        # 断言从分布中抽取样本的形状为标量
        self.assertEqual(st.sample().size(), torch.Size())
        # 断言从分布中抽取多个样本的形状为 (3, 2)
        self.assertEqual(st.sample((3, 2)).size(), torch.Size((3, 2)))
        # 检查是否引发 ValueError 异常，因为 scalar_sample 的形状不兼容
        self.assertRaises(ValueError, st.log_prob, self.scalar_sample)
        # 断言计算 tensor_sample_1 样本的对数概率密度的形状为 (3, 2)
        self.assertEqual(st.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        # 断言计算 tensor_sample_2 样本的对数概率密度的形状为 (3, 2, 3)
        self.assertEqual(
            st.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    # 测试 StudentT 分布的形状，使用张量参数
    def test_studentT_shape_tensor_params(self):
        # 创建 StudentT 分布对象，自由度为 [1.0, 1.0]
        st = StudentT(torch.tensor([1.0, 1.0]))
        # 断言批处理形状为 (2,)
        self.assertEqual(st._batch_shape, torch.Size((2,)))
        # 断言事件形状为空
        self.assertEqual(st._event_shape, torch.Size(()))
        # 断言从分布中抽取样本的形状为 (2,)
        self.assertEqual(st.sample().size(), torch.Size((2,)))
        # 断言从分布中抽取多个样本的形状为 (3, 2, 2)
        self.assertEqual(st.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        # 断言计算 tensor_sample_1 样本的对数概率密度的形状为 (3, 2)
        self.assertEqual(st.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        # 检查是否引发 ValueError 异常，因为 tensor_sample_2 的形状不兼容
        self.assertRaises(ValueError, st.log_prob, self.tensor_sample_2)
        # 断言计算全为 1 的 tensor 样本的对数概率密度的形状为 (2, 2)
        self.assertEqual(st.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    # 测试 Pareto 分布的形状，使用标量参数
    def test_pareto_shape_scalar_params(self):
        # 创建 Pareto 分布对象，scale 为 1，alpha 为 1
        pareto = Pareto(1, 1)
        # 断言批处理形状为空
        self.assertEqual(pareto._batch_shape, torch.Size())
        # 断言事件形状为空
        self.assertEqual(pareto._event_shape, torch.Size())
        # 断言从分布中抽取样本的形状为标量
        self.assertEqual(pareto.sample().size(), torch.Size())
        # 断言从分布中抽取多个样本的形状为 (3, 2)
        self.assertEqual(pareto.sample((3, 2)).size(), torch.Size((3, 2)))
        # 断言计算 tensor_sample_1 + 1 样本的对数概率密度的形状为 (3, 2)
        self.assertEqual(
            pareto.log_prob(self.tensor_sample_1 + 1).size(), torch.Size((3, 2))
        )
        # 断言计算 tensor_sample_2 + 1 样本的对数概率密度的形状为 (3, 2, 3)
        self.assertEqual(
            pareto.log_prob(self.tensor_sample_2 + 1).size(), torch.Size((3, 2, 3))
        )
    # 测试 Gumbel 分布的形状，使用标量参数
    def test_gumbel_shape_scalar_params(self):
        # 创建 Gumbel 分布对象，参数 loc=1, scale=1
        gumbel = Gumbel(1, 1)
        # 断言批次形状为标量
        self.assertEqual(gumbel._batch_shape, torch.Size())
        # 断言事件形状为标量
        self.assertEqual(gumbel._event_shape, torch.Size())
        # 断言从分布中抽样的结果形状为标量
        self.assertEqual(gumbel.sample().size(), torch.Size())
        # 断言从分布中抽样并指定形状 (3, 2) 的结果形状为 (3, 2)
        self.assertEqual(gumbel.sample((3, 2)).size(), torch.Size((3, 2)))
        # 断言计算给定样本的对数概率密度函数值的结果形状为 (3, 2)
        self.assertEqual(
            gumbel.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言计算给定另一组样本的对数概率密度函数值的结果形状为 (3, 2, 3)
        self.assertEqual(
            gumbel.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    # 测试 Kumaraswamy 分布的形状，使用标量参数
    def test_kumaraswamy_shape_scalar_params(self):
        # 创建 Kumaraswamy 分布对象，参数 concentration1=1, concentration0=1
        kumaraswamy = Kumaraswamy(1, 1)
        # 断言批次形状为标量
        self.assertEqual(kumaraswamy._batch_shape, torch.Size())
        # 断言事件形状为标量
        self.assertEqual(kumaraswamy._event_shape, torch.Size())
        # 断言从分布中抽样的结果形状为标量
        self.assertEqual(kumaraswamy.sample().size(), torch.Size())
        # 断言从分布中抽样并指定形状 (3, 2) 的结果形状为 (3, 2)
        self.assertEqual(kumaraswamy.sample((3, 2)).size(), torch.Size((3, 2)))
        # 断言计算给定样本的对数概率密度函数值的结果形状为 (3, 2)
        self.assertEqual(
            kumaraswamy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言计算给定另一组样本的对数概率密度函数值的结果形状为 (3, 2, 3)
        self.assertEqual(
            kumaraswamy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    # 测试 VonMises 分布的形状，使用张量参数
    def test_vonmises_shape_tensor_params(self):
        # 创建 VonMises 分布对象，参数 loc=torch.tensor([0.0, 0.0]), concentration=torch.tensor([1.0, 1.0])
        von_mises = VonMises(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        # 断言批次形状为 (2,)
        self.assertEqual(von_mises._batch_shape, torch.Size((2,)))
        # 断言事件形状为标量
        self.assertEqual(von_mises._event_shape, torch.Size(()))
        # 断言从分布中抽样的结果形状为 (2,)
        self.assertEqual(von_mises.sample().size(), torch.Size((2,)))
        # 断言从分布中抽样并指定形状 (3, 2) 的结果形状为 (3, 2, 2)
        self.assertEqual(
            von_mises.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2))
        )
        # 断言计算给定样本的对数概率密度函数值的结果形状为 (3, 2)
        self.assertEqual(
            von_mises.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言计算给定张量样本的对数概率密度函数值的结果形状为 (2, 2)
        self.assertEqual(
            von_mises.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2))
        )

    # 测试 VonMises 分布的形状，使用标量参数
    def test_vonmises_shape_scalar_params(self):
        # 创建 VonMises 分布对象，参数 loc=0.0, concentration=1.0
        von_mises = VonMises(0.0, 1.0)
        # 断言批次形状为标量
        self.assertEqual(von_mises._batch_shape, torch.Size())
        # 断言事件形状为标量
        self.assertEqual(von_mises._event_shape, torch.Size())
        # 断言从分布中抽样的结果形状为标量
        self.assertEqual(von_mises.sample().size(), torch.Size())
        # 断言从分布中抽样并指定形状 (3, 2) 的结果形状为 (3, 2)
        self.assertEqual(
            von_mises.sample(torch.Size((3, 2))).size(), torch.Size((3, 2))
        )
        # 断言计算给定样本的对数概率密度函数值的结果形状为 (3, 2)
        self.assertEqual(
            von_mises.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言计算给定另一组样本的对数概率密度函数值的结果形状为 (3, 2, 3)
        self.assertEqual(
            von_mises.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )
    # 测试 Weibull 分布的标量参数的形状
    def test_weibull_scale_scalar_params(self):
        # 创建 Weibull 分布对象，参数为形状参数为 1，尺度参数为 1
        weibull = Weibull(1, 1)
        # 断言批次形状为标量
        self.assertEqual(weibull._batch_shape, torch.Size())
        # 断言事件形状为标量
        self.assertEqual(weibull._event_shape, torch.Size())
        # 断言从分布中抽样一次的结果形状为标量
        self.assertEqual(weibull.sample().size(), torch.Size())
        # 断言从分布中抽样形状为 (3, 2) 的结果形状为 (3, 2)
        self.assertEqual(weibull.sample((3, 2)).size(), torch.Size((3, 2)))
        # 断言计算使用给定样本 self.tensor_sample_1 的对数概率密度函数结果的形状为 (3, 2)
        self.assertEqual(
            weibull.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言计算使用给定样本 self.tensor_sample_2 的对数概率密度函数结果的形状为 (3, 2, 3)

    # 测试 Wishart 分布的标量参数的形状
    def test_wishart_shape_scalar_params(self):
        # 创建 Wishart 分布对象，参数为自由度为 1，精度矩阵为 [[1.0]]
        wishart = Wishart(torch.tensor(1), torch.tensor([[1.0]]))
        # 断言批次形状为标量
        self.assertEqual(wishart._batch_shape, torch.Size())
        # 断言事件形状为 (1, 1)
        self.assertEqual(wishart._event_shape, torch.Size((1, 1)))
        # 断言从分布中抽样一次的结果形状为 (1, 1)
        self.assertEqual(wishart.sample().size(), torch.Size((1, 1)))
        # 断言从分布中抽样形状为 (3, 2) 的结果形状为 (3, 2, 1, 1)
        self.assertEqual(wishart.sample((3, 2)).size(), torch.Size((3, 2, 1, 1)))
        # 使用标量样本抛出 ValueError 异常
        self.assertRaises(ValueError, wishart.log_prob, self.scalar_sample)

    # 测试 Wishart 分布的张量参数的形状
    def test_wishart_shape_tensor_params(self):
        # 创建 Wishart 分布对象，参数为自由度为 [1.0, 1.0]，精度矩阵为 [[[1.0]], [[1.0]]]
        wishart = Wishart(torch.tensor([1.0, 1.0]), torch.tensor([[[1.0]], [[1.0]]]))
        # 断言批次形状为 (2,)
        self.assertEqual(wishart._batch_shape, torch.Size((2,)))
        # 断言事件形状为 (1, 1)
        self.assertEqual(wishart._event_shape, torch.Size((1, 1)))
        # 断言从分布中抽样一次的结果形状为 (2, 1, 1)
        self.assertEqual(wishart.sample().size(), torch.Size((2, 1, 1)))
        # 断言从分布中抽样形状为 (3, 2) 的结果形状为 (3, 2, 2, 1, 1)
        self.assertEqual(wishart.sample((3, 2)).size(), torch.Size((3, 2, 2, 1, 1)))
        # 使用 tensor_sample_2 样本抛出 ValueError 异常
        self.assertRaises(ValueError, wishart.log_prob, self.tensor_sample_2)
        # 断言计算使用全为 1 的张量样本的对数概率密度函数结果的形状为 (2,)
        self.assertEqual(wishart.log_prob(torch.ones(2, 1, 1)).size(), torch.Size((2,)))

    # 测试 Normal 分布的标量参数的形状
    def test_normal_shape_scalar_params(self):
        # 创建 Normal 分布对象，参数为均值 0，标准差 1
        normal = Normal(0, 1)
        # 断言批次形状为标量
        self.assertEqual(normal._batch_shape, torch.Size())
        # 断言事件形状为标量
        self.assertEqual(normal._event_shape, torch.Size())
        # 断言从分布中抽样一次的结果形状为标量
        self.assertEqual(normal.sample().size(), torch.Size())
        # 断言从分布中抽样形状为 (3, 2) 的结果形状为 (3, 2)
        self.assertEqual(normal.sample((3, 2)).size(), torch.Size((3, 2)))
        # 使用标量样本抛出 ValueError 异常
        self.assertRaises(ValueError, normal.log_prob, self.scalar_sample)
        # 断言计算使用给定样本 self.tensor_sample_1 的对数概率密度函数结果的形状为 (3, 2)
        self.assertEqual(
            normal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 断言计算使用给定样本 self.tensor_sample_2 的对数概率密度函数结果的形状为 (3, 2, 3)

    # 测试 Normal 分布的张量参数的形状
    def test_normal_shape_tensor_params(self):
        # 创建 Normal 分布对象，参数为均值 [0.0, 0.0]，标准差 [1.0, 1.0]
        normal = Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        # 断言批次形状为 (2,)
        self.assertEqual(normal._batch_shape, torch.Size((2,)))
        # 断言事件形状为标量
        self.assertEqual(normal._event_shape, torch.Size(()))
        # 断言从分布中抽样一次的结果形状为 (2,)
        self.assertEqual(normal.sample().size(), torch.Size((2,)))
        # 断言从分布中抽样形状为 (3, 2) 的结果形状为 (3, 2, 2)
        self.assertEqual(normal.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        # 断言计算使用给定样本 self.tensor_sample_1 的对数概率密度函数结果的形状为 (3, 2)
        self.assertEqual(
            normal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 使用 tensor_sample_2 样本抛出 ValueError 异常
        self.assertRaises(ValueError, normal.log_prob, self.tensor_sample_2)
        # 断言计算使用全为 1 的张量样本的对数概率密度函数结果的形状为 (2, 2)
        self.assertEqual(normal.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))
    # 测试Uniform分布类在使用标量参数时的形状验证
    def test_uniform_shape_scalar_params(self):
        # 创建一个0到1的均匀分布对象
        uniform = Uniform(0, 1)
        # 验证批处理形状为torch.Size(())
        self.assertEqual(uniform._batch_shape, torch.Size())
        # 验证事件形状为torch.Size(())
        self.assertEqual(uniform._event_shape, torch.Size())
        # 验证从分布中抽样的结果的形状
        self.assertEqual(uniform.sample().size(), torch.Size())
        # 验证从分布中抽样并指定形状的结果的形状
        self.assertEqual(uniform.sample(torch.Size((3, 2))).size(), torch.Size((3, 2)))
        # 验证对标量样本计算对数概率会引发ValueError异常
        self.assertRaises(ValueError, uniform.log_prob, self.scalar_sample)
        # 验证对给定tensor样本计算对数概率的结果形状为torch.Size((3, 2))
        self.assertEqual(
            uniform.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 验证对给定tensor样本计算对数概率的结果形状为torch.Size((3, 2, 3))
        self.assertEqual(
            uniform.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    # 测试Uniform分布类在使用tensor参数时的形状验证
    def test_uniform_shape_tensor_params(self):
        # 创建一个从[0.0, 0.0]到[1.0, 1.0]的均匀分布对象
        uniform = Uniform(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        # 验证批处理形状为torch.Size((2,))
        self.assertEqual(uniform._batch_shape, torch.Size((2,)))
        # 验证事件形状为torch.Size(())
        self.assertEqual(uniform._event_shape, torch.Size(()))
        # 验证从分布中抽样的结果的形状为torch.Size((2,))
        self.assertEqual(uniform.sample().size(), torch.Size((2,)))
        # 验证从分布中抽样并指定形状的结果的形状为torch.Size((3, 2, 2))
        self.assertEqual(
            uniform.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2))
        )
        # 验证对给定tensor样本计算对数概率的结果形状为torch.Size((3, 2))
        self.assertEqual(
            uniform.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 验证对给定tensor样本计算对数概率会引发ValueError异常
        self.assertRaises(ValueError, uniform.log_prob, self.tensor_sample_2)
        # 验证对给定tensor样本计算对数概率的结果形状为torch.Size((2, 2))
        self.assertEqual(uniform.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    # 测试Exponential分布类在使用标量参数时的形状验证
    def test_exponential_shape_scalar_param(self):
        # 创建一个参数为1.0的指数分布对象
        expon = Exponential(1.0)
        # 验证批处理形状为torch.Size(())
        self.assertEqual(expon._batch_shape, torch.Size())
        # 验证事件形状为torch.Size(())
        self.assertEqual(expon._event_shape, torch.Size())
        # 验证从分布中抽样的结果的形状
        self.assertEqual(expon.sample().size(), torch.Size())
        # 验证从分布中抽样并指定形状的结果的形状为torch.Size((3, 2))
        self.assertEqual(expon.sample((3, 2)).size(), torch.Size((3, 2)))
        # 验证对标量样本计算对数概率会引发ValueError异常
        self.assertRaises(ValueError, expon.log_prob, self.scalar_sample)
        # 验证对给定tensor样本计算对数概率的结果形状为torch.Size((3, 2))
        self.assertEqual(
            expon.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 验证对给定tensor样本计算对数概率的结果形状为torch.Size((3, 2, 3))
        self.assertEqual(
            expon.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    # 测试Exponential分布类在使用tensor参数时的形状验证
    def test_exponential_shape_tensor_param(self):
        # 创建一个参数为[1.0, 1.0]的指数分布对象
        expon = Exponential(torch.tensor([1.0, 1.0]))
        # 验证批处理形状为torch.Size((2,))
        self.assertEqual(expon._batch_shape, torch.Size((2,)))
        # 验证事件形状为torch.Size(())
        self.assertEqual(expon._event_shape, torch.Size(()))
        # 验证从分布中抽样的结果的形状为torch.Size((2,))
        self.assertEqual(expon.sample().size(), torch.Size((2,)))
        # 验证从分布中抽样并指定形状的结果的形状为torch.Size((3, 2, 2))
        self.assertEqual(expon.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        # 验证对给定tensor样本计算对数概率的结果形状为torch.Size((3, 2))
        self.assertEqual(
            expon.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        # 验证对给定tensor样本计算对数概率会引发ValueError异常
        self.assertRaises(ValueError, expon.log_prob, self.tensor_sample_2)
        # 验证对给定tensor样本计算对数概率的结果形状为torch.Size((2, 2))
        self.assertEqual(expon.log_prob(torch.ones(2, 2)).size(), torch.Size((2, 2)))
    
    # 定义测试函数，验证 ContinuousBernoulli 分布的形状、采样、对数概率计算等方法
    def test_continuous_bernoulli_shape_tensor_params(self):
        # 创建 ContinuousBernoulli 分布对象，输入概率张量
        continuous_bernoulli = ContinuousBernoulli(
            torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]])
        )
        # 验证批次形状（batch shape）为 (3, 2)
        self.assertEqual(continuous_bernoulli._batch_shape, torch.Size((3, 2)))
        # 验证事件形状（event shape）为空
        self.assertEqual(continuous_bernoulli._event_shape, torch.Size(()))
        # 验证单次采样的形状为 (3, 2)
        self.assertEqual(continuous_bernoulli.sample().size(), torch.Size((3, 2)))
        # 验证指定多次采样的形状为 (3, 2, 3, 2)
        self.assertEqual(
            continuous_bernoulli.sample((3, 2)).size(), torch.Size((3, 2, 3, 2))
        )
        # 验证给定样本的对数概率形状为 (3, 2)
        self.assertEqual(
            continuous_bernoulli.log_prob(self.tensor_sample_1).size(),
            torch.Size((3, 2)),
        )
        # 验证对于无效样本抛出 ValueError 异常
        self.assertRaises(
            ValueError, continuous_bernoulli.log_prob, self.tensor_sample_2
        )
        # 验证对于形状为 (3, 1, 1) 的张量样本，对数概率的形状为 (3, 3, 2)
        self.assertEqual(
            continuous_bernoulli.log_prob(torch.ones(3, 1, 1)).size(),
            torch.Size((3, 3, 2)),
        )

    # 如果当前环境不适合 TorchDynamo 测试，则跳过此测试用例
    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    # 测试混合分布中组件分布的均值形状
    def test_mixture_same_family_mean_shape(self):
        # 创建一个混合分布，使用 Categorical 分布作为混合系数分布，Normal 分布作为组件分布
        mix_distribution = Categorical(torch.ones([3, 1, 3]))
        component_distribution = Normal(torch.zeros([3, 3, 3]), torch.ones([3, 3, 3]))
        gmm = MixtureSameFamily(mix_distribution, component_distribution)
        # 验证混合分布的均值张量的维度为 2
        self.assertEqual(len(gmm.mean.shape), 2)
# 使用 @skipIfTorchDynamo 装饰器，跳过不适合 TorchDynamo 的测试用例
@skipIfTorchDynamo("Not a TorchDynamo suitable test")
# 定义 TestKL 类，继承自 DistributionsTestCase 类
class TestKL(DistributionsTestCase):
    # 测试 KL 散度的 Monte Carlo 方法
    def test_kl_monte_carlo(self):
        set_rng_seed(0)  # 设置随机数种子，见注释 [Randomized statistical tests]
        # 对每一对 (p, q) 在 self.finite_examples 中执行测试
        for (p, _), (_, q) in self.finite_examples:
            actual = kl_divergence(p, q)
            numerator = 0  # 初始化分子
            denominator = 0  # 初始化分母
            # 使用 Monte Carlo 方法计算 KL 散度
            while denominator < self.max_samples:
                x = p.sample(sample_shape=(self.samples_per_batch,))
                numerator += (p.log_prob(x) - q.log_prob(x)).sum(0)
                denominator += x.size(0)
                expected = numerator / denominator  # 计算期望值
                # 计算误差
                error = torch.abs(expected - actual) / (1 + expected)
                # 如果误差小于精度要求，则退出循环
                if error[error == error].max() < self.precision:
                    break
            # 断言误差小于指定精度
            self.assertLess(
                error[error == error].max(),
                self.precision,
                "\n".join(
                    [
                        f"Incorrect KL({type(p).__name__}, {type(q).__name__}).",
                        f"Expected ({denominator} Monte Carlo samples): {expected}",
                        f"Actual (analytic): {actual}",
                    ]
                ),
            )

    # 对于 Multivariate normal 分布，由于需要随机生成正定（半正定）矩阵，因此需要单独的基于 Monte Carlo 的测试
    def test_kl_multivariate_normal(self):
        set_rng_seed(0)  # 设置随机数种子，见注释 [Randomized statistical tests]
        n = 5  # Multivariate normal 分布的测试次数
        # 对每个测试实例进行循环
        for i in range(0, n):
            # 随机生成 loc
            loc = [torch.randn(4) for _ in range(0, 2)]
            # 随机生成 scale_tril，并转换为半正定
            scale_tril = [
                transform_to(constraints.lower_cholesky)(torch.randn(4, 4))
                for _ in range(0, 2)
            ]
            # 创建两个 MultivariateNormal 实例 p 和 q
            p = MultivariateNormal(loc=loc[0], scale_tril=scale_tril[0])
            q = MultivariateNormal(loc=loc[1], scale_tril=scale_tril[1])
            actual = kl_divergence(p, q)  # 计算实际的 KL 散度
            numerator = 0  # 初始化分子
            denominator = 0  # 初始化分母
            # 使用 Monte Carlo 方法计算 KL 散度
            while denominator < self.max_samples:
                x = p.sample(sample_shape=(self.samples_per_batch,))
                numerator += (p.log_prob(x) - q.log_prob(x)).sum(0)
                denominator += x.size(0)
                expected = numerator / denominator  # 计算期望值
                # 计算误差
                error = torch.abs(expected - actual) / (1 + expected)
                # 如果误差小于精度要求，则退出循环
                if error[error == error].max() < self.precision:
                    break
            # 断言误差小于指定精度
            self.assertLess(
                error[error == error].max(),
                self.precision,
                "\n".join(
                    [
                        f"Incorrect KL(MultivariateNormal, MultivariateNormal) instance {i + 1}/{n}",
                        f"Expected ({denominator} Monte Carlo sample): {expected}",
                        f"Actual (analytic): {actual}",
                    ]
                ),
            )
    # 定义一个测试函数，用于测试批量多元正态分布的 KL 散度计算
    def test_kl_multivariate_normal_batched(self):
        b = 7  # 批次数
        # 生成两个长度为 b 的列表，每个元素是一个形状为 (b, 3) 的张量，元素值服从标准正态分布
        loc = [torch.randn(b, 3) for _ in range(0, 2)]
        # 生成两个长度为 b 的列表，每个元素是一个形状为 (b, 3, 3) 的下三角矩阵张量
        scale_tril = [
            transform_to(constraints.lower_cholesky)(torch.randn(b, 3, 3))
            for _ in range(0, 2)
        ]
        # 计算期望的 KL 散度，通过将每对多元正态分布的参数传递给 kl_divergence 函数并堆叠结果张量
        expected_kl = torch.stack(
            [
                kl_divergence(
                    MultivariateNormal(loc[0][i], scale_tril=scale_tril[0][i]),
                    MultivariateNormal(loc[1][i], scale_tril=scale_tril[1][i]),
                )
                for i in range(0, b)
            ]
        )
        # 计算实际的 KL 散度，通过将两个多元正态分布的参数传递给 kl_divergence 函数
        actual_kl = kl_divergence(
            MultivariateNormal(loc[0], scale_tril=scale_tril[0]),
            MultivariateNormal(loc[1], scale_tril=scale_tril[1]),
        )
        # 使用单元测试断言函数检查期望的 KL 散度和实际计算的 KL 散度是否相等
        self.assertEqual(expected_kl, actual_kl)

    # 定义另一个测试函数，用于测试广播后的批量多元正态分布的 KL 散度计算
    def test_kl_multivariate_normal_batched_broadcasted(self):
        b = 7  # 批次数
        # 生成两个长度为 b 的列表，每个元素是一个形状为 (b, 3) 的张量，元素值服从标准正态分布
        loc = [torch.randn(b, 3) for _ in range(0, 2)]
        # 生成两个元素分别为形状为 (b, 3, 3) 和 (3, 3) 的下三角矩阵张量列表
        scale_tril = [
            transform_to(constraints.lower_cholesky)(torch.randn(b, 3, 3)),
            transform_to(constraints.lower_cholesky)(torch.randn(3, 3)),
        ]
        # 计算期望的 KL 散度，通过将每对多元正态分布的参数传递给 kl_divergence 函数并堆叠结果张量
        expected_kl = torch.stack(
            [
                kl_divergence(
                    MultivariateNormal(loc[0][i], scale_tril=scale_tril[0][i]),
                    MultivariateNormal(loc[1][i], scale_tril=scale_tril[1]),
                )
                for i in range(0, b)
            ]
        )
        # 计算实际的 KL 散度，通过将两个多元正态分布的参数传递给 kl_divergence 函数
        actual_kl = kl_divergence(
            MultivariateNormal(loc[0], scale_tril=scale_tril[0]),
            MultivariateNormal(loc[1], scale_tril=scale_tril[1]),
        )
        # 使用单元测试断言函数检查期望的 KL 散度和实际计算的 KL 散度是否相等
        self.assertEqual(expected_kl, actual_kl)
    # 定义一个测试函数，用于测试低秩多变量正态分布的 KL 散度计算
    def test_kl_lowrank_multivariate_normal(self):
        set_rng_seed(0)  # 设置随机数种子，见注释 [Randomized statistical tests]
        n = 5  # 设定进行 lowrank_multivariate_normal 测试的次数
        # 循环执行测试，共进行 n 次
        for i in range(0, n):
            # 生成两组均值向量 loc，每个元素为四维随机数张量
            loc = [torch.randn(4) for _ in range(0, 2)]
            # 生成两组因子矩阵 cov_factor，每个元素为四行三列的随机数张量
            cov_factor = [torch.randn(4, 3) for _ in range(0, 2)]
            # 生成两组对角矩阵 cov_diag，每个元素为四维随机数，经过转换为正数后的张量
            cov_diag = [
                transform_to(constraints.positive)(torch.randn(4)) for _ in range(0, 2)
            ]
            # 生成两组协方差矩阵 covariance_matrix，每个元素为对应的因子矩阵乘以其转置加上对角矩阵得到的张量
            covariance_matrix = [
                cov_factor[i].matmul(cov_factor[i].t()) + cov_diag[i].diag()
                for i in range(0, 2)
            ]
            # 创建 LowRankMultivariateNormal 对象 p，用 loc[0]、cov_factor[0]、cov_diag[0] 初始化
            p = LowRankMultivariateNormal(loc[0], cov_factor[0], cov_diag[0])
            # 创建 LowRankMultivariateNormal 对象 q，用 loc[1]、cov_factor[1]、cov_diag[1] 初始化
            q = LowRankMultivariateNormal(loc[1], cov_factor[1], cov_diag[1])
            # 创建 MultivariateNormal 对象 p_full，用 loc[0]、covariance_matrix[0] 初始化
            p_full = MultivariateNormal(loc[0], covariance_matrix[0])
            # 创建 MultivariateNormal 对象 q_full，用 loc[1]、covariance_matrix[1] 初始化
            q_full = MultivariateNormal(loc[1], covariance_matrix[1])
            # 计算从 p_full 到 q_full 的 KL 散度，作为预期值
            expected = kl_divergence(p_full, q_full)

            # 计算从 p 到 q 的 KL 散度，作为实际值
            actual_lowrank_lowrank = kl_divergence(p, q)
            # 计算误差，即实际值与预期值的最大绝对差
            error_lowrank_lowrank = torch.abs(actual_lowrank_lowrank - expected).max()
            # 断言误差小于预设精度值 self.precision，并输出详细错误信息
            self.assertLess(
                error_lowrank_lowrank,
                self.precision,
                "\n".join(
                    [
                        f"Incorrect KL(LowRankMultivariateNormal, LowRankMultivariateNormal) instance {i + 1}/{n}",
                        f"Expected (from KL MultivariateNormal): {expected}",
                        f"Actual (analytic): {actual_lowrank_lowrank}",
                    ]
                ),
            )

            # 计算从 p 到 q_full 的 KL 散度，作为实际值
            actual_lowrank_full = kl_divergence(p, q_full)
            # 计算误差，即实际值与预期值的最大绝对差
            error_lowrank_full = torch.abs(actual_lowrank_full - expected).max()
            # 断言误差小于预设精度值 self.precision，并输出详细错误信息
            self.assertLess(
                error_lowrank_full,
                self.precision,
                "\n".join(
                    [
                        f"Incorrect KL(LowRankMultivariateNormal, MultivariateNormal) instance {i + 1}/{n}",
                        f"Expected (from KL MultivariateNormal): {expected}",
                        f"Actual (analytic): {actual_lowrank_full}",
                    ]
                ),
            )

            # 计算从 p_full 到 q 的 KL 散度，作为实际值
            actual_full_lowrank = kl_divergence(p_full, q)
            # 计算误差，即实际值与预期值的最大绝对差
            error_full_lowrank = torch.abs(actual_full_lowrank - expected).max()
            # 断言误差小于预设精度值 self.precision，并输出详细错误信息
            self.assertLess(
                error_full_lowrank,
                self.precision,
                "\n".join(
                    [
                        f"Incorrect KL(MultivariateNormal, LowRankMultivariateNormal) instance {i + 1}/{n}",
                        f"Expected (from KL MultivariateNormal): {expected}",
                        f"Actual (analytic): {actual_full_lowrank}",
                    ]
                ),
            )
    def test_kl_lowrank_multivariate_normal_batched(self):
        b = 7  # Number of batches
        loc = [torch.randn(b, 3) for _ in range(0, 2)]  # Generate random mean tensors for two distributions
        cov_factor = [torch.randn(b, 3, 2) for _ in range(0, 2)]  # Generate random factor tensors for two distributions
        cov_diag = [
            transform_to(constraints.positive)(torch.randn(b, 3)) for _ in range(0, 2)
        ]  # Generate random diagonal covariance tensors for two distributions, transformed to ensure positivity
        expected_kl = torch.stack(
            [
                kl_divergence(
                    LowRankMultivariateNormal(
                        loc[0][i], cov_factor[0][i], cov_diag[0][i]
                    ),  # Construct the first low-rank multivariate normal distribution
                    LowRankMultivariateNormal(
                        loc[1][i], cov_factor[1][i], cov_diag[1][i]
                    ),  # Construct the second low-rank multivariate normal distribution
                )
                for i in range(0, b)  # Iterate over each batch index
            ]
        )  # Stack KL divergences into a tensor
        actual_kl = kl_divergence(
            LowRankMultivariateNormal(loc[0], cov_factor[0], cov_diag[0]),  # Construct the batched first distribution
            LowRankMultivariateNormal(loc[1], cov_factor[1], cov_diag[1]),  # Construct the batched second distribution
        )  # Calculate KL divergence between batched distributions
        self.assertEqual(expected_kl, actual_kl)  # Assert equality of expected and actual KL divergences

    def test_kl_exponential_family(self):
        for (p, _), (_, q) in self.finite_examples:  # Iterate over pairs of distributions in finite examples
            if type(p) == type(q) and issubclass(type(p), ExponentialFamily):  # Check if both distributions are of the same type and are subclasses of ExponentialFamily
                actual = kl_divergence(p, q)  # Calculate KL divergence between p and q
                expected = _kl_expfamily_expfamily(p, q)  # Calculate expected KL divergence using Bregman Divergence
                self.assertEqual(
                    actual,
                    expected,
                    msg="\n".join(
                        [
                            f"Incorrect KL({type(p).__name__}, {type(q).__name__}).",  # Error message header
                            f"Expected (using Bregman Divergence) {expected}",  # Expected KL divergence
                            f"Actual (analytic) {actual}",  # Actual computed KL divergence
                            f"max error = {torch.abs(actual - expected).max()}",  # Maximum absolute error
                        ]
                    ),
                )  # Assert equality of expected and actual KL divergences

    def test_kl_infinite(self):
        for p, q in self.infinite_examples:  # Iterate over pairs of distributions in infinite examples
            self.assertTrue(
                (kl_divergence(p, q) == inf).all(),  # Check if KL divergence between p and q is infinity for all elements
                f"Incorrect KL({type(p).__name__}, {type(q).__name__})",  # Error message if KL divergence is not infinity
            )

    def test_kl_edgecases(self):
        self.assertEqual(kl_divergence(Bernoulli(0), Bernoulli(0)), 0)  # KL divergence between identical Bernoulli distributions is zero
        self.assertEqual(kl_divergence(Bernoulli(1), Bernoulli(1)), 0)  # KL divergence between identical Bernoulli distributions is zero
        self.assertEqual(
            kl_divergence(
                Categorical(torch.tensor([0.0, 1.0])),  # Distribution with probabilities [0.0, 1.0]
                Categorical(torch.tensor([0.0, 1.0])),  # Distribution with probabilities [0.0, 1.0]
            ),
            0,  # KL divergence between identical Categorical distributions is zero
        )
    # 定义测试函数，用于测试 KL 散度的形状
    def test_kl_shape(self):
        # 获取示例分布及其参数
        for Dist, params in _get_examples():
            # 遍历参数列表
            for i, param in enumerate(params):
                # 根据参数创建分布对象
                dist = Dist(**param)
                try:
                    # 计算同一分布之间的 KL 散度
                    kl = kl_divergence(dist, dist)
                except NotImplementedError:
                    # 如果计算 KL 散度未实现，继续下一个循环
                    continue
                # 计算期望的形状，若无批次形状则为空
                expected_shape = dist.batch_shape if dist.batch_shape else torch.Size()
                # 断言 KL 散度的形状是否符合预期
                self.assertEqual(
                    kl.shape,
                    expected_shape,
                    msg="\n".join(
                        [
                            # 错误消息，显示示例的名称、预期和实际的形状信息
                            f"{Dist.__name__} example {i + 1}/{len(params)}",
                            f"Expected {expected_shape}",
                            f"Actual {kl.shape}",
                        ]
                    ),
                )

    # 定义测试函数，用于测试变换后的 KL 散度
    def test_kl_transformed(self):
        # 修正问题：https://github.com/pytorch/pytorch/issues/34859 的回归测试
        scale = torch.ones(2, 3)
        loc = torch.zeros(2, 3)
        # 创建正态分布对象
        normal = Normal(loc=loc, scale=scale)
        # 将正态分布变为独立分布
        diag_normal = Independent(normal, reinterpreted_batch_ndims=1)
        # 创建变换分布对象
        trans_dist = TransformedDistribution(
            diag_normal, AffineTransform(loc=0.0, scale=2.0)
        )
        # 断言独立分布与自身的 KL 散度形状为 (2,)
        self.assertEqual(kl_divergence(diag_normal, diag_normal).shape, (2,))
        # 断言变换分布与自身的 KL 散度形状为 (2,)
        self.assertEqual(kl_divergence(trans_dist, trans_dist).shape, (2,))

    # 将默认数据类型设置为 double，并定义蒙特卡洛熵的测试函数
    @set_default_dtype(torch.double)
    def test_entropy_monte_carlo(self):
        set_rng_seed(0)  # 设置随机数生成器种子，见 Note [Randomized statistical tests]
        # 遍历示例分布及其参数
        for Dist, params in _get_examples():
            # 遍历参数列表
            for i, param in enumerate(params):
                # 根据参数创建分布对象
                dist = Dist(**param)
                try:
                    # 计算分布的熵
                    actual = dist.entropy()
                except NotImplementedError:
                    # 若计算熵未实现，继续下一个循环
                    continue
                # 从分布中抽样 60000 次
                x = dist.sample(sample_shape=(60000,))
                # 使用抽样数据计算熵的蒙特卡洛估计值
                expected = -dist.log_prob(x).mean(0)
                # 忽略无穷大和无穷小的值
                ignore = (expected == inf) | (expected == -inf)
                expected[ignore] = actual[ignore]
                # 断言实际熵与期望的蒙特卡洛估计值是否相等，给定容忍度 atol=0.2, rtol=0
                self.assertEqual(
                    actual,
                    expected,
                    atol=0.2,
                    rtol=0,
                    msg="\n".join(
                        [
                            # 错误消息，显示示例的名称、期望和实际的熵值信息
                            f"{Dist.__name__} example {i + 1}/{len(params)}, incorrect .entropy().",
                            f"Expected (monte carlo) {expected}",
                            f"Actual (analytic) {actual}",
                            f"max error = {torch.abs(actual - expected).max()}",
                        ]
                    ),
                )

    # 将默认数据类型设置为 double
    @set_default_dtype(torch.double)
    # 定义测试熵的指数族分布函数
    def test_entropy_exponential_family(self):
        # 获取所有指数族分布的示例及其参数
        for Dist, params in _get_examples():
            # 如果当前分布不是指数族分布，则跳过
            if not issubclass(Dist, ExponentialFamily):
                continue
            # 遍历当前分布的每个参数组合
            for i, param in enumerate(params):
                # 根据当前参数创建指数族分布对象
                dist = Dist(**param)
                try:
                    # 尝试计算当前分布的熵
                    actual = dist.entropy()
                except NotImplementedError:
                    # 如果分布对象未实现熵的计算，则继续下一个参数组合
                    continue
                try:
                    # 尝试使用指数族分布类的静态方法计算熵
                    expected = ExponentialFamily.entropy(dist)
                except NotImplementedError:
                    # 如果静态方法未实现熵的计算，则继续下一个参数组合
                    continue
                # 断言实际计算的熵与预期计算的熵相等
                self.assertEqual(
                    actual,
                    expected,
                    # 如果不相等，输出详细的错误消息
                    msg="\n".join(
                        [
                            f"{Dist.__name__} example {i + 1}/{len(params)}, incorrect .entropy().",
                            f"Expected (Bregman Divergence) {expected}",
                            f"Actual (analytic) {actual}",
                            f"max error = {torch.abs(actual - expected).max()}",
                        ]
                    ),
                )
# 定义一个测试类，继承自 DistributionsTestCase，用于测试概率分布相关的约束
class TestConstraints(DistributionsTestCase):

    # 测试参数约束的方法
    def test_params_constraints(self):
        # 定义需要标准化概率分布的类列表
        normalize_probs_dists = (
            Categorical,
            Multinomial,
            OneHotCategorical,
            OneHotCategoricalStraightThrough,
            RelaxedOneHotCategorical,
        )

        # 遍历 _get_examples() 返回的示例
        for Dist, params in _get_examples():
            # 遍历每个示例中的参数
            for i, param in enumerate(params):
                # 根据参数创建对应的分布对象
                dist = Dist(**param)
                # 遍历每个参数的名称和值
                for name, value in param.items():
                    # 如果值是数字，则转换为张量
                    if isinstance(value, numbers.Number):
                        value = torch.tensor([value])
                    # 对于标准化概率分布，确保概率值符合单纯形约束
                    if Dist in normalize_probs_dists and name == "probs":
                        # 这些分布接受正的概率值，但在其他地方我们使用更严格的单纯形约束
                        value = value / value.sum(-1, True)
                    try:
                        # 获取参数对应的约束条件
                        constraint = dist.arg_constraints[name]
                    except KeyError:
                        continue  # 忽略可选参数

                    # 检查参数的形状是否与分布的形状兼容
                    self.assertGreaterEqual(value.dim(), constraint.event_dim)
                    value_batch_shape = value.shape[
                        : value.dim() - constraint.event_dim
                    ]
                    torch.broadcast_shapes(dist.batch_shape, value_batch_shape)

                    # 如果约束是依赖性的，则跳过
                    if is_dependent(constraint):
                        continue

                    # 构造错误消息，描述参数不满足约束条件的情况
                    message = f"{Dist.__name__} example {i + 1}/{len(params)} parameter {name} = {value}"
                    # 断言参数满足约束条件
                    self.assertTrue(constraint.check(value).all(), msg=message)

    # 测试支持约束的方法
    def test_support_constraints(self):
        # 遍历 _get_examples() 返回的示例
        for Dist, params in _get_examples():
            # 确保分布的支持是一个约束对象
            self.assertIsInstance(Dist.support, Constraint)
            # 遍历每个示例中的参数
            for i, param in enumerate(params):
                # 根据参数创建对应的分布对象
                dist = Dist(**param)
                # 从分布中采样一个值
                value = dist.sample()
                # 获取分布的支持约束
                constraint = dist.support
                # 构造错误消息，描述采样值与分布事件形状的匹配情况
                message = (
                    f"{Dist.__name__} example {i + 1}/{len(params)} sample = {value}"
                )
                # 断言约束的事件维度与分布的事件形状长度相等
                self.assertEqual(
                    constraint.event_dim, len(dist.event_shape), msg=message
                )
                # 检查约束是否满足采样值的形状
                ok = constraint.check(value)
                self.assertEqual(ok.shape, dist.batch_shape, msg=message)
                # 断言约束对所有元素均满足
                self.assertTrue(ok.all(), msg=message)


# 使用装饰器跳过不适合 TorchDynamo 的测试
@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestNumericalStability(DistributionsTestCase):

    # 测试 PDF 得分的方法
    def _test_pdf_score(
        self,
        dist_class,
        x,
        expected_value,
        probs=None,
        logits=None,
        expected_gradient=None,
        atol=1e-5,
    ):
        # 如果概率不为空，则将概率值分离并要求其梯度
        if probs is not None:
            p = probs.detach().requires_grad_()
            # 使用给定的概率创建指定概率分布类的实例
            dist = dist_class(p)
        else:
            p = logits.detach().requires_grad_()
            # 使用给定的logits创建指定概率分布类的实例
            dist = dist_class(logits=p)
        # 计算对数概率密度函数
        log_pdf = dist.log_prob(x)
        # 计算对数概率密度函数的总和并进行反向传播
        log_pdf.sum().backward()
        # 断言对数概率密度函数的值是否等于预期值
        self.assertEqual(
            log_pdf,
            expected_value,
            atol=atol,
            rtol=0,
            msg=f"Incorrect value for tensor type: {type(x)}. Expected = {expected_value}, Actual = {log_pdf}",
        )
        # 如果期望的梯度不为空，则断言梯度是否等于预期梯度
        if expected_gradient is not None:
            self.assertEqual(
                p.grad,
                expected_gradient,
                atol=atol,
                rtol=0,
                msg=f"Incorrect gradient for tensor type: {type(x)}. Expected = {expected_gradient}, Actual = {p.grad}",
            )

    def test_bernoulli_gradient(self):
        # 针对不同的张量类型进行测试
        for tensor_type in [torch.FloatTensor, torch.DoubleTensor]:
            # 测试 Bernoulli 分布的概率密度函数和梯度
            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([0]),
                x=tensor_type([0]),
                expected_value=tensor_type([0]),
                expected_gradient=tensor_type([0]),
            )

            # 测试 Bernoulli 分布的概率密度函数和梯度，给定不同的输入值
            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([0]),
                x=tensor_type([1]),
                expected_value=tensor_type(
                    [torch.finfo(tensor_type([]).dtype).eps]
                ).log(),
                expected_gradient=tensor_type([0]),
            )

            # 测试 Bernoulli 分布的概率密度函数和梯度，给定不同的概率和输入值
            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([1e-4]),
                x=tensor_type([1]),
                expected_value=tensor_type([math.log(1e-4)]),
                expected_gradient=tensor_type([10000]),
            )

            # 由于精度较低，测试 Bernoulli 分布的概率密度函数和梯度，给定不同的概率和输入值
            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([1 - 1e-4]),
                x=tensor_type([0]),
                expected_value=tensor_type([math.log(1e-4)]),
                expected_gradient=tensor_type([-10000]),
                atol=2,
            )

            # 测试 Bernoulli 分布的概率密度函数和梯度，给定 logits 和输入值
            self._test_pdf_score(
                dist_class=Bernoulli,
                logits=tensor_type([math.log(9999)]),
                x=tensor_type([0]),
                expected_value=tensor_type([math.log(1e-4)]),
                expected_gradient=tensor_type([-1]),
                atol=1e-3,
            )
    # 定义测试函数，用于测试 Bernoulli 分布在 logits 下溢的情况
    def test_bernoulli_with_logits_underflow(self):
        # 遍历不同的张量类型和极限值
        for tensor_type, lim in [
            (torch.FloatTensor, -1e38),
            (torch.DoubleTensor, -1e308),
        ]:
            # 调用内部方法 _test_pdf_score 测试 Bernoulli 分布的概率密度函数得分
            self._test_pdf_score(
                dist_class=Bernoulli,
                logits=tensor_type([lim]),  # 设置 logits 的值为 lim
                x=tensor_type([0]),  # 设置输入 x 的值为 [0]
                expected_value=tensor_type([0]),  # 期望的值为 [0]
                expected_gradient=tensor_type([0]),  # 期望的梯度为 [0]
            )

    # 定义测试函数，用于测试 Bernoulli 分布在 logits 上溢的情况
    def test_bernoulli_with_logits_overflow(self):
        # 遍历不同的张量类型和极限值
        for tensor_type, lim in [
            (torch.FloatTensor, 1e38),
            (torch.DoubleTensor, 1e308),
        ]:
            # 调用内部方法 _test_pdf_score 测试 Bernoulli 分布的概率密度函数得分
            self._test_pdf_score(
                dist_class=Bernoulli,
                logits=tensor_type([lim]),  # 设置 logits 的值为 lim
                x=tensor_type([1]),  # 设置输入 x 的值为 [1]
                expected_value=tensor_type([0]),  # 期望的值为 [0]
                expected_gradient=tensor_type([0]),  # 期望的梯度为 [0]
            )

    # 定义测试函数，用于测试 categorical 分布的对数概率
    def test_categorical_log_prob(self):
        # 遍历不同的数据类型
        for dtype in [torch.float, torch.double]:
            # 创建一个张量 p，包含 [0, 1]，设置为需要计算梯度
            p = torch.tensor([0, 1], dtype=dtype, requires_grad=True)
            # 创建 OneHotCategorical 分布对象
            categorical = OneHotCategorical(p)
            # 计算给定输入 [0, 1] 的对数概率密度
            log_pdf = categorical.log_prob(torch.tensor([0, 1], dtype=dtype))
            # 断言对数概率是否等于 0
            self.assertEqual(log_pdf.item(), 0)

    # 定义测试函数，用于测试具有 logits 的 categorical 分布的对数概率
    def test_categorical_log_prob_with_logits(self):
        # 遍历不同的数据类型
        for dtype in [torch.float, torch.double]:
            # 创建一个张量 p，包含 [-inf, 0]，设置为需要计算梯度
            p = torch.tensor([-float('inf'), 0], dtype=dtype, requires_grad=True)
            # 创建 logits 形式的 OneHotCategorical 分布对象
            categorical = OneHotCategorical(logits=p)
            # 计算给定输入 [0, 1] 的对数概率密度
            log_pdf_prob_1 = categorical.log_prob(torch.tensor([0, 1], dtype=dtype))
            # 断言对数概率是否等于 0
            self.assertEqual(log_pdf_prob_1.item(), 0)
            # 计算给定输入 [1, 0] 的对数概率密度
            log_pdf_prob_0 = categorical.log_prob(torch.tensor([1, 0], dtype=dtype))
            # 断言对数概率是否为负无穷
            self.assertEqual(log_pdf_prob_0.item(), -float('inf'))

    # 定义测试函数，用于测试 multinomial 分布的对数概率
    def test_multinomial_log_prob(self):
        # 遍历不同的数据类型
        for dtype in [torch.float, torch.double]:
            # 创建一个张量 p，包含 [0, 1]，设置为需要计算梯度
            p = torch.tensor([0, 1], dtype=dtype, requires_grad=True)
            # 创建 Multinomial 分布对象，总样本数为 10
            multinomial = Multinomial(10, p)
            # 计算给定样本 s=[0, 10] 的对数概率密度
            log_pdf = multinomial.log_prob(torch.tensor([0, 10], dtype=dtype))
            # 断言对数概率是否等于 0
            self.assertEqual(log_pdf.item(), 0)

    # 定义测试函数，用于测试具有 logits 的 multinomial 分布的对数概率
    def test_multinomial_log_prob_with_logits(self):
        # 遍历不同的数据类型
        for dtype in [torch.float, torch.double]:
            # 创建一个张量 p，包含 [-inf, 0]，设置为需要计算梯度
            p = torch.tensor([-float('inf'), 0], dtype=dtype, requires_grad=True)
            # 创建 logits 形式的 Multinomial 分布对象，总样本数为 10
            multinomial = Multinomial(10, logits=p)
            # 计算给定样本 s=[0, 10] 的对数概率密度
            log_pdf_prob_1 = multinomial.log_prob(torch.tensor([0, 10], dtype=dtype))
            # 断言对数概率是否等于 0
            self.assertEqual(log_pdf_prob_1.item(), 0)
            # 计算给定样本 s=[10, 0] 的对数概率密度
            log_pdf_prob_0 = multinomial.log_prob(torch.tensor([10, 0], dtype=dtype))
            # 断言对数概率是否为负无穷
            self.assertEqual(log_pdf_prob_0.item(), -float('inf'))
    # 定义测试连续伯努利分布在 logits 下溢的情况
    def test_continuous_bernoulli_with_logits_underflow(self):
        # 遍历不同的张量类型、界限值和期望结果
        for tensor_type, lim, expected in [
            (torch.FloatTensor, -1e38, 2.76898),  # 使用单精度浮点张量，界限为 -1e38，期望值为 2.76898
            (torch.DoubleTensor, -1e308, 3.58473),  # 使用双精度浮点张量，界限为 -1e308，期望值为 3.58473
        ]:
            # 调用 _test_pdf_score 方法进行 PDF 评分测试
            self._test_pdf_score(
                dist_class=ContinuousBernoulli,  # 指定分布类为连续伯努利分布
                logits=tensor_type([lim]),  # 设置 logits 为指定张量类型和界限值的张量
                x=tensor_type([0]),  # 设置输入 x 为张量类型的 0
                expected_value=tensor_type([expected]),  # 设置期望值为指定张量类型和期望结果的张量
                expected_gradient=tensor_type([0.0]),  # 设置期望梯度为指定张量类型的 0.0
            )

    # 定义测试连续伯努利分布在 logits 上溢的情况
    def test_continuous_bernoulli_with_logits_overflow(self):
        # 遍历不同的张量类型、界限值和期望结果
        for tensor_type, lim, expected in [
            (torch.FloatTensor, 1e38, 2.76898),  # 使用单精度浮点张量，界限为 1e38，期望值为 2.76898
            (torch.DoubleTensor, 1e308, 3.58473),  # 使用双精度浮点张量，界限为 1e308，期望值为 3.58473
        ]:
            # 调用 _test_pdf_score 方法进行 PDF 评分测试
            self._test_pdf_score(
                dist_class=ContinuousBernoulli,  # 指定分布类为连续伯努利分布
                logits=tensor_type([lim]),  # 设置 logits 为指定张量类型和界限值的张量
                x=tensor_type([1]),  # 设置输入 x 为张量类型的 1
                expected_value=tensor_type([expected]),  # 设置期望值为指定张量类型和期望结果的张量
                expected_gradient=tensor_type([0.0]),  # 设置期望梯度为指定张量类型的 0.0
            )
# TODO: make this a pytest parameterized test
# 创建一个测试类 TestLazyLogitsInitialization，继承自 DistributionsTestCase
class TestLazyLogitsInitialization(DistributionsTestCase):
    
    # 在每个测试方法运行前执行的设置方法
    def setUp(self):
        super().setUp()
        # 从 _get_examples() 中获取例子，筛选出需要测试的示例
        # 要求 e.Dist 在 (Categorical, OneHotCategorical, Bernoulli, Binomial, Multinomial) 中
        self.examples = [
            e
            for e in _get_examples()
            if e.Dist in (Categorical, OneHotCategorical, Bernoulli, Binomial, Multinomial)
        ]

    # 测试延迟 logits 初始化的方法
    def test_lazy_logits_initialization(self):
        # 遍历所有示例
        for Dist, params in self.examples:
            # 拷贝第一个参数，并进行判断
            param = params[0].copy()
            if "probs" not in param:
                continue
            # 从参数中移除 'probs' 并计算 'logits'
            probs = param.pop("probs")
            param["logits"] = probs_to_logits(probs)
            # 使用参数初始化分布对象
            dist = Dist(**param)
            # 生成一个有效样本，以创建一个新的实例
            dist.log_prob(Dist(**param).sample())
            # 消息字符串，指示测试失败的具体分布和示例索引
            message = f"Failed for {Dist.__name__} example 0/{len(params)}"
            # 确保分布对象的 __dict__ 不包含 'probs'
            self.assertNotIn("probs", dist.__dict__, msg=message)
            # 尝试调用 enumerate_support 方法，捕获 NotImplementedError 异常
            try:
                dist.enumerate_support()
            except NotImplementedError:
                pass
            # 再次确认分布对象的 __dict__ 中不包含 'probs'
            self.assertNotIn("probs", dist.__dict__, msg=message)
            # 获取分布对象的 batch_shape 和 event_shape 属性
            batch_shape, event_shape = dist.batch_shape, dist.event_shape
            # 确保分布对象的 __dict__ 中不包含 'probs'
            self.assertNotIn("probs", dist.__dict__, msg=message)

    # 测试延迟 probs 初始化的方法
    def test_lazy_probs_initialization(self):
        # 遍历所有示例
        for Dist, params in self.examples:
            # 拷贝第一个参数，并进行判断
            param = params[0].copy()
            if "probs" not in param:
                continue
            # 使用参数初始化分布对象
            dist = Dist(**param)
            # 生成一个样本
            dist.sample()
            # 消息字符串，指示测试失败的具体分布和示例索引
            message = f"Failed for {Dist.__name__} example 0/{len(params)}"
            # 确保分布对象的 __dict__ 不包含 'logits'
            self.assertNotIn("logits", dist.__dict__, msg=message)
            # 尝试调用 enumerate_support 方法，捕获 NotImplementedError 异常
            try:
                dist.enumerate_support()
            except NotImplementedError:
                pass
            # 再次确认分布对象的 __dict__ 中不包含 'logits'
            self.assertNotIn("logits", dist.__dict__, msg=message)
            # 获取分布对象的 batch_shape 和 event_shape 属性
            batch_shape, event_shape = dist.batch_shape, dist.event_shape
            # 确保分布对象的 __dict__ 中不包含 'logits'
            self.assertNotIn("logits", dist.__dict__, msg=message)


# 使用 unittest 装饰器，如果未找到 NumPy，跳过该测试类
@unittest.skipIf(not TEST_NUMPY, "NumPy not found")
# 使用 skipIfTorchDynamo 装饰器，如果在 TorchDynamo 环境中，跳过该测试类
@skipIfTorchDynamo("FIXME: Tries to trace through SciPy and fails")
# 创建一个测试类 TestAgainstScipy，继承自 DistributionsTestCase
class TestAgainstScipy(DistributionsTestCase):
    
    # 测试均值的方法
    def test_mean(self):
        # 遍历分布对
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            # 对于 Cauchy 和 HalfCauchy 分布，均值为 NaN，跳过检查
            if isinstance(pytorch_dist, (Cauchy, HalfCauchy)):
                continue
            # 对于 LowRankMultivariateNormal 和 MultivariateNormal 分布，比较均值
            elif isinstance(pytorch_dist, (LowRankMultivariateNormal, MultivariateNormal)):
                self.assertEqual(pytorch_dist.mean, scipy_dist.mean, msg=pytorch_dist)
            # 对于其他分布，比较均值
            else:
                self.assertEqual(pytorch_dist.mean, scipy_dist.mean(), msg=pytorch_dist)
    # 定义一个测试函数，用于测试方差和标准差计算的准确性
    def test_variance_stddev(self):
        # 遍历分布对的列表，每个对包含 PyTorch 分布和对应的 SciPy 分布
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            # 如果是 Cauchy、HalfCauchy 或 VonMises 分布，则它们的标准差是 NaN，跳过检查
            if isinstance(pytorch_dist, (Cauchy, HalfCauchy, VonMises)):
                continue  # 跳过当前循环，进入下一次循环
            # 如果是 Multinomial 或 OneHotCategorical 分布
            elif isinstance(pytorch_dist, (Multinomial, OneHotCategorical)):
                # 检查 PyTorch 分布的方差是否与 scipy 分布的协方差对角线一致
                self.assertEqual(
                    pytorch_dist.variance, np.diag(scipy_dist.cov()), msg=pytorch_dist
                )
                # 检查 PyTorch 分布的标准差是否与 scipy 分布的协方差对角线开平方一致
                self.assertEqual(
                    pytorch_dist.stddev,
                    np.diag(scipy_dist.cov()) ** 0.5,
                    msg=pytorch_dist,
                )
            # 如果是 LowRankMultivariateNormal 或 MultivariateNormal 分布
            elif isinstance(
                pytorch_dist, (LowRankMultivariateNormal, MultivariateNormal)
            ):
                # 检查 PyTorch 分布的方差是否与 scipy 分布的协方差对角线一致
                self.assertEqual(
                    pytorch_dist.variance, np.diag(scipy_dist.cov), msg=pytorch_dist
                )
                # 检查 PyTorch 分布的标准差是否与 scipy 分布的协方差对角线开平方一致
                self.assertEqual(
                    pytorch_dist.stddev,
                    np.diag(scipy_dist.cov) ** 0.5,
                    msg=pytorch_dist,
                )
            else:
                # 对于其他类型的分布，检查 PyTorch 分布的方差是否与 scipy 分布的方差一致
                self.assertEqual(
                    pytorch_dist.variance, scipy_dist.var(), msg=pytorch_dist
                )
                # 检查 PyTorch 分布的标准差是否与 scipy 分布的标准差一致
                self.assertEqual(
                    pytorch_dist.stddev, scipy_dist.var() ** 0.5, msg=pytorch_dist
                )

    # 设置默认数据类型为双精度浮点数，并定义一个测试累积分布函数的方法
    @set_default_dtype(torch.double)
    def test_cdf(self):
        # 遍历分布对的列表，每个对包含 PyTorch 分布和对应的 SciPy 分布
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            # 从 PyTorch 分布中抽样生成样本
            samples = pytorch_dist.sample((5,))
            try:
                # 尝试计算 PyTorch 分布的累积分布函数（CDF）
                cdf = pytorch_dist.cdf(samples)
            except NotImplementedError:
                continue  # 如果不支持，则跳过当前循环，进入下一次循环
            # 检查 PyTorch 分布计算得到的 CDF 是否与 scipy 分布计算得到的 CDF 一致
            self.assertEqual(cdf, scipy_dist.cdf(samples), msg=pytorch_dist)

    # 定义一个测试逆累积分布函数的方法
    def test_icdf(self):
        # 遍历分布对的列表，每个对包含 PyTorch 分布和对应的 SciPy 分布
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            # 生成随机样本，样本的形状为 (5, 分布的批量形状)
            samples = torch.rand((5,) + pytorch_dist.batch_shape, dtype=torch.double)
            try:
                # 尝试计算 PyTorch 分布的逆累积分布函数（ICDF）
                icdf = pytorch_dist.icdf(samples)
            except NotImplementedError:
                continue  # 如果不支持，则跳过当前循环，进入下一次循环
            # 检查 PyTorch 分布计算得到的 ICDF 是否与 scipy 分布计算得到的 PPF（百分点函数）一致
            self.assertEqual(icdf, scipy_dist.ppf(samples), msg=pytorch_dist)
# 创建一个名为 TestFunctors 的测试类，继承自 DistributionsTestCase
class TestFunctors(DistributionsTestCase):
    
    # 定义一个测试方法 test_cat_transform
    def test_cat_transform(self):
        
        # 生成一个 1 到 100 之间的递减序列，并转换为 torch 的浮点数张量 x1
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        
        # 生成一个 0 到 99 之间的递增序列，并将其归一化后作为 x2
        x2 = (torch.arange(1, 101, dtype=torch.float).view(-1, 100) - 1) / 100
        
        # 生成一个 1 到 100 的递增序列作为 x3
        x3 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        
        # 创建三个变换对象 t1, t2, t3，分别为指数变换、仿射变换和恒等变换
        t1, t2, t3 = ExpTransform(), AffineTransform(1, 100), identity_transform
        
        # 设定维度为 0，将 x1, x2, x3 沿指定维度拼接成张量 x
        dim = 0
        x = torch.cat([x1, x2, x3], dim=dim)
        
        # 创建一个 CatTransform 对象 t，将 t1, t2, t3 沿指定维度拼接成变换序列
        t = CatTransform([t1, t2, t3], dim=dim)
        
        # 计算 t 对 x 的作用结果并检查定义域的一致性
        actual_dom_check = t.domain.check(x)
        
        # 生成期望的定义域检查结果，将 t1, t2, t3 分别应用于 x1, x2, x3 后拼接成张量
        expected_dom_check = torch.cat(
            [t1.domain.check(x1), t2.domain.check(x2), t3.domain.check(x3)], dim=dim
        )
        
        # 断言实际的定义域检查结果与期望的一致
        self.assertEqual(expected_dom_check, actual_dom_check)
        
        # 计算 t 对 x 的作用结果，并期望与将 t1, t2, t3 分别应用于 x1, x2, x3 后拼接成的张量相等
        actual = t(x)
        expected = torch.cat([t1(x1), t2(x2), t3(x3)], dim=dim)
        self.assertEqual(expected, actual)
        
        # 生成三个递增序列 y1, y2, y3，并将它们沿指定维度拼接成张量 y
        y1 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y2 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y3 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y = torch.cat([y1, y2, y3], dim=dim)
        
        # 计算 t 对 y 的逆作用结果并检查值域的一致性
        actual_cod_check = t.codomain.check(y)
        
        # 生成期望的值域检查结果，将 t1, t2, t3 分别应用于 y1, y2, y3 后拼接成张量
        expected_cod_check = torch.cat(
            [t1.codomain.check(y1), t2.codomain.check(y2), t3.codomain.check(y3)],
            dim=dim,
        )
        
        # 断言实际的值域检查结果与期望的一致
        self.assertEqual(actual_cod_check, expected_cod_check)
        
        # 计算 t 对 y 的逆作用结果，并期望与将 t1, t2, t3 分别应用于 y1, y2, y3 后拼接成的张量相等
        actual_inv = t.inv(y)
        expected_inv = torch.cat([t1.inv(y1), t2.inv(y2), t3.inv(y3)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        
        # 计算 t 对 x, y 的雅可比对数绝对值行列式并检查结果
        actual_jac = t.log_abs_det_jacobian(x, y)
        
        # 生成期望的雅可比对数绝对值行列式结果，将 t1, t2, t3 分别应用于 x1, x2, x3 和 y1, y2, y3 后拼接成张量
        expected_jac = torch.cat(
            [
                t1.log_abs_det_jacobian(x1, y1),
                t2.log_abs_det_jacobian(x2, y2),
                t3.log_abs_det_jacobian(x3, y3),
            ],
            dim=dim,
        )
        
        # 断言实际的雅可比对数绝对值行列式结果与期望的一致
        self.assertEqual(actual_jac, expected_jac)
    # 定义测试方法：测试非均匀拼接变换
    def test_cat_transform_non_uniform(self):
        # 创建第一个输入张量 x1，包含从 -1 到 -100 的浮点数序列，reshape 成 1x100 的张量
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        # 创建第二个输入张量 x2，通过拼接两个张量得到：第一个部分是 (0 到 99) / 100，第二个部分是 1 到 100
        x2 = torch.cat(
            [
                (torch.arange(1, 101, dtype=torch.float).view(-1, 100) - 1) / 100,
                torch.arange(1, 101, dtype=torch.float).view(-1, 100),
            ]
        )
        # 创建指数变换对象 t1
        t1 = ExpTransform()
        # 创建拼接变换对象 t2，包含 AffineTransform 和 identity_transform
        t2 = CatTransform([AffineTransform(1, 100), identity_transform], dim=0)
        # 设置拼接的维度为 0
        dim = 0
        # 将 x1 和 x2 按照指定维度 dim 进行拼接，得到张量 x
        x = torch.cat([x1, x2], dim=dim)
        # 创建拼接变换对象 t，包含 t1 和 t2，各自作用在不同长度的部分上
        t = CatTransform([t1, t2], dim=dim, lengths=[1, 2])
        # 检查输入 x 的定义域，得到实际的定义域检查结果
        actual_dom_check = t.domain.check(x)
        # 预期的定义域检查结果，通过分别检查 t1 在 x1 和 t2 在 x2 上的定义域来得到
        expected_dom_check = torch.cat(
            [t1.domain.check(x1), t2.domain.check(x2)], dim=dim
        )
        # 断言实际的定义域检查结果与预期的一致
        self.assertEqual(expected_dom_check, actual_dom_check)
        # 对输入 x 应用拼接变换 t，得到实际的输出结果 actual
        actual = t(x)
        # 通过分别对 x1 和 x2 应用 t1 和 t2 来得到预期的输出结果 expected
        expected = torch.cat([t1(x1), t2(x2)], dim=dim)
        # 断言实际的输出结果与预期的一致
        self.assertEqual(expected, actual)
        # 创建第一个输出张量 y1，包含从 1 到 100 的浮点数序列，reshape 成 1x100 的张量
        y1 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        # 创建第二个输出张量 y2，通过拼接两个张量得到：两个相同的 1 到 100 的浮点数序列
        y2 = torch.cat(
            [
                torch.arange(1, 101, dtype=torch.float).view(-1, 100),
                torch.arange(1, 101, dtype=torch.float).view(-1, 100),
            ]
        )
        # 将 y1 和 y2 按照指定维度 dim 进行拼接，得到张量 y
        y = torch.cat([y1, y2], dim=dim)
        # 检查输出 y 的值域，得到实际的值域检查结果
        actual_cod_check = t.codomain.check(y)
        # 预期的值域检查结果，通过分别检查 t1 在 y1 和 t2 在 y2 上的值域来得到
        expected_cod_check = torch.cat(
            [t1.codomain.check(y1), t2.codomain.check(y2)], dim=dim
        )
        # 断言实际的值域检查结果与预期的一致
        self.assertEqual(actual_cod_check, expected_cod_check)
        # 对输出 y 应用拼接变换 t 的逆变换，得到实际的逆变换结果 actual_inv
        actual_inv = t.inv(y)
        # 通过分别对 y1 和 y2 应用 t1 和 t2 的逆变换来得到预期的逆变换结果 expected_inv
        expected_inv = torch.cat([t1.inv(y1), t2.inv(y2)], dim=dim)
        # 断言实际的逆变换结果与预期的一致
        self.assertEqual(expected_inv, actual_inv)
        # 计算拼接变换 t 在输入 x 和输出 y 上的对数绝对值行列式的雅可比矩阵，得到实际的雅可比矩阵结果 actual_jac
        actual_jac = t.log_abs_det_jacobian(x, y)
        # 通过分别对 x1 和 y1 以及 x2 和 y2 计算 t1 和 t2 的对数绝对值行列式的雅可比矩阵来得到预期的雅可比矩阵结果 expected_jac
        expected_jac = torch.cat(
            [t1.log_abs_det_jacobian(x1, y1), t2.log_abs_det_jacobian(x2, y2)], dim=dim
        )
        # 断言实际的雅可比矩阵结果与预期的一致
        self.assertEqual(actual_jac, expected_jac)

    # 定义测试方法：测试拼接变换对象的事件维度
    def test_cat_event_dim(self):
        # 创建第一个仿射变换对象 t1，其事件维度为 1
        t1 = AffineTransform(0, 2 * torch.ones(2), event_dim=1)
        # 创建第二个仿射变换对象 t2，其事件维度为 1
        t2 = AffineTransform(0, 2 * torch.ones(2), event_dim=1)
        # 设置拼接的维度为 1
        dim = 1
        # 设置 batch size 为 16
        bs = 16
        # 创建第一个输入张量 x1，形状为 (16, 2) 的随机张量
        x1 = torch.randn(bs, 2)
        # 创建第二个输入张量 x2，形状为 (16, 2) 的随机张量
        x2 = torch.randn(bs, 2)
        # 将 x1 和 x2 按照指定维度 dim 进行拼接，得到张量 x
        x = torch.cat([x1, x2], dim=1)
        # 创建拼接变换对象 t，包含 t1 和 t2，各自作用在长度为 2 的部分上
        t = CatTransform([t1, t2], dim=dim, lengths=[2, 2])
        # 分别对 x1 和 x2 应用 t1 和 t2，得到 y1 和 y2
        y1 = t1(x1)
        y2 = t2(x2)
        # 对输入 x 应用拼接变换 t，得到张量 y
        y = t(x)
        # 计算拼接变换 t 在输入 x 和输出 y 上的对数绝对值行列式的雅可比矩阵，得到实际的雅可比矩阵结果 actual_jac
        actual_jac = t.log_abs_det_jacobian(x, y)
        # 计算 t1 和 t2 分别在 x1 和 y1 以及 x2 和 y2 上的对数绝对值行列式的雅可比矩阵，将结果相加得到预期的雅可比矩阵结果 expected_jac
        expected_jac = sum(
            [t1.log_abs_det_jacobian(x1, y1), t2.log_abs_det_jacobian(x2, y2)]
        )
    # 定义一个测试函数，用于测试堆叠变换的功能
    def test_stack_transform(self):
        # 创建三个张量：x1为-1乘以从1到100的浮点数张量
        x1 = -1 * torch.arange(1, 101, dtype=torch.float)
        # 创建三个张量：x2为从0到1的百分比值张量
        x2 = (torch.arange(1, 101, dtype=torch.float) - 1) / 100
        # 创建三个张量：x3为从1到100的浮点数张量
        x3 = torch.arange(1, 101, dtype=torch.float)
        # 创建三个变换对象：t1为指数变换，t2为仿射变换(1, 100)，t3为恒等变换
        t1, t2, t3 = ExpTransform(), AffineTransform(1, 100), identity_transform
        # 指定堆叠的维度为0，将x1、x2、x3堆叠成一个张量x
        dim = 0
        x = torch.stack([x1, x2, x3], dim=dim)
        # 创建一个堆叠变换对象t，包含t1、t2、t3三个变换对象，指定堆叠维度为0
        t = StackTransform([t1, t2, t3], dim=dim)
        # 检查堆叠变换对象t对输入张量x的定义域检查结果
        actual_dom_check = t.domain.check(x)
        # 期望的定义域检查结果，分别对应t1、t2、t3对x1、x2、x3的检查结果，堆叠维度为0
        expected_dom_check = torch.stack(
            [t1.domain.check(x1), t2.domain.check(x2), t3.domain.check(x3)], dim=dim
        )
        # 断言实际的定义域检查结果与期望的检查结果相等
        self.assertEqual(expected_dom_check, actual_dom_check)
        # 对输入张量x应用堆叠变换t，得到实际的变换结果
        actual = t(x)
        # 期望的堆叠变换结果，分别对应t1、t2、t3对x1、x2、x3的变换结果，堆叠维度为0
        expected = torch.stack([t1(x1), t2(x2), t3(x3)], dim=dim)
        # 断言实际的变换结果与期望的变换结果相等
        self.assertEqual(expected, actual)
        # 创建三个张量：y1、y2、y3分别为从1到100的浮点数张量
        y1 = torch.arange(1, 101, dtype=torch.float)
        y2 = torch.arange(1, 101, dtype=torch.float)
        y3 = torch.arange(1, 101, dtype=torch.float)
        # 将y1、y2、y3堆叠成一个张量y，堆叠维度为0
        y = torch.stack([y1, y2, y3], dim=dim)
        # 检查堆叠变换对象t对输入张量y的值域检查结果
        actual_cod_check = t.codomain.check(y)
        # 期望的值域检查结果，分别对应t1、t2、t3对y1、y2、y3的检查结果，堆叠维度为0
        expected_cod_check = torch.stack(
            [t1.codomain.check(y1), t2.codomain.check(y2), t3.codomain.check(y3)],
            dim=dim,
        )
        # 断言实际的值域检查结果与期望的检查结果相等
        self.assertEqual(actual_cod_check, expected_cod_check)
        # 对输入张量x应用堆叠变换t的逆变换，得到实际的逆变换结果
        actual_inv = t.inv(x)
        # 期望的逆变换结果，分别对应t1、t2、t3对x1、x2、x3的逆变换结果，堆叠维度为0
        expected_inv = torch.stack([t1.inv(x1), t2.inv(x2), t3.inv(x3)], dim=dim)
        # 断言实际的逆变换结果与期望的逆变换结果相等
        self.assertEqual(expected_inv, actual_inv)
        # 计算输入张量x到输出张量y的对数绝对行列式的雅可比行列式的对数，得到实际的雅可比行列式
        actual_jac = t.log_abs_det_jacobian(x, y)
        # 期望的雅可比行列式，分别对应t1、t2、t3对x1、x2、x3到y1、y2、y3的雅可比行列式，堆叠维度为0
        expected_jac = torch.stack(
            [
                t1.log_abs_det_jacobian(x1, y1),
                t2.log_abs_det_jacobian(x2, y2),
                t3.log_abs_det_jacobian(x3, y3),
            ],
            dim=dim,
        )
        # 断言实际的雅可比行列式与期望的雅可比行列式相等
        self.assertEqual(actual_jac, expected_jac)
# 定义一个测试类 TestValidation，继承自 DistributionsTestCase
class TestValidation(DistributionsTestCase):
    
    # 定义一个测试方法 test_valid，用于测试参数有效性
    def test_valid(self):
        # 遍历 _get_examples() 返回的分布类 Dist 和参数 params
        for Dist, params in _get_examples():
            # 对每个参数组合进行测试
            for param in params:
                # 创建分布对象，开启参数验证功能，并使用给定参数初始化
                Dist(validate_args=True, **param)

    # 使用装饰器设置默认数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    # 定义一个测试方法 test_invalid_log_probs_arg，用于测试参数无效时的行为
    def test_invalid_log_probs_arg(self):
        # 检查验证错误确实已禁用，但可能会引发其他错误
        # 遍历 _get_examples() 返回的分布类 Dist 和参数 params
        for Dist, params in _get_examples():
            # 对于 TransformedDistribution 类型的分布，由于其参数是另一个分布实例，无法进行验证处理
            if Dist == TransformedDistribution:
                continue
            # 遍历每个参数组合
            for i, param in enumerate(params):
                # 创建禁用参数验证的分布实例 d_nonval
                d_nonval = Dist(validate_args=False, **param)
                # 创建启用参数验证的分布实例 d_val
                d_val = Dist(validate_args=True, **param)
                
                # 对于一组错误形状的样本，应该仅引发 ValueError
                for v in torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]):
                    try:
                        log_prob = d_val.log_prob(v)
                    except ValueError:
                        pass
                    
                    # 获取正确形状的样本
                    val = torch.full(d_val.batch_shape + d_val.event_shape, v)
                    
                    # 检查具有错误支持的样本
                    try:
                        log_prob = d_val.log_prob(val)
                    except ValueError as e:
                        # 如果错误信息中包含 "must be within the support"，则尝试对禁用验证的分布进行计算
                        if e.args and "must be within the support" in e.args[0]:
                            try:
                                log_prob = d_nonval.log_prob(val)
                            except RuntimeError:
                                pass
                
                # 检查正确的样本是否能够正常处理
                valid_value = d_val.sample()
                d_val.log_prob(valid_value)
                
                # 检查无效值是否会引发 ValueError
                if valid_value.dtype == torch.long:
                    valid_value = valid_value.float()
                invalid_value = torch.full_like(valid_value, math.nan)
                try:
                    # 使用断言检查是否会引发指定的 ValueError 异常
                    with self.assertRaisesRegex(
                        ValueError,
                        "Expected value argument .* to be within the support .*",
                    ):
                        d_val.log_prob(invalid_value)
                except AssertionError as e:
                    # 如果断言失败，抛出详细的 AssertionError
                    fail_string = "Support ValueError not raised for {} example {}/{}"
                    raise AssertionError(
                        fail_string.format(Dist.__name__, i + 1, len(params))
                    ) from e

    # 再次使用装饰器设置默认数据类型为双精度浮点数
    @set_default_dtype(torch.double)
    # 定义一个测试方法，用于测试处理无效情况
    def test_invalid(self):
        # 遍历获取无效示例的分布类型和参数组合
        for Dist, params in _get_bad_examples():
            # 遍历参数列表
            for i, param in enumerate(params):
                try:
                    # 断言期望抛出值错误异常
                    with self.assertRaises(ValueError):
                        # 使用给定的参数创建分布对象，同时验证参数为真
                        Dist(validate_args=True, **param)
                except AssertionError as e:
                    fail_string = "ValueError not raised for {} example {}/{}"
                    # 抛出新的断言错误，指明未正确引发异常的情况
                    raise AssertionError(
                        fail_string.format(Dist.__name__, i + 1, len(params))
                    ) from e

    # 定义一个测试方法，用于测试未实现约束条件时的警告
    def test_warning_unimplemented_constraints(self):
        # 定义一个 Delta 类，继承自 Distribution 类
        class Delta(Distribution):
            # 初始化方法
            def __init__(self, validate_args=True):
                super().__init__(validate_args=validate_args)

            # 采样方法，返回一个零张量
            def sample(self, sample_shape=torch.Size()):
                return torch.tensor(0.0).expand(sample_shape)

            # 对数概率方法，根据验证参数决定是否验证样本
            def log_prob(self, value):
                if self._validate_args:
                    self._validate_sample(value)
                # 将不为零的值设为负无穷，零值设为零
                value[value != 0.0] = -float("inf")
                value[value == 0.0] = 0.0
                return value

        # 断言期望引发用户警告
        with self.assertWarns(UserWarning):
            # 创建 Delta 类的实例对象
            d = Delta()
        # 对 Delta 实例对象进行采样，期望引发用户警告
        sample = d.sample((2,))
        with self.assertWarns(UserWarning):
            # 对 Delta 实例对象的采样结果进行对数概率计算，期望引发用户警告
            d.log_prob(sample)
# 定义名为 TestJit 的类，继承自 DistributionsTestCase 类
class TestJit(DistributionsTestCase):

    # 定义名为 _examples 的方法
    def _examples(self):
        # 通过 _get_examples() 获取分布和参数的示例
        for Dist, params in _get_examples():
            # 遍历每个参数
            for param in params:
                # 获取参数的键（即参数名）
                keys = param.keys()
                # 获取参数的值，并将其转换为元组
                values = tuple(param[key] for key in keys)
                # 如果所有值均为 torch.Tensor 类型，则执行以下操作
                if not all(isinstance(x, torch.Tensor) for x in values):
                    continue
                # 使用参数初始化分布对象 Dist，并生成一个样本
                sample = Dist(**param).sample()
                # 生成器返回分布对象、参数键、参数值元组和生成的样本
                yield Dist, keys, values, sample

    # 定义名为 _perturb_tensor 的方法，用于对张量进行扰动
    def _perturb_tensor(self, value, constraint):
        # 如果约束类型是 IntegerGreaterThan，则将值增加 1
        if isinstance(constraint, constraints._IntegerGreaterThan):
            return value + 1
        # 如果约束类型是 PositiveDefinite 或 PositiveSemidefinite，则增加单位矩阵
        if isinstance(
            constraint,
            (constraints._PositiveDefinite, constraints._PositiveSemidefinite),
        ):
            return value + torch.eye(value.shape[-1])
        # 如果值的数据类型是浮点型，则进行变换操作
        if value.dtype in [torch.float, torch.double]:
            # 根据约束转换值，并添加一个新的正态分布噪声 delta
            transform = transform_to(constraint)
            delta = value.new(value.shape).normal_()
            return transform(transform.inv(value) + delta)
        # 如果值的数据类型是长整型，则对其进行特定的取反处理
        if value.dtype == torch.long:
            result = value.clone()
            result[value == 0] = 1
            result[value == 1] = 0
            return result
        # 抛出未实现的错误，如果数据类型不是上述类型之一
        raise NotImplementedError

    # 定义名为 _perturb 的方法，用于对分布进行扰动
    def _perturb(self, Dist, keys, values, sample):
        # 使用 torch.no_grad() 禁用梯度计算
        with torch.no_grad():
            # 如果分布是均匀分布 Uniform
            if Dist is Uniform:
                # 创建参数字典，将原始参数的低端和高端各增加随机数
                param = dict(zip(keys, values))
                param["low"] = param["low"] - torch.rand(param["low"].shape)
                param["high"] = param["high"] + torch.rand(param["high"].shape)
                values = [param[key] for key in keys]
            else:
                # 对每个参数键值对进行扰动，使用 Dist 的参数约束
                values = [
                    self._perturb_tensor(
                        value, Dist.arg_constraints.get(key, constraints.real)
                    )
                    for key, value in zip(keys, values)
                ]
            # 创建新的参数字典，根据扰动后的值重新生成样本
            param = dict(zip(keys, values))
            sample = Dist(**param).sample()
            # 返回扰动后的参数值列表和生成的新样本
            return values, sample

    # 使用 torch.double 设置默认的张量数据类型
    @set_default_dtype(torch.double)
    # 定义测试方法 test_sample，用于测试样本采样函数的正确性
    def test_sample(self):
        # 遍历 self._examples() 方法返回的每个示例元组 (Dist, keys, values, sample)
        for Dist, keys, values, sample in self._examples():

            # 定义内部函数 f，接受任意数量的参数 values，返回分布 Dist 的样本值
            def f(*values):
                # 将 keys 和 values 组合成参数字典 param
                param = dict(zip(keys, values))
                # 使用参数创建分布对象 dist
                dist = Dist(**param)
                # 返回该分布的样本值
                return dist.sample()

            # 使用 torch.jit.trace 对函数 f 进行跟踪，以进行 JIT 编译
            traced_f = torch.jit.trace(f, values, check_trace=False)

            # FIXME Schema not found for node
            # 定义无法成功跟踪的分布类型列表 xfail
            xfail = [
                Cauchy,  # aten::cauchy(Double(2,1), float, float, Generator)
                HalfCauchy,  # aten::cauchy(Double(2, 1), float, float, Generator)
                VonMises,  # Variance is not Euclidean
            ]
            # 如果当前的分布类型 Dist 存在于 xfail 中，则跳过当前循环
            if Dist in xfail:
                continue

            # 在随机数环境中执行函数 f(*values)，获取其样本值 sample
            with torch.random.fork_rng():
                sample = f(*values)
            # 调用 JIT 编译后的 traced_f(*values) 获取跟踪函数的样本值 traced_sample
            traced_sample = traced_f(*values)
            # 断言 sample 和 traced_sample 相等
            self.assertEqual(sample, traced_sample)

            # FIXME no nondeterministic nodes found in trace
            # 定义在跟踪中无法找到非确定性节点的分布类型列表 xfail
            xfail = [Beta, Dirichlet]
            # 如果当前的分布类型 Dist 不在 xfail 中，则执行下面的断言
            if Dist not in xfail:
                # 断言 traced_f.graph.nodes() 中是否存在任何非确定性节点
                self.assertTrue(
                    any(n.isNondeterministic() for n in traced_f.graph.nodes())
                )

    # 定义测试方法 test_rsample，用于测试反向样本采样函数的正确性
    def test_rsample(self):
        # 遍历 self._examples() 方法返回的每个示例元组 (Dist, keys, values, sample)
        for Dist, keys, values, sample in self._examples():
            # 如果当前的分布类型 Dist 不支持 rsample，则跳过当前循环
            if not Dist.has_rsample:
                continue

            # 定义内部函数 f，接受任意数量的参数 values，返回分布 Dist 的反向样本值
            def f(*values):
                # 将 keys 和 values 组合成参数字典 param
                param = dict(zip(keys, values))
                # 使用参数创建分布对象 dist
                dist = Dist(**param)
                # 返回该分布的反向样本值
                return dist.rsample()

            # 使用 torch.jit.trace 对函数 f 进行跟踪，以进行 JIT 编译
            traced_f = torch.jit.trace(f, values, check_trace=False)

            # FIXME Schema not found for node
            # 定义无法成功跟踪的分布类型列表 xfail
            xfail = [
                Cauchy,  # aten::cauchy(Double(2,1), float, float, Generator)
                HalfCauchy,  # aten::cauchy(Double(2, 1), float, float, Generator)
            ]
            # 如果当前的分布类型 Dist 存在于 xfail 中，则跳过当前循环
            if Dist in xfail:
                continue

            # 在随机数环境中执行函数 f(*values)，获取其反向样本值 sample
            with torch.random.fork_rng():
                sample = f(*values)
            # 调用 JIT 编译后的 traced_f(*values) 获取跟踪函数的反向样本值 traced_sample
            traced_sample = traced_f(*values)
            # 断言 sample 和 traced_sample 相等
            self.assertEqual(sample, traced_sample)

            # FIXME no nondeterministic nodes found in trace
            # 定义在跟踪中无法找到非确定性节点的分布类型列表 xfail
            xfail = [Beta, Dirichlet]
            # 如果当前的分布类型 Dist 不在 xfail 中，则执行下面的断言
            if Dist not in xfail:
                # 断言 traced_f.graph.nodes() 中是否存在任何非确定性节点
                self.assertTrue(
                    any(n.isNondeterministic() for n in traced_f.graph.nodes())
                )
    def test_log_prob(self):
        # 对于每个分布 Dist、参数 keys、values 和样本 sample，使用 _examples() 方法获取示例
        for Dist, keys, values, sample in self._examples():
            # FIXME: 跟踪的函数会产生不正确的结果
            xfail = [LowRankMultivariateNormal, MultivariateNormal]
            # 如果当前 Dist 在 xfail 列表中，则跳过当前循环
            if Dist in xfail:
                continue

            # 定义函数 f，接受一个样本 sample 和多个值 *values 作为参数
            def f(sample, *values):
                # 使用 keys 和 values 创建参数字典 param
                param = dict(zip(keys, values))
                # 根据参数创建分布对象 dist
                dist = Dist(**param)
                # 计算给定样本的对数概率值并返回
                return dist.log_prob(sample)

            # 使用 torch.jit.trace 方法对函数 f 进行跟踪
            traced_f = torch.jit.trace(f, (sample,) + values)

            # 在不同数据上进行检查
            values, sample = self._perturb(Dist, keys, values, sample)
            # 计算期望的对数概率值
            expected = f(sample, *values)
            # 使用跟踪后的函数计算实际的对数概率值
            actual = traced_f(sample, *values)
            # 断言期望值与实际值相等，否则输出错误信息
            self.assertEqual(
                expected,
                actual,
                msg=f"{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )

    def test_enumerate_support(self):
        # 对于每个分布 Dist、参数 keys、values 和样本 sample，使用 _examples() 方法获取示例
        for Dist, keys, values, sample in self._examples():
            # FIXME: 跟踪的函数会产生不正确的结果
            xfail = [Binomial]
            # 如果当前 Dist 在 xfail 列表中，则跳过当前循环
            if Dist in xfail:
                continue

            # 定义函数 f，接受多个值 *values 作为参数
            def f(*values):
                # 使用 keys 和 values 创建参数字典 param
                param = dict(zip(keys, values))
                # 根据参数创建分布对象 dist
                dist = Dist(**param)
                # 返回分布的支持集合
                return dist.enumerate_support()

            try:
                # 尝试使用 torch.jit.trace 方法对函数 f 进行跟踪
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                # 如果跟踪方法不可用，则继续下一个循环
                continue

            # 在不同数据上进行检查
            values, sample = self._perturb(Dist, keys, values, sample)
            # 计算期望的支持集合
            expected = f(*values)
            # 使用跟踪后的函数计算实际的支持集合
            actual = traced_f(*values)
            # 断言期望值与实际值相等，否则输出错误信息
            self.assertEqual(
                expected,
                actual,
                msg=f"{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )

    def test_mean(self):
        # 对于每个分布 Dist、参数 keys、values 和样本 sample，使用 _examples() 方法获取示例
        for Dist, keys, values, sample in self._examples():

            # 定义函数 f，接受多个值 *values 作为参数
            def f(*values):
                # 使用 keys 和 values 创建参数字典 param
                param = dict(zip(keys, values))
                # 根据参数创建分布对象 dist
                dist = Dist(**param)
                # 返回分布的均值
                return dist.mean

            try:
                # 尝试使用 torch.jit.trace 方法对函数 f 进行跟踪
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                # 如果跟踪方法不可用，则继续下一个循环
                continue

            # 在不同数据上进行检查
            values, sample = self._perturb(Dist, keys, values, sample)
            # 计算期望的均值
            expected = f(*values)
            # 使用跟踪后的函数计算实际的均值
            actual = traced_f(*values)
            # 将无限值（inf）替换为 0.0，因为均值不应为无限
            expected[expected == float("inf")] = 0.0
            actual[actual == float("inf")] = 0.0
            # 断言期望值与实际值相等，否则输出错误信息
            self.assertEqual(
                expected,
                actual,
                msg=f"{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )
    # 定义测试方差的方法
    def test_variance(self):
        # 对每个分布和其对应的示例进行迭代
        for Dist, keys, values, sample in self._examples():
            # 如果分布是 Cauchy 或 HalfCauchy，则跳过，因为它们具有无限方差
            if Dist in [Cauchy, HalfCauchy]:
                continue  # 无限方差

            # 定义函数 f，接受任意数量的参数值 values，并返回该分布的方差
            def f(*values):
                # 将参数名 keys 和对应的参数值 values 组合成字典 param
                param = dict(zip(keys, values))
                # 使用参数 param 创建指定分布 Dist 的实例 dist
                dist = Dist(**param)
                # 返回该分布的方差
                return dist.variance

            try:
                # 尝试对函数 f 进行 Torch JIT 追踪，以优化性能
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                # 如果追踪不支持，则继续下一个分布的测试
                continue

            # 使用不同的数据进行验证
            values, sample = self._perturb(Dist, keys, values, sample)
            # 计算预期的方差值
            expected = f(*values).clone()
            # 计算通过 JIT 追踪后的方差值
            actual = traced_f(*values).clone()
            # 将预期结果中的无穷值替换为 0.0
            expected[expected == float("inf")] = 0.0
            # 将实际结果中的无穷值替换为 0.0
            actual[actual == float("inf")] = 0.0
            # 断言预期方差与实际通过 JIT 追踪得到的方差相等
            self.assertEqual(
                expected,
                actual,
                msg=f"{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )

    @set_default_dtype(torch.double)
    # 定义测试熵的方法，并将 Torch 默认数据类型设置为双精度浮点数
    def test_entropy(self):
        # 对每个分布和其对应的示例进行迭代
        for Dist, keys, values, sample in self._examples():
            # 确定哪些分布会导致追踪函数产生不正确的结果
            xfail = [LowRankMultivariateNormal, MultivariateNormal]
            if Dist in xfail:
                continue

            # 定义函数 f，接受任意数量的参数值 values，并返回该分布的熵
            def f(*values):
                # 将参数名 keys 和对应的参数值 values 组合成字典 param
                param = dict(zip(keys, values))
                # 使用参数 param 创建指定分布 Dist 的实例 dist
                dist = Dist(**param)
                # 返回该分布的熵
                return dist.entropy()

            try:
                # 尝试对函数 f 进行 Torch JIT 追踪，以优化性能
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                # 如果追踪不支持，则继续下一个分布的测试
                continue

            # 使用不同的数据进行验证
            values, sample = self._perturb(Dist, keys, values, sample)
            # 计算预期的熵值
            expected = f(*values)
            # 计算通过 JIT 追踪后的熵值
            actual = traced_f(*values)
            # 断言预期熵与实际通过 JIT 追踪得到的熵相等
            self.assertEqual(
                expected,
                actual,
                msg=f"{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )

    @set_default_dtype(torch.double)
    # 定义测试累积分布函数 (CDF) 的方法，并将 Torch 默认数据类型设置为双精度浮点数
    def test_cdf(self):
        # 对每个分布和其对应的示例进行迭代
        for Dist, keys, values, sample in self._examples():

            # 定义函数 f，接受采样值 sample 和任意数量的参数值 values，并返回根据 CDF 反推的值
            def f(sample, *values):
                # 将参数名 keys 和对应的参数值 values 组合成字典 param
                param = dict(zip(keys, values))
                # 使用参数 param 创建指定分布 Dist 的实例 dist
                dist = Dist(**param)
                # 计算采样值 sample 的累积分布函数 (CDF)
                cdf = dist.cdf(sample)
                # 返回根据 CDF 反推的值
                return dist.icdf(cdf)

            try:
                # 尝试对函数 f 进行 Torch JIT 追踪，以优化性能
                traced_f = torch.jit.trace(f, (sample,) + values)
            except NotImplementedError:
                # 如果追踪不支持，则继续下一个分布的测试
                continue

            # 使用不同的数据进行验证
            values, sample = self._perturb(Dist, keys, values, sample)
            # 计算预期的反推值
            expected = f(sample, *values)
            # 计算通过 JIT 追踪后的反推值
            actual = traced_f(sample, *values)
            # 断言预期反推值与实际通过 JIT 追踪得到的值相等
            self.assertEqual(
                expected,
                actual,
                msg=f"{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )
# 如果当前脚本被直接执行，并且 torch 库支持 LAPACK（线性代数包），
# 则设置 TestCase 类的默认数据类型检查启用标志为 True
if __name__ == "__main__" and torch._C.has_lapack:
    # 启用 TestCase 类的默认数据类型检查
    TestCase._default_dtype_check_enabled = True
    # 运行测试函数或测试套件
    run_tests()
```