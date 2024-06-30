# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\test_factor_analysis.py`

```
# 导入 itertools 模块中的 combinations 函数
from itertools import combinations

# 导入 numpy 和 pytest 模块
import numpy as np
import pytest

# 导入 scikit-learn 中的 FactorAnalysis 类和 _ortho_rotation 函数
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition._factor_analysis import _ortho_rotation

# 导入警告处理相关模块
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    ignore_warnings,
)

# 使用 ignore_warnings 装饰器忽略警告
@ignore_warnings
# 定义测试函数 test_factor_analysis，接收全局随机种子作为参数
def test_factor_analysis(global_random_seed):
    # 使用全局随机种子创建随机数生成器 rng
    rng = np.random.RandomState(global_random_seed)
    n_samples, n_features, n_components = 20, 5, 3

    # 生成模型的随机设置
    W = rng.randn(n_components, n_features)
    # 维度为 3 的潜在变量，共 20 个样本
    h = rng.randn(n_samples, n_components)
    # 使用 gamma 分布生成噪声方差
    noise = rng.gamma(1, size=n_features) * rng.randn(n_samples, n_features)

    # 生成观测数据 X
    X = np.dot(h, W) + noise

    # 存储 FactorAnalysis 对象的列表
    fas = []
    for method in ["randomized", "lapack"]:
        # 创建 FactorAnalysis 对象 fa，指定参数和 SVD 方法
        fa = FactorAnalysis(n_components=n_components, svd_method=method)
        # 在数据 X 上拟合 FactorAnalysis 模型
        fa.fit(X)
        # 将拟合好的 FactorAnalysis 对象添加到 fas 列表中
        fas.append(fa)

        # 变换数据 X 并检查维度
        X_t = fa.transform(X)
        assert X_t.shape == (n_samples, n_components)

        # 检查对数似然最终值和分数样本的总和近似相等
        assert_almost_equal(fa.loglike_[-1], fa.score_samples(X).sum())
        # 检查分数样本的均值和 score 方法的结果近似相等
        assert_almost_equal(fa.score_samples(X).mean(), fa.score(X))

        # 检查对数似然的增加情况
        diff = np.all(np.diff(fa.loglike_))
        assert diff > 0.0, "Log likelihood did not increase"

        # 计算样本协方差矩阵
        scov = np.cov(X, rowvar=0.0, bias=1.0)

        # 获取模型估计的协方差矩阵
        mcov = fa.get_covariance()
        # 计算平均绝对差异
        diff = np.sum(np.abs(scov - mcov)) / W.size
        assert diff < 0.2, "Mean absolute difference is %f" % diff

        # 使用初始噪声方差初始化 FactorAnalysis 对象 fa
        fa = FactorAnalysis(
            n_components=n_components, noise_variance_init=np.ones(n_features)
        )
        # 预期引发 ValueError 异常
        with pytest.raises(ValueError):
            fa.fit(X[:, :2])

    # 定义辅助函数 f，用于返回属性的绝对值
    def f(x, y):
        return np.abs(getattr(x, y))

    # 分别对比两个 FactorAnalysis 对象 fa1 和 fa2 的指定属性
    fa1, fa2 = fas
    for attr in ["loglike_", "components_", "noise_variance_"]:
        assert_almost_equal(f(fa1, attr), f(fa2, attr))

    # 设置 fa1 的最大迭代次数为 1，开启 verbose 模式
    fa1.max_iter = 1
    fa1.verbose = True
    # 预期引发 ConvergenceWarning 警告
    with pytest.warns(ConvergenceWarning):
        fa1.fit(X)

    # 测试在不同的 n_components 设置下 get_covariance 和 get_precision 方法
    for n_components in [0, 2, X.shape[1]]:
        fa.n_components = n_components
        fa.fit(X)
        # 获取协方差矩阵和精度矩阵
        cov = fa.get_covariance()
        precision = fa.get_precision()
        # 检查它们的乘积是否接近单位矩阵
        assert_array_almost_equal(np.dot(cov, precision), np.eye(X.shape[1]), 12)

    # 测试旋转功能
    n_components = 2

    results, projections = {}, {}
    # 对每种因子分析方法进行循环：无旋转、varimax、quartimax
    for method in (None, "varimax", "quartimax"):
        # 使用 FactorAnalysis 进行因子分析，设置成分数目和旋转方法
        fa_var = FactorAnalysis(n_components=n_components, rotation=method)
        # 对数据 X 进行因子分析并获取转换后的结果
        results[method] = fa_var.fit_transform(X)
        # 获取因子分析的投影矩阵
        projections[method] = fa_var.get_covariance()
    
    # 对于所有可能的两两旋转组合，执行以下断言：
    for rot1, rot2 in combinations([None, "varimax", "quartimax"], 2):
        # 断言两种旋转方法下的结果不完全相等
        assert not np.allclose(results[rot1], results[rot2])
        # 断言两种旋转方法下的投影矩阵在指定容差下近似相等
        assert np.allclose(projections[rot1], projections[rot2], atol=3)

    # 针对 R 中 psych::principal 函数使用 rotate="varimax" 的测试
    # （即下面的值来源于在 R 中对组件进行旋转）
    # R 的因子分析返回的值可能会有较大差异，因此我们只测试旋转本身的效果
    # 定义预先计算的因子矩阵和 R 中的旋转后的解决方案
    factors = np.array(
        [
            [0.89421016, -0.35854928, -0.27770122, 0.03773647],
            [-0.45081822, -0.89132754, 0.0932195, -0.01787973],
            [0.99500666, -0.02031465, 0.05426497, -0.11539407],
            [0.96822861, -0.06299656, 0.24411001, 0.07540887],
        ]
    )
    # R 中旋转后的解决方案
    r_solution = np.array(
        [[0.962, 0.052], [-0.141, 0.989], [0.949, -0.300], [0.937, -0.251]]
    )
    # 对 factors 进行 varimax 正交旋转，然后进行转置
    rotated = _ortho_rotation(factors[:, :n_components], method="varimax").T
    # 断言正交旋转后的绝对值近似等于 R 中的旋转解决方案的绝对值，精度为 3 位小数
    assert_array_almost_equal(np.abs(rotated), np.abs(r_solution), decimal=3)
```