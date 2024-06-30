# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_common.py`

```
# SPDX-License-Identifier: BSD-3-Clause
# 导入必要的库和模块
import inspect  # 用于检查对象的属性和方法
import numpy as np  # 用于科学计算的库
import pytest  # 用于编写和运行测试的框架
from sklearn.base import is_classifier  # 检查是否为分类器的函数
from sklearn.datasets import make_low_rank_matrix  # 生成低秩矩阵的函数
from sklearn.linear_model import (  # 导入多个线性模型类
    ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, Lars, LarsCV,
    Lasso, LassoCV, LassoLarsCV, LassoLarsIC, LinearRegression,
    LogisticRegression, LogisticRegressionCV, MultiTaskElasticNet,
    MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV,
    OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV,
    PoissonRegressor, Ridge, RidgeCV, SGDRegressor, TweedieRegressor
)


# Note: GammaRegressor() and TweedieRegressor(power != 1) have a non-canonical link.
# 使用pytest.mark.parametrize装饰器，定义测试参数化，针对不同的模型进行测试
@pytest.mark.parametrize(
    "model",
    [
        ARDRegression(),  # 贝叶斯自相关稀疏度回归模型
        BayesianRidge(),  # 贝叶斯岭回归模型
        ElasticNet(),  # 弹性网络模型
        ElasticNetCV(),  # 带交叉验证的弹性网络模型
        Lars(),  # 最小角回归模型
        LarsCV(),  # 带交叉验证的最小角回归模型
        Lasso(),  # 套索回归模型
        LassoCV(),  # 带交叉验证的套索回归模型
        LassoLarsCV(),  # 带交叉验证的Lasso最小角回归模型
        LassoLarsIC(),  # 基于信息准则的Lasso最小角回归模型
        LinearRegression(),  # 线性回归模型
        # TODO: FIx SAGA which fails badly with sample_weights.
        # This is a known limitation, see:
        # https://github.com/scikit-learn/scikit-learn/issues/21305
        # 参数化一个LogisticRegression对象，用于测试
        pytest.param(
            LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5, tol=1e-15
            ),
            marks=pytest.mark.xfail(reason="Missing importance sampling scheme"),  # 标记为预期失败，原因是缺少重要性抽样方案
        ),
        LogisticRegressionCV(tol=1e-6),  # 带交叉验证的逻辑回归模型
        MultiTaskElasticNet(),  # 多任务弹性网络模型
        MultiTaskElasticNetCV(),  # 带交叉验证的多任务弹性网络模型
        MultiTaskLasso(),  # 多任务套索回归模型
        MultiTaskLassoCV(),  # 带交叉验证的多任务套索回归模型
        OrthogonalMatchingPursuit(),  # 正交匹配追踪模型
        OrthogonalMatchingPursuitCV(),  # 带交叉验证的正交匹配追踪模型
        PoissonRegressor(),  # 泊松回归模型
        Ridge(),  # 岭回归模型
        RidgeCV(),  # 带交叉验证的岭回归模型
        pytest.param(
            SGDRegressor(tol=1e-15),
            marks=pytest.mark.xfail(reason="Insufficient precision."),  # 标记为预期失败，原因是精度不足
        ),
        SGDRegressor(penalty="elasticnet", max_iter=10_000),  # 带有弹性网络惩罚项的随机梯度下降模型
        TweedieRegressor(power=0),  # Tweedie回归模型，参数power=0时等同于岭回归
    ],
    ids=lambda x: x.__class__.__name__,  # 用类名作为测试用例的标识符
)
@pytest.mark.parametrize("with_sample_weight", [False, True])  # 参数化样本权重是否存在
def test_balance_property(model, with_sample_weight, global_random_seed):
    # 测试平衡性质，即在训练集上预测值的总和应等于观测值的总和
    # 对于所有具有指数分布损失和相应的典型链接函数的线性模型，如果fit_intercept=True，则必须成立
    # 例如：
    #     - 平方误差和恒等链接（大多数线性模型）
    #     - 带对数链接的泊松偏差
    #     - 带logit链接的对数损失
    # 这被称为平衡性质或无条件校准/无偏性。
    # 参考文献：见M.V. Wuthrich和M. Merz的第3.18、3.20和第5.1.5章
    # "Statistical Foundations of Actuarial Learning and its Applications"（2022年6月3日）
    # http://doi.org/10.2139/ssrn.3822407
    # 如果需要使用样本权重，并且模型的 fit 方法的参数中没有 sample_weight，则跳过测试
    if (
        with_sample_weight
        and "sample_weight" not in inspect.signature(model.fit).parameters.keys()
    ):
        pytest.skip("Estimator does not support sample_weight.")

    # 设置测试精度为 2e-4
    rel = 2e-4  # test precision
    # 如果模型是 SGDRegressor 类型，则调整测试精度为 1e-1
    if isinstance(model, SGDRegressor):
        rel = 1e-1
    # 如果模型具有 solver 属性且 solver 属性为 "saga"，则调整测试精度为 1e-2
    elif hasattr(model, "solver") and model.solver == "saga":
        rel = 1e-2

    # 使用全局随机种子创建随机数生成器对象
    rng = np.random.RandomState(global_random_seed)
    n_train, n_features, n_targets = 100, 10, None
    # 如果模型属于以下多任务学习类别，则设置目标数为 3
    if isinstance(
        model,
        (MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV),
    ):
        n_targets = 3
    # 创建低秩矩阵作为特征矩阵 X
    X = make_low_rank_matrix(n_samples=n_train, n_features=n_features, random_state=rng)
    if n_targets:
        # 如果存在多个目标，则生成系数矩阵 coef
        coef = (
            rng.uniform(low=-2, high=2, size=(n_features, n_targets))
            / np.max(X, axis=0)[:, None]
        )
    else:
        # 否则，生成单一目标的系数向量 coef
        coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)

    # 根据 X 和 coef 计算期望值，使用指数函数计算
    expectation = np.exp(X @ coef + 0.5)
    # 生成泊松分布的随机数作为目标值 y，并确保严格为正数（即 y > 0）
    y = rng.poisson(lam=expectation) + 1  # strict positive, i.e. y > 0
    # 如果模型是分类器，则根据预期值和阈值进行分类
    if is_classifier(model):
        y = (y > expectation + 1).astype(np.float64)

    # 如果需要使用样本权重，则生成样本权重 sw，否则设为 None
    if with_sample_weight:
        sw = rng.uniform(low=1, high=10, size=y.shape[0])
    else:
        sw = None

    # 设置模型参数，确保 fit_intercept 为 True
    model.set_params(fit_intercept=True)  # to be sure
    # 根据是否需要使用样本权重，调用模型的 fit 方法进行训练
    if with_sample_weight:
        model.fit(X, y, sample_weight=sw)
    else:
        model.fit(X, y)

    # 断言分类器的预测结果平均值与目标值平均值相等，或者回归模型的预测结果平均值与目标值平均值相等，使用权重 sw 进行加权
    if is_classifier(model):
        assert np.average(model.predict_proba(X)[:, 1], weights=sw) == pytest.approx(
            np.average(y, weights=sw), rel=rel
        )
    else:
        assert np.average(model.predict(X), weights=sw, axis=0) == pytest.approx(
            np.average(y, weights=sw, axis=0), rel=rel
        )
```