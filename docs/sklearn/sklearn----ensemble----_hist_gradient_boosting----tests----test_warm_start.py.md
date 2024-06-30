# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\test_warm_start.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 Pytest 库，用于单元测试
from numpy.testing import assert_allclose, assert_array_equal  # 导入 NumPy 测试工具中的函数

from sklearn.base import clone  # 导入 sklearn 库中的 clone 函数
from sklearn.datasets import make_classification, make_regression  # 导入 sklearn 库中的数据生成函数
from sklearn.ensemble import (  # 导入 sklearn 库中的集成模型
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import check_scoring  # 导入 sklearn 库中的评估函数

X_classification, y_classification = make_classification(random_state=0)  # 生成分类数据集
X_regression, y_regression = make_regression(random_state=0)  # 生成回归数据集


def _assert_predictor_equal(gb_1, gb_2, X):
    """Assert that two HistGBM instances are identical."""
    # 检查每棵树的节点是否完全相同
    for pred_ith_1, pred_ith_2 in zip(gb_1._predictors, gb_2._predictors):
        for predictor_1, predictor_2 in zip(pred_ith_1, pred_ith_2):
            assert_array_equal(predictor_1.nodes, predictor_2.nodes)

    # 检查预测结果是否完全相同
    assert_allclose(gb_1.predict(X), gb_2.predict(X))


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),  # 参数化测试：分类问题
        (HistGradientBoostingRegressor, X_regression, y_regression),  # 参数化测试：回归问题
    ],
)
def test_max_iter_with_warm_start_validation(GradientBoosting, X, y):
    # 检查当 warm_start 为 True 时，如果最大迭代次数小于上一次拟合的迭代次数，是否会引发 ValueError

    estimator = GradientBoosting(max_iter=10, early_stopping=False, warm_start=True)
    estimator.fit(X, y)
    estimator.set_params(max_iter=5)
    err_msg = (
        "max_iter=5 must be larger than or equal to n_iter_=10 when warm_start==True"
    )
    with pytest.raises(ValueError, match=err_msg):
        estimator.fit(X, y)


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),  # 参数化测试：分类问题
        (HistGradientBoostingRegressor, X_regression, y_regression),  # 参数化测试：回归问题
    ],
)
def test_warm_start_yields_identical_results(GradientBoosting, X, y):
    # 确保先拟合 50 次迭代，然后通过 warm start 再拟合 25 次迭代，结果与直接拟合 75 次迭代相同

    rng = 42
    gb_warm_start = GradientBoosting(
        n_iter_no_change=100, max_iter=50, random_state=rng, warm_start=True
    )
    gb_warm_start.fit(X, y).set_params(max_iter=75).fit(X, y)

    gb_no_warm_start = GradientBoosting(
        n_iter_no_change=100, max_iter=75, random_state=rng, warm_start=False
    )
    gb_no_warm_start.fit(X, y)

    # 检查两个预测器是否完全相同
    _assert_predictor_equal(gb_warm_start, gb_no_warm_start, X)


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),  # 参数化测试：分类问题
        (HistGradientBoostingRegressor, X_regression, y_regression),  # 参数化测试：回归问题
    ],
)
def test_warm_start_max_depth(GradientBoosting, X, y):
    # 测试在集成中是否可以拟合不同深度的树。
    # 创建一个梯度提升模型对象，设置初始参数
    gb = GradientBoosting(
        max_iter=20,                 # 最大迭代次数为20
        min_samples_leaf=1,          # 叶子节点的最小样本数为1
        warm_start=True,             # 开启热启动模式
        max_depth=2,                 # 每棵树的最大深度为2
        early_stopping=False,        # 禁用早停策略
    )
    # 使用训练数据 X 和标签 y 来拟合梯度提升模型
    gb.fit(X, y)
    # 修改模型参数：增加最大迭代次数至30，每棵树的最大深度增加至3，迭代不改变时的最大迭代次数设置为110
    gb.set_params(max_iter=30, max_depth=3, n_iter_no_change=110)
    # 使用更新后的参数重新拟合梯度提升模型
    gb.fit(X, y)

    # 检查前20棵树的最大深度是否为2
    for i in range(20):
        # 断言第i棵树的最大深度为2
        assert gb._predictors[i][0].get_max_depth() == 2
    # 检查最后10棵树的最大深度是否为3
    for i in range(1, 11):
        # 断言倒数第i棵树的最大深度为3
        assert gb._predictors[-i][0].get_max_depth() == 3
@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
@pytest.mark.parametrize("scoring", (None, "loss"))
def test_warm_start_early_stopping(GradientBoosting, X, y, scoring):
    # 确保在使用温暖启动进行第二次拟合时，早停在少量迭代后发生。

    n_iter_no_change = 5
    # 创建梯度提升对象，设置早停、温暖启动等参数
    gb = GradientBoosting(
        n_iter_no_change=n_iter_no_change,
        max_iter=10000,
        early_stopping=True,
        random_state=42,
        warm_start=True,
        tol=1e-3,
        scoring=scoring,
    )
    gb.fit(X, y)
    n_iter_first_fit = gb.n_iter_
    gb.fit(X, y)
    n_iter_second_fit = gb.n_iter_
    # 断言第二次拟合的迭代次数比第一次多且不超过 n_iter_no_change
    assert 0 < n_iter_second_fit - n_iter_first_fit < n_iter_no_change


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
def test_warm_start_equal_n_estimators(GradientBoosting, X, y):
    # 测试当 n_estimators 相等时，温暖启动不产生效果

    gb_1 = GradientBoosting(max_depth=2, early_stopping=False)
    gb_1.fit(X, y)

    gb_2 = clone(gb_1)
    gb_2.set_params(max_iter=gb_1.max_iter, warm_start=True, n_iter_no_change=5)
    gb_2.fit(X, y)

    # 检查两个预测器是否相等
    _assert_predictor_equal(gb_1, gb_2, X)


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
def test_warm_start_clear(GradientBoosting, X, y):
    # 测试拟合是否清除状态。

    gb_1 = GradientBoosting(n_iter_no_change=5, random_state=42)
    gb_1.fit(X, y)

    gb_2 = GradientBoosting(n_iter_no_change=5, random_state=42, warm_start=True)
    gb_2.fit(X, y)  # 初始化状态
    gb_2.set_params(warm_start=False)
    gb_2.fit(X, y)  # 清除旧状态并等效于估计器

    # 检查两个预测器是否具有相同的 train_score_ 和 validation_score_ 属性
    assert_allclose(gb_1.train_score_, gb_2.train_score_)
    assert_allclose(gb_1.validation_score_, gb_2.validation_score_)

    # 检查两个预测器是否相等
    _assert_predictor_equal(gb_1, gb_2, X)


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
@pytest.mark.parametrize("rng_type", ("none", "int", "instance"))
def test_random_seeds_warm_start(GradientBoosting, X, y, rng_type):
    # 确保在温暖启动上下文中，正确设置了训练/验证集分割和小训练集子采样的种子。
    # 定义一个辅助函数，根据 rng_type 返回相应的随机数生成器或 None
    def _get_rng(rng_type):
        # 如果 rng_type 为 "none"，返回 None
        if rng_type == "none":
            return None
        # 如果 rng_type 为 "int"，返回固定的整数 42
        elif rng_type == "int":
            return 42
        else:
            # 否则返回一个以种子 0 初始化的 NumPy 随机数生成器对象
            return np.random.RandomState(0)

    # 使用 _get_rng 函数根据 rng_type 获取随机数生成器的状态
    random_state = _get_rng(rng_type)

    # 创建第一个梯度提升树模型，启用早停，最大迭代次数为 2，随机数种子为 random_state
    gb_1 = GradientBoosting(early_stopping=True, max_iter=2, random_state=random_state)

    # 设置模型的评分方法为 check_scoring 返回的评分方法
    gb_1.set_params(scoring=check_scoring(gb_1))

    # 使用训练集 X 和标签 y 拟合第一个模型
    gb_1.fit(X, y)

    # 获取第一个模型的随机数种子
    random_seed_1_1 = gb_1._random_seed

    # 再次使用相同训练集拟合第一个模型，这会清除旧状态，生成不同的随机种子
    gb_1.fit(X, y)
    random_seed_1_2 = gb_1._random_seed  # 清除旧状态，生成不同的随机种子

    # 使用相同的 random_state 创建第二个梯度提升树模型，并启用热启动
    gb_2 = GradientBoosting(
        early_stopping=True, max_iter=2, random_state=random_state, warm_start=True
    )

    # 设置第二个模型的评分方法为 check_scoring 返回的评分方法
    gb_2.set_params(scoring=check_scoring(gb_2))

    # 使用训练集 X 和标签 y 拟合第二个模型，初始化其状态
    gb_2.fit(X, y)

    # 获取第二个模型的随机数种子
    random_seed_2_1 = gb_2._random_seed

    # 再次使用相同训练集拟合第二个模型，这会清除旧状态，生成相同的随机种子
    gb_2.fit(X, y)

    # 获取第二个模型再次拟合后的随机数种子
    random_seed_2_2 = gb_2._random_seed

    # 根据 rng_type 的不同，验证随机种子的状态
    if rng_type == "none":
        # 如果 rng_type 是 "none"，则断言所有随机种子都不相等
        assert random_seed_1_1 != random_seed_1_2 != random_seed_2_1
    elif rng_type == "int":
        # 如果 rng_type 是 "int"，则断言所有随机种子都相等
        assert random_seed_1_1 == random_seed_1_2 == random_seed_2_1
    else:
        # 否则断言第一个和第二个模型的第一个随机种子相等，但第一个和第二个模型的第二个随机种子不相等
        assert random_seed_1_1 == random_seed_2_1 != random_seed_1_2

    # 当启用热启动时，断言第二个模型的两个随机种子相等
    assert random_seed_2_1 == random_seed_2_2
```