# `D:\src\scipysrc\scikit-learn\sklearn\inspection\tests\test_partial_dependence.py`

```
"""
Testing for the partial dependence module.
"""

# 导入所需的库和模块
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest用于测试

import sklearn  # 导入scikit-learn库
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_regressor  # 导入BaseEstimator等类和函数
from sklearn.cluster import KMeans  # 导入KMeans聚类算法
from sklearn.compose import make_column_transformer  # 导入make_column_transformer函数
from sklearn.datasets import load_iris, make_classification, make_regression  # 导入数据集加载和生成函数
from sklearn.dummy import DummyClassifier  # 导入DummyClassifier虚拟分类器
from sklearn.ensemble import (  # 导入集成学习算法
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError  # 导入NotFittedError异常
from sklearn.inspection import partial_dependence  # 导入partial_dependence方法
from sklearn.inspection._partial_dependence import (  # 导入偏依赖相关的内部函数
    _grid_from_X,
    _partial_dependence_brute,
    _partial_dependence_recursion,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, MultiTaskLasso  # 导入线性模型和多任务Lasso模型
from sklearn.metrics import r2_score  # 导入R²评分指标
from sklearn.pipeline import make_pipeline  # 导入make_pipeline函数
from sklearn.preprocessing import (  # 导入数据预处理函数
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    scale,
)
from sklearn.tree import DecisionTreeRegressor  # 导入决策树回归模型
from sklearn.tree.tests.test_tree import assert_is_subtree  # 导入测试函数用于确认子树
from sklearn.utils._testing import assert_allclose, assert_array_equal  # 导入测试函数用于数组比较
from sklearn.utils.fixes import _IS_32BIT  # 导入修复功能
from sklearn.utils.validation import check_random_state  # 导入验证函数，检查随机状态

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]  # 创建一个简单的数据集X
y = [-1, -1, -1, 1, 1, 1]  # 对应的目标变量y


# (X, y), n_targets  <-- as expected in the output of partial_dep()
binary_classification_data = (make_classification(n_samples=50, random_state=0), 1)  # 生成二元分类数据
multiclass_classification_data = (  # 生成多类分类数据
    make_classification(
        n_samples=50, n_classes=3, n_clusters_per_class=1, random_state=0
    ),
    3,
)
regression_data = (make_regression(n_samples=50, random_state=0), 1)  # 生成回归数据
multioutput_regression_data = (  # 生成多输出回归数据
    make_regression(n_samples=50, n_targets=2, random_state=0),
    2,
)

# iris数据集
iris = load_iris()


@pytest.mark.parametrize(  # 使用pytest参数化装饰器定义测试参数
    "Estimator, method, data",  # 参数包括估计器、方法和数据
    [  # 参数化的测试用例
        (GradientBoostingClassifier, "auto", binary_classification_data),
        (GradientBoostingClassifier, "auto", multiclass_classification_data),
        (GradientBoostingClassifier, "brute", binary_classification_data),
        (GradientBoostingClassifier, "brute", multiclass_classification_data),
        (GradientBoostingRegressor, "auto", regression_data),
        (GradientBoostingRegressor, "brute", regression_data),
        (DecisionTreeRegressor, "brute", regression_data),
        (LinearRegression, "brute", regression_data),
        (LinearRegression, "brute", multioutput_regression_data),
        (LogisticRegression, "brute", binary_classification_data),
        (LogisticRegression, "brute", multiclass_classification_data),
        (MultiTaskLasso, "brute", multioutput_regression_data),
    ],
)
@pytest.mark.parametrize("grid_resolution", (5, 10))  # 参数化网格分辨率
@pytest.mark.parametrize("features", ([1], [1, 2]))  # 参数化特征列表
@pytest.mark.parametrize("kind", ("average", "individual", "both"))
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数，以测试不同的 kind 参数取值

def test_output_shape(Estimator, method, data, grid_resolution, features, kind):
    # 检查 partial_dependence 对不同类型的估计器的输出形状是否一致：
    # - 二分类和多分类设置下的分类器
    # - 回归器
    # - 多任务回归器

    est = Estimator()
    if hasattr(est, "n_estimators"):
        est.set_params(n_estimators=2)  # 加快计算速度

    # n_target 对应类别数量（二分类为1）或多任务设置下的任务/输出数量。对于经典回归数据，它为1。
    (X, y), n_targets = data
    n_instances = X.shape[0]

    est.fit(X, y)
    result = partial_dependence(
        est,
        X=X,
        features=features,
        method=method,
        kind=kind,
        grid_resolution=grid_resolution,
    )
    pdp, axes = result, result["grid_values"]

    expected_pdp_shape = (n_targets, *[grid_resolution for _ in range(len(features))])
    expected_ice_shape = (
        n_targets,
        n_instances,
        *[grid_resolution for _ in range(len(features))],
    )
    if kind == "average":
        assert pdp.average.shape == expected_pdp_shape
    elif kind == "individual":
        assert pdp.individual.shape == expected_ice_shape
    else:  # 'both'
        assert pdp.average.shape == expected_pdp_shape
        assert pdp.individual.shape == expected_ice_shape

    expected_axes_shape = (len(features), grid_resolution)
    assert axes is not None
    assert np.asarray(axes).shape == expected_axes_shape


def test_grid_from_X():
    # _grid_from_X 函数的测试：输出的合理性检查，以及形状检查。

    # 确保网格是输入的笛卡尔积（将使用唯一值而不是百分位数）
    percentiles = (0.05, 0.95)
    grid_resolution = 100
    is_categorical = [False, False]
    X = np.asarray([[1, 2], [3, 4]])
    grid, axes = _grid_from_X(X, percentiles, is_categorical, grid_resolution)
    assert_array_equal(grid, [[1, 2], [1, 4], [3, 2], [3, 4]])
    assert_array_equal(axes, X.T)

    # 测试返回对象的形状，取决于特征的唯一值数量。
    rng = np.random.RandomState(0)
    grid_resolution = 15

    # n_unique_values > grid_resolution
    X = rng.normal(size=(20, 2))
    grid, axes = _grid_from_X(
        X, percentiles, is_categorical, grid_resolution=grid_resolution
    )
    assert grid.shape == (grid_resolution * grid_resolution, X.shape[1])
    assert np.asarray(axes).shape == (2, grid_resolution)

    # n_unique_values < grid_resolution, 将使用实际值
    n_unique_values = 12
    X[n_unique_values - 1 :, 0] = 12345
    rng.shuffle(X)  # 为确保顺序无关而洗牌
    grid, axes = _grid_from_X(
        X, percentiles, is_categorical, grid_resolution=grid_resolution
    )
    # 断言检查网格形状是否符合预期，应为 (唯一值数量 * 网格分辨率, X 的列数)
    assert grid.shape == (n_unique_values * grid_resolution, X.shape[1])
    
    # 断言检查 axes 的第一个元素的形状是否为 (唯一值数量,)，即一个包含唯一值数量个元素的数组
    assert axes[0].shape == (n_unique_values,)
    
    # 断言检查 axes 的第二个元素的形状是否为 (网格分辨率,)，即一个包含网格分辨率个元素的数组
    assert axes[1].shape == (grid_resolution,)
@pytest.mark.parametrize(
    "grid_resolution",
    [
        2,  # 因为 n_categories > 2，所以不应使用分位数重采样
        100,
    ],
)
def test_grid_from_X_with_categorical(grid_resolution):
    """检查 `_grid_from_X` 总是从类别中采样，不依赖于百分位数。"""
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如果不存在则跳过测试
    percentiles = (0.05, 0.95)
    is_categorical = [True]
    X = pd.DataFrame({"cat_feature": ["A", "B", "C", "A", "B", "D", "E"]})
    grid, axes = _grid_from_X(
        X, percentiles, is_categorical, grid_resolution=grid_resolution
    )
    assert grid.shape == (5, X.shape[1])  # 检查网格形状是否正确
    assert axes[0].shape == (5,)  # 检查第一个轴的形状是否正确


@pytest.mark.parametrize("grid_resolution", [3, 100])
def test_grid_from_X_heterogeneous_type(grid_resolution):
    """检查 `_grid_from_X` 总是从类别中采样，不依赖于百分位数。"""
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如果不存在则跳过测试
    percentiles = (0.05, 0.95)
    is_categorical = [True, False]
    X = pd.DataFrame(
        {
            "cat": ["A", "B", "C", "A", "B", "D", "E", "A", "B", "D"],
            "num": [1, 1, 1, 2, 5, 6, 6, 6, 6, 8],
        }
    )
    nunique = X.nunique()  # 统计唯一值的数量

    grid, axes = _grid_from_X(
        X, percentiles, is_categorical, grid_resolution=grid_resolution
    )
    if grid_resolution == 3:
        assert grid.shape == (15, 2)  # 检查网格形状是否正确
        assert axes[0].shape[0] == nunique["num"]  # 检查第一个轴的长度是否正确
        assert axes[1].shape[0] == grid_resolution  # 检查第二个轴的长度是否正确
    else:
        assert grid.shape == (25, 2)  # 检查网格形状是否正确
        assert axes[0].shape[0] == nunique["cat"]  # 检查第一个轴的长度是否正确
        assert axes[1].shape[0] == nunique["cat"]  # 检查第二个轴的长度是否正确


@pytest.mark.parametrize(
    "grid_resolution, percentiles, err_msg",
    [
        (2, (0, 0.0001), "percentiles are too close"),  # 百分位数过于接近的错误情况
        (100, (1, 2, 3, 4), "'percentiles' must be a sequence of 2 elements"),  # 百分位数元组长度错误的错误情况
        (100, 12345, "'percentiles' must be a sequence of 2 elements"),  # 百分位数类型错误的错误情况
        (100, (-1, 0.95), r"'percentiles' values must be in \[0, 1\]"),  # 百分位数范围错误的错误情况
        (100, (0.05, 2), r"'percentiles' values must be in \[0, 1\]"),  # 百分位数范围错误的错误情况
        (100, (0.9, 0.1), r"percentiles\[0\] must be strictly less than"),  # 百分位数顺序错误的错误情况
        (1, (0.05, 0.95), "'grid_resolution' must be strictly greater than 1"),  # 网格分辨率过小的错误情况
    ],
)
def test_grid_from_X_error(grid_resolution, percentiles, err_msg):
    X = np.asarray([[1, 2], [3, 4]])
    is_categorical = [False]
    with pytest.raises(ValueError, match=err_msg):  # 检查是否抛出预期的 ValueError 异常
        _grid_from_X(X, percentiles, is_categorical, grid_resolution)


@pytest.mark.parametrize("target_feature", range(5))
@pytest.mark.parametrize(
    "est, method",
    [
        (LinearRegression(), "brute"),
        (GradientBoostingRegressor(random_state=0), "brute"),
        (GradientBoostingRegressor(random_state=0), "recursion"),
        (HistGradientBoostingRegressor(random_state=0), "brute"),
        (HistGradientBoostingRegressor(random_state=0), "recursion"),
    ],
)
def test_partial_dependence_helpers(est, method, target_feature):
    # 这是一个待补充的测试函数，需要根据具体情况添加测试代码
    # 检查 _partial_dependence_brute 或 _partial_dependence_recursion 返回的结果，
    # 是否等同于手动设置目标特征为给定值，并计算所有样本的平均预测值。
    # 这也检查 brute 和 recursion 方法是否给出相同的输出。
    # 需要注意的是，即使在训练集上，brute 和 recursion 方法并不总是严格等价，
    # 特别是当慢速方法生成在输入特征联合分布中具有低质量的不真实样本时，以及一些特征存在依赖关系时。
    # 因此在检查中允许较高的容差。

    X, y = make_regression(random_state=0, n_features=5, n_informative=5)
    # 对于 GBDT 的 'init' 估计器（这里是平均预测值），由于技术原因，递归方法中没有被考虑。
    # 我们将均值设置为0，以确保这个 'bug' 不产生任何影响。
    y = y - y.mean()
    est.fit(X, y)

    # 将目标特征设置为 0.5 和 123
    features = np.array([target_feature], dtype=np.intp)
    grid = np.array([[0.5], [123]])

    if method == "brute":
        pdp, predictions = _partial_dependence_brute(
            est, grid, features, X, response_method="auto"
        )
    else:
        pdp = _partial_dependence_recursion(est, grid, features)

    mean_predictions = []
    for val in (0.5, 123):
        X_ = X.copy()
        X_[:, target_feature] = val
        mean_predictions.append(est.predict(X_).mean())

    pdp = pdp[0]  # (shape is (1, 2) so make it (2,))

    # 对于 recursion 方法，允许更大的误差容差
    rtol = 1e-1 if method == "recursion" else 1e-3
    # 断言 pdp 和 mean_predictions 数组的所有元素在指定容差范围内相等
    assert np.allclose(pdp, mean_predictions, rtol=rtol)
# 使用 pytest 的装饰器标记参数化测试，仅测试一个种子值
@pytest.mark.parametrize("seed", range(1))
def test_recursion_decision_tree_vs_forest_and_gbdt(seed):
    # 确保递归方法在 DecisionTreeRegressor 和 GradientBoostingRegressor 或
    # RandomForestRegressor 上给出相同结果，且具有相同的参数设置。

    # 创建一个特定种子的随机数生成器
    rng = np.random.RandomState(seed)

    # 创建一个完全随机的数据集，以避免相关特征
    n_samples = 1000
    n_features = 5
    X = rng.randn(n_samples, n_features)  # 随机特征数据
    y = rng.randn(n_samples) * 10  # 随机目标数据

    # 对于 GBDT，'init' 估计器（这里是平均预测）由于技术原因不会被递归方法考虑，因此设置均值为0以避免此问题
    y = y - y.mean()

    # 设置决策树的最大深度，不要设置得太高以避免相同增益但不同特征的分割
    max_depth = 5

    tree_seed = 0
    forest = RandomForestRegressor(
        n_estimators=1,
        max_features=None,
        bootstrap=False,
        max_depth=max_depth,
        random_state=tree_seed,
    )
    # 森林将使用 ensemble.base._set_random_states 来设置树的子估计器的随机状态。
    # 在这里模拟这个过程以确保估计器是等价的。
    equiv_random_state = check_random_state(tree_seed).randint(np.iinfo(np.int32).max)
    gbdt = GradientBoostingRegressor(
        n_estimators=1,
        learning_rate=1,
        criterion="squared_error",
        max_depth=max_depth,
        random_state=equiv_random_state,
    )
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=equiv_random_state)

    # 分别拟合三种模型
    forest.fit(X, y)
    gbdt.fit(X, y)
    tree.fit(X, y)

    # 检查：如果三种树模型不相同，则偏依赖值也不会相等
    try:
        assert_is_subtree(tree.tree_, gbdt[0, 0].tree_)
        assert_is_subtree(tree.tree_, forest[0].tree_)
    except AssertionError:
        # 对于某些原因，32位系统上的树不完全相等，因此偏依赖值也不会相等。
        assert _IS_32BIT, "this should only fail on 32 bit platforms"
        return

    # 创建一个网格用于计算偏依赖
    grid = rng.randn(50).reshape(-1, 1)
    for f in range(n_features):
        features = np.array([f], dtype=np.intp)

        # 使用递归方法计算偏依赖值
        pdp_forest = _partial_dependence_recursion(forest, grid, features)
        pdp_gbdt = _partial_dependence_recursion(gbdt, grid, features)
        pdp_tree = _partial_dependence_recursion(tree, grid, features)

        # 断言偏依赖值在三种模型中的相等性
        np.testing.assert_allclose(pdp_gbdt, pdp_tree)
        np.testing.assert_allclose(pdp_forest, pdp_tree)


@pytest.mark.parametrize(
    "est",
    (
        GradientBoostingClassifier(random_state=0),
        HistGradientBoostingClassifier(random_state=0),
    ),
)
@pytest.mark.parametrize("target_feature", (0, 1, 2, 3, 4, 5))
def test_recursion_decision_function(est, target_feature):
    # 确保递归方法（隐式使用 decision_function）可以正确工作
    # 使用 make_classification 生成一个二分类数据集 X 和标签 y
    X, y = make_classification(n_classes=2, n_clusters_per_class=1, random_state=1)
    # 断言确保标签 y 的均值为 0.5，即初始估算器总是预测为 0
    assert np.mean(y) == 0.5  # make sure the init estimator predicts 0 anyway

    # 使用 est 拟合数据集 X 和标签 y
    est.fit(X, y)

    # 调用 partial_dependence 函数计算第一种方法（递归法）的偏依赖预测
    preds_1 = partial_dependence(
        est,
        X,
        [target_feature],
        response_method="decision_function",
        method="recursion",
        kind="average",
    )
    # 调用 partial_dependence 函数计算第二种方法（暴力法）的偏依赖预测
    preds_2 = partial_dependence(
        est,
        X,
        [target_feature],
        response_method="decision_function",
        method="brute",
        kind="average",
    )

    # 使用 assert_allclose 函数断言两种方法得到的偏依赖预测结果在给定的容差范围内相等
    assert_allclose(preds_1["average"], preds_2["average"], atol=1e-7)
@pytest.mark.parametrize(
    "est",
    (
        LinearRegression(),
        GradientBoostingRegressor(random_state=0),
        HistGradientBoostingRegressor(
            random_state=0, min_samples_leaf=1, max_leaf_nodes=None, max_iter=1
        ),
        DecisionTreeRegressor(random_state=0),
    ),
)
@pytest.mark.parametrize("power", (1, 2))
def test_partial_dependence_easy_target(est, power):
    # 如果目标变量 y 明显依赖于单个特征（线性或二次），那么该特征的偏依赖应该反映出来。
    # 在这里，我们拟合一个线性回归模型（如果需要，使用多项式特征），并计算 r_squared 来检查偏依赖是否正确反映目标变量。

    rng = np.random.RandomState(0)
    n_samples = 200
    target_variable = 2
    X = rng.normal(size=(n_samples, 5))
    y = X[:, target_variable] ** power

    est.fit(X, y)

    pdp = partial_dependence(
        est, features=[target_variable], X=X, grid_resolution=1000, kind="average"
    )

    new_X = pdp["grid_values"][0].reshape(-1, 1)
    new_y = pdp["average"][0]
    # 如果需要，添加多项式特征
    new_X = PolynomialFeatures(degree=power).fit_transform(new_X)

    lr = LinearRegression().fit(new_X, new_y)
    r2 = r2_score(new_y, lr.predict(new_X))

    assert r2 > 0.99


@pytest.mark.parametrize(
    "Estimator",
    (
        sklearn.tree.DecisionTreeClassifier,
        sklearn.tree.ExtraTreeClassifier,
        sklearn.ensemble.ExtraTreesClassifier,
        sklearn.neighbors.KNeighborsClassifier,
        sklearn.neighbors.RadiusNeighborsClassifier,
        sklearn.ensemble.RandomForestClassifier,
    ),
)
def test_multiclass_multioutput(Estimator):
    # 确保对多类多输出分类器引发错误

    # 创建多类多输出数据集
    X, y = make_classification(n_classes=3, n_clusters_per_class=1, random_state=0)
    y = np.array([y, y]).T

    est = Estimator()
    est.fit(X, y)

    with pytest.raises(
        ValueError, match="Multiclass-multioutput estimators are not supported"
    ):
        partial_dependence(est, X, [0])


class NoPredictProbaNoDecisionFunction(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        # 模拟存在某些类别
        self.classes_ = [0, 1]
        return self


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize(
    "estimator, params, err_msg",
    [
        # 第一个元组：使用KMeans模型，设置随机种子为0，n_init参数自动选择，提示错误信息为'estimator'必须是拟合过的回归器或分类器
        (
            KMeans(random_state=0, n_init="auto"),
            {"features": [0]},
            "'estimator' must be a fitted regressor or classifier",
        ),
        # 第二个元组：使用LinearRegression模型，指定特征为第一个特征，response_method参数设置为'predict_proba'，但该参数对回归器无效，提示信息说明该参数被忽略
        (
            LinearRegression(),
            {"features": [0], "response_method": "predict_proba"},
            "The response_method parameter is ignored for regressors",
        ),
        # 第三个元组：使用GradientBoostingClassifier模型，设置随机种子为0，特征为第一个特征，response_method参数设置为'predict_proba'，method参数设置为'recursion'，提示错误信息指出在'recursion'方法下，response_method必须是'decision_function'
        (
            GradientBoostingClassifier(random_state=0),
            {
                "features": [0],
                "response_method": "predict_proba",
                "method": "recursion",
            },
            "'recursion' method, the response_method must be 'decision_function'",
        ),
        # 第四个元组：使用GradientBoostingClassifier模型，设置随机种子为0，特征为第一个特征，response_method参数设置为'predict_proba'，method参数设置为'auto'，同样提示'recursion'方法下，response_method必须是'decision_function'
        (
            GradientBoostingClassifier(random_state=0),
            {"features": [0], "response_method": "predict_proba", "method": "auto"},
            "'recursion' method, the response_method must be 'decision_function'",
        ),
        # 第五个元组：使用LinearRegression模型，指定特征为第一个特征，method参数设置为'recursion'，kind参数设置为'individual'，提示说明'recursion'方法只在'kind'设置为'average'时有效
        (
            LinearRegression(),
            {"features": [0], "method": "recursion", "kind": "individual"},
            "The 'recursion' method only applies when 'kind' is set to 'average'",
        ),
        # 第六个元组：使用LinearRegression模型，指定特征为第一个特征，method参数设置为'recursion'，kind参数设置为'both'，同样提示'recursion'方法只在'kind'设置为'average'时有效
        (
            LinearRegression(),
            {"features": [0], "method": "recursion", "kind": "both"},
            "The 'recursion' method only applies when 'kind' is set to 'average'",
        ),
        # 第七个元组：使用LinearRegression模型，指定特征为第一个特征，method参数设置为'recursion'，提示说明只有特定的估算器支持'recursion'方法
        (
            LinearRegression(),
            {"features": [0], "method": "recursion"},
            "Only the following estimators support the 'recursion' method:",
        ),
    ],
def test_partial_dependence_error(estimator, params, err_msg):
    # 使用 make_classification 创建一个随机的分类数据集 X, y
    X, y = make_classification(random_state=0)
    # 使用给定的 estimator 拟合数据集
    estimator.fit(X, y)

    # 使用 pytest 检查是否会抛出 ValueError 异常，并检查异常消息是否匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        # 调用 partial_dependence 函数，传入 estimator 和 X 数据，以及额外的参数 params
        partial_dependence(estimator, X, **params)


@pytest.mark.parametrize(
    "estimator", [LinearRegression(), GradientBoostingClassifier(random_state=0)]
)
@pytest.mark.parametrize("features", [-1, 10000])
def test_partial_dependence_unknown_feature_indices(estimator, features):
    # 使用 make_classification 创建一个随机的分类数据集 X, y
    X, y = make_classification(random_state=0)
    # 使用给定的 estimator 拟合数据集
    estimator.fit(X, y)

    # 设置错误消息字符串
    err_msg = "all features must be in"
    # 使用 pytest 检查是否会抛出 ValueError 异常，并检查异常消息是否匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        # 调用 partial_dependence 函数，传入 estimator、X 数据和 features 参数
        partial_dependence(estimator, X, [features])


@pytest.mark.parametrize(
    "estimator", [LinearRegression(), GradientBoostingClassifier(random_state=0)]
)
def test_partial_dependence_unknown_feature_string(estimator):
    # 导入 pandas 库，如果不存在则跳过这个测试
    pd = pytest.importorskip("pandas")
    # 使用 make_classification 创建一个随机的分类数据集 X, y
    X, y = make_classification(random_state=0)
    # 将 X 转换为 DataFrame
    df = pd.DataFrame(X)
    # 使用给定的 estimator 拟合 DataFrame 数据集
    estimator.fit(df, y)

    # 设置 features 为一个未知列名的列表
    features = ["random"]
    # 设置错误消息字符串
    err_msg = "A given column is not a column of the dataframe"
    # 使用 pytest 检查是否会抛出 ValueError 异常，并检查异常消息是否匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        # 调用 partial_dependence 函数，传入 estimator、DataFrame 和 features 参数
        partial_dependence(estimator, df, features)


@pytest.mark.parametrize(
    "estimator", [LinearRegression(), GradientBoostingClassifier(random_state=0)]
)
def test_partial_dependence_X_list(estimator):
    # 检查是否接受类似数组的对象
    # 使用 make_classification 创建一个随机的分类数据集 X, y
    X, y = make_classification(random_state=0)
    # 使用给定的 estimator 拟合数据集
    estimator.fit(X, y)
    # 调用 partial_dependence 函数，传入 estimator、X 数据、特定的 features 和 kind 参数
    partial_dependence(estimator, list(X), [0], kind="average")


def test_warning_recursion_non_constant_init():
    # 确保将非常量 init 参数传递给 GBDT 模型并使用递归方法会产生警告
    # 创建一个 GBDT 分类器，init 参数为一个非常量的 DummyClassifier 对象，随机种子为 0
    gbc = GradientBoostingClassifier(init=DummyClassifier(), random_state=0)
    # 使用给定的数据集 X, y 拟合 GBDT 分类器
    gbc.fit(X, y)

    # 使用 pytest 检查是否会产生 UserWarning 警告，并检查警告消息是否匹配特定的字符串
    with pytest.warns(
        UserWarning, match="Using recursion method with a non-constant init predictor"
    ):
        # 调用 partial_dependence 函数，传入 gbc、X 数据、特定的 features 和 method 参数
        partial_dependence(gbc, X, [0], method="recursion", kind="average")

    # 再次使用 pytest 检查是否会产生 UserWarning 警告，并检查警告消息是否匹配特定的字符串
    with pytest.warns(
        UserWarning, match="Using recursion method with a non-constant init predictor"
    ):
        # 调用 partial_dependence 函数，传入 gbc、X 数据、特定的 features 和 method 参数
        partial_dependence(gbc, X, [0], method="recursion", kind="average")


def test_partial_dependence_sample_weight_of_fitted_estimator():
    # 测试部分依赖与对角线近乎完美相关性，当样本权重强调 y = x 预测时
    # 非回归测试用例 #13193
    # TODO: 扩展到 HistGradientBoosting 一旦支持 sample_weight
    N = 1000
    rng = np.random.RandomState(123456)
    mask = rng.randint(2, size=N, dtype=bool)

    x = rng.rand(N)
    # 在 mask 上设置 y = x，而在非 mask 上设置 y = -x
    y = x.copy()
    y[~mask] = -y[~mask]
    X = np.c_[mask, x]
    # 设置样本权重以强调 y = x 的数据点
    sample_weight = np.ones(N)
    sample_weight[mask] = 1000.0

    # 创建一个梯度提升回归器，设置 n_estimators 和随机种子
    clf = GradientBoostingRegressor(n_estimators=10, random_state=1)
    # 使用给定的 X, y 和 sample_weight 拟合回归器
    clf.fit(X, y, sample_weight=sample_weight)
    # 使用偏依赖分析计算分类器 clf 在特征集 X 上特征索引为 1 的特征的偏依赖
    pdp = partial_dependence(clf, X, features=[1], kind="average")
    
    # 断言偏依赖分析结果中 "average" 和 "grid_values" 之间的皮尔逊相关系数大于 0.99
    assert np.corrcoef(pdp["average"], pdp["grid_values"])[0, 1] > 0.99
# 定义一个测试函数，测试 HistGradientBoostingRegressor 不支持带样本权重的情况
def test_hist_gbdt_sw_not_supported():
    # 创建 HistGradientBoostingRegressor 分类器对象，设定随机种子为1
    clf = HistGradientBoostingRegressor(random_state=1)
    # 使用均匀权重训练分类器
    clf.fit(X, y, sample_weight=np.ones(len(X)))

    # 使用 pytest 来检查是否抛出 NotImplementedError 异常，匹配指定错误信息
    with pytest.raises(
        NotImplementedError, match="does not support partial dependence"
    ):
        # 调用 partial_dependence 函数，尝试计算模型 clf 在数据集 X 上特征[1]的部分依赖
        partial_dependence(clf, X, features=[1])


# 定义一个测试函数，验证 partial_dependence 函数支持数据流水线操作
def test_partial_dependence_pipeline():
    # 载入鸢尾花数据集
    iris = load_iris()

    # 创建数据预处理器对象：标准化器
    scaler = StandardScaler()
    # 创建分类器对象：虚拟分类器
    clf = DummyClassifier(random_state=42)
    # 创建流水线对象，依次执行标准化和分类操作
    pipe = make_pipeline(scaler, clf)

    # 在标准化后的数据上训练分类器 clf
    clf.fit(scaler.fit_transform(iris.data), iris.target)
    # 在原始数据上训练流水线 pipe
    pipe.fit(iris.data, iris.target)

    # 设定要计算部分依赖的特征
    features = 0
    # 计算流水线 pipe 在数据集 iris.data 上特征[0]的部分依赖
    pdp_pipe = partial_dependence(
        pipe, iris.data, features=[features], grid_resolution=10, kind="average"
    )
    # 计算分类器 clf 在标准化后的数据上特征[0]的部分依赖
    pdp_clf = partial_dependence(
        clf,
        scaler.transform(iris.data),
        features=[features],
        grid_resolution=10,
        kind="average",
    )

    # 使用 assert_allclose 函数检查两个部分依赖结果的一致性
    assert_allclose(pdp_pipe["average"], pdp_clf["average"])
    # 根据标准化器的缩放和平均值，调整流水线部分依赖结果的网格值
    assert_allclose(
        pdp_pipe["grid_values"][0],
        pdp_clf["grid_values"][0] * scaler.scale_[features] + scaler.mean_[features],
    )


# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数，测试数据帧上的部分依赖
@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(max_iter=1000, random_state=0),
        GradientBoostingClassifier(random_state=0, n_estimators=5),
    ],
    ids=["estimator-brute", "estimator-recursion"],
)
@pytest.mark.parametrize(
    "preprocessor",
    [
        None,
        make_column_transformer(
            (StandardScaler(), [iris.feature_names[i] for i in (0, 2)]),
            (RobustScaler(), [iris.feature_names[i] for i in (1, 3)]),
        ),
        make_column_transformer(
            (StandardScaler(), [iris.feature_names[i] for i in (0, 2)]),
            remainder="passthrough",
        ),
    ],
    ids=["None", "column-transformer", "column-transformer-passthrough"],
)
@pytest.mark.parametrize(
    "features",
    [[0, 2], [iris.feature_names[i] for i in (0, 2)]],
    ids=["features-integer", "features-string"],
)
# 定义测试函数，验证数据帧和流水线操作下的部分依赖计算
def test_partial_dependence_dataframe(estimator, preprocessor, features):
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建 DataFrame 对象，对数据进行标准化处理，列名为 iris.feature_names
    df = pd.DataFrame(scale(iris.data), columns=iris.feature_names)

    # 创建流水线对象，依次执行预处理器和分类器
    pipe = make_pipeline(preprocessor, estimator)
    # 在标准化后的数据上训练流水线
    pipe.fit(df, iris.target)

    # 计算流水线 pipe 在数据集 df 上指定特征的部分依赖
    pdp_pipe = partial_dependence(
        pipe, df, features=features, grid_resolution=10, kind="average"
    )

    # 如果存在预处理器，则对 DataFrame 进行克隆并变换
    if preprocessor is not None:
        X_proc = clone(preprocessor).fit_transform(df)
        features_clf = [0, 1]
    else:
        X_proc = df
        features_clf = [0, 2]

    # 克隆分类器对象，对处理后的数据 X_proc 进行训练
    clf = clone(estimator).fit(X_proc, iris.target)
    pdp_clf = partial_dependence(
        clf,
        X_proc,
        features=features_clf,
        method="brute",
        grid_resolution=10,
        kind="average",
    )

# 使用偏依赖方法计算分类器 `clf` 对处理后数据集 `X_proc` 的部分依赖性。
# `features_clf` 是用于计算部分依赖性的特征列表，`method="brute"` 表示使用暴力方法，`grid_resolution=10` 表示网格分辨率为10，`kind="average"` 表示计算平均部分依赖性。


    assert_allclose(pdp_pipe["average"], pdp_clf["average"])

# 断言检查管道 `pdp_pipe` 中的平均部分依赖值与 `pdp_clf` 中的平均部分依赖值是否接近（数值上相等）。


    if preprocessor is not None:
        scaler = preprocessor.named_transformers_["standardscaler"]
        assert_allclose(
            pdp_pipe["grid_values"][1],
            pdp_clf["grid_values"][1] * scaler.scale_[1] + scaler.mean_[1],
        )
    else:
        assert_allclose(pdp_pipe["grid_values"][1], pdp_clf["grid_values"][1])

# 如果预处理器 `preprocessor` 存在，则获取标准化变换器 `standardscaler`，并断言管道 `pdp_pipe` 和 `pdp_clf` 中网格值的第二个元素是否在经过标准化调整后接近。
# 否则，断言管道 `pdp_pipe` 和 `pdp_clf` 中网格值的第二个元素是否数值上相等。
# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
@pytest.mark.parametrize(
    "features, expected_pd_shape",
    [
        (0, (3, 10)),  # 测试特征为整数0时的期望输出形状
        (iris.feature_names[0], (3, 10)),  # 测试特征为第一个特征名时的期望输出形状
        ([0, 2], (3, 10, 10)),  # 测试特征为整数列表[0, 2]时的期望输出形状
        ([iris.feature_names[i] for i in (0, 2)], (3, 10, 10)),  # 测试特征为特征名列表时的期望输出形状
        ([True, False, True, False], (3, 10, 10)),  # 测试特征为布尔值列表时的期望输出形状
    ],
    ids=["scalar-int", "scalar-str", "list-int", "list-str", "mask"],  # 为测试用例指定标识符
)
def test_partial_dependence_feature_type(features, expected_pd_shape):
    # 导入 pandas 库，如果导入失败则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建包含鸢尾花数据的 DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # 创建数据预处理器和管道
    preprocessor = make_column_transformer(
        (StandardScaler(), [iris.feature_names[i] for i in (0, 2)]),
        (RobustScaler(), [iris.feature_names[i] for i in (1, 3)]),
    )
    pipe = make_pipeline(
        preprocessor, LogisticRegression(max_iter=1000, random_state=0)
    )
    # 在数据上拟合管道
    pipe.fit(df, iris.target)
    # 计算偏依赖图
    pdp_pipe = partial_dependence(
        pipe, df, features=features, grid_resolution=10, kind="average"
    )
    # 断言偏依赖图的形状符合预期
    assert pdp_pipe["average"].shape == expected_pd_shape
    # 断言偏依赖图的网格值数量符合预期
    assert len(pdp_pipe["grid_values"]) == len(pdp_pipe["average"].shape) - 1


# 参数化测试用例，测试未拟合的估计器
@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        LogisticRegression(),
        GradientBoostingRegressor(),
        GradientBoostingClassifier(),
    ],
)
def test_partial_dependence_unfitted(estimator):
    X = iris.data
    # 创建数据预处理器和管道
    preprocessor = make_column_transformer(
        (StandardScaler(), [0, 2]), (RobustScaler(), [1, 3])
    )
    pipe = make_pipeline(preprocessor, estimator)
    # 使用 pytest.raises 断言捕获 NotFittedError 异常
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        partial_dependence(pipe, X, features=[0, 2], grid_resolution=10)
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        partial_dependence(estimator, X, features=[0, 2], grid_resolution=10)


# 参数化测试用例，测试 kind 为 average 和 individual 时的偏依赖图
@pytest.mark.parametrize(
    "Estimator, data",
    [
        (LinearRegression, multioutput_regression_data),
        (LogisticRegression, binary_classification_data),
    ],
)
def test_kind_average_and_average_of_individual(Estimator, data):
    est = Estimator()
    (X, y), n_targets = data
    est.fit(X, y)

    # 计算平均偏依赖图和个体偏依赖图
    pdp_avg = partial_dependence(est, X=X, features=[1, 2], kind="average")
    pdp_ind = partial_dependence(est, X=X, features=[1, 2], kind="individual")
    avg_ind = np.mean(pdp_ind["individual"], axis=1)
    # 断言平均偏依赖图和个体偏依赖图的平均值一致
    assert_allclose(avg_ind, pdp_avg["average"])


# 参数化测试用例，测试 kind 为 individual 时 sample_weight 的影响
@pytest.mark.parametrize(
    "Estimator, data",
    [
        (LinearRegression, multioutput_regression_data),
        (LogisticRegression, binary_classification_data),
    ],
)
def test_partial_dependence_kind_individual_ignores_sample_weight(Estimator, data):
    """Check that `sample_weight` does not have any effect on reported ICE."""
    est = Estimator()
    (X, y), n_targets = data
    sample_weight = np.arange(X.shape[0])
    est.fit(X, y)

    # 计算个体偏依赖图
    pdp_nsw = partial_dependence(est, X=X, features=[1, 2], kind="individual")
    # 使用 partial_dependence 函数计算偏依赖，针对特征索引为 1 和 2 的个体偏依赖
    pdp_sw = partial_dependence(
        est, X=X, features=[1, 2], kind="individual", sample_weight=sample_weight
    )
    # 断言验证无样本权重情况下的个体偏依赖和有样本权重情况下的个体偏依赖是否近似相等
    assert_allclose(pdp_nsw["individual"], pdp_sw["individual"])
    # 断言验证无样本权重情况下的网格值和有样本权重情况下的网格值是否近似相等
    assert_allclose(pdp_nsw["grid_values"], pdp_sw["grid_values"])
@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),  # 使用线性回归模型
        LogisticRegression(),  # 使用逻辑回归模型
        RandomForestRegressor(),  # 使用随机森林回归模型
        GradientBoostingClassifier(),  # 使用梯度提升分类器模型
    ],
)
@pytest.mark.parametrize("non_null_weight_idx", [0, 1, -1])
def test_partial_dependence_non_null_weight_idx(estimator, non_null_weight_idx):
    """检查如果我们使用一个样本权重数组，其中只有一个索引的权重为1，其他为0，
    那么使用这个样本权重计算出来的平均偏依赖值应该等于相应索引处的个体偏依赖值。
    """
    X, y = iris.data, iris.target
    preprocessor = make_column_transformer(
        (StandardScaler(), [0, 2]), (RobustScaler(), [1, 3])
    )
    pipe = make_pipeline(preprocessor, estimator).fit(X, y)

    sample_weight = np.zeros_like(y)
    sample_weight[non_null_weight_idx] = 1
    pdp_sw = partial_dependence(
        pipe,
        X,
        [2, 3],
        kind="average",
        sample_weight=sample_weight,
        grid_resolution=10,
    )
    pdp_ind = partial_dependence(pipe, X, [2, 3], kind="individual", grid_resolution=10)
    output_dim = 1 if is_regressor(pipe) else len(np.unique(y))
    for i in range(output_dim):
        assert_allclose(
            pdp_ind["individual"][i][non_null_weight_idx],
            pdp_sw["average"][i],
        )


@pytest.mark.parametrize(
    "Estimator, data",
    [
        (LinearRegression, multioutput_regression_data),  # 使用线性回归模型和多输出回归数据
        (LogisticRegression, binary_classification_data),  # 使用逻辑回归模型和二分类数据
    ],
)
def test_partial_dependence_equivalence_equal_sample_weight(Estimator, data):
    """检查 `sample_weight=None` 是否等效于所有权重相等的情况。"""

    est = Estimator()
    (X, y), n_targets = data
    est.fit(X, y)

    sample_weight, params = None, {"X": X, "features": [1, 2], "kind": "average"}
    pdp_sw_none = partial_dependence(est, **params, sample_weight=sample_weight)
    sample_weight = np.ones(len(y))
    pdp_sw_unit = partial_dependence(est, **params, sample_weight=sample_weight)
    assert_allclose(pdp_sw_none["average"], pdp_sw_unit["average"])
    sample_weight = 2 * np.ones(len(y))
    pdp_sw_doubling = partial_dependence(est, **params, sample_weight=sample_weight)
    assert_allclose(pdp_sw_none["average"], pdp_sw_doubling["average"])


def test_partial_dependence_sample_weight_size_error():
    """检查当 `sample_weight` 的大小与 `X` 和 `y` 不一致时是否会引发错误。"""
    est = LogisticRegression()
    (X, y), n_targets = binary_classification_data
    sample_weight = np.ones_like(y)
    est.fit(X, y)

    with pytest.raises(ValueError, match="sample_weight.shape =="):
        partial_dependence(
            est, X, features=[0], sample_weight=sample_weight[1:], grid_resolution=10
        )


def test_partial_dependence_sample_weight_with_recursion():
    """检查当 `sample_weight` 与 `"recursion"` 方法一起使用时是否会引发错误。"""
    # 创建一个随机森林回归器对象
    est = RandomForestRegressor()
    # 从 regression_data 中解包得到特征数据 X 和目标数据 y，同时获取目标数 n_targets
    (X, y), n_targets = regression_data
    # 创建一个与 y 相同形状的样本权重数组
    sample_weight = np.ones_like(y)
    # 使用特征数据 X 和目标数据 y 进行随机森林回归器的拟合，使用 sample_weight 作为样本权重
    est.fit(X, y, sample_weight=sample_weight)

    # 使用 pytest 模块进行异常断言，期望捕获 ValueError 异常，并匹配指定的错误消息
    with pytest.raises(ValueError, match="'recursion' method can only be applied when"):
        # 调用 partial_dependence 函数，对 est 模型的特征进行部分依赖分析
        # 仅针对第一个特征（features=[0]），使用 method="recursion" 和指定的样本权重进行计算
        partial_dependence(
            est, X, features=[0], method="recursion", sample_weight=sample_weight
        )
def test_mixed_type_categorical():
    """Check that we raise a proper error when a column has mixed types and
    the sorting of `np.unique` will fail."""
    # 创建一个包含混合类型和缺失值的 NumPy 数组，使用对象类型保留所有数据类型
    X = np.array(["A", "B", "C", np.nan], dtype=object).reshape(-1, 1)
    # 创建目标变量数组
    y = np.array([0, 1, 0, 1])

    # 导入需要的库和模块
    from sklearn.preprocessing import OrdinalEncoder

    # 使用管道创建分类器，包括特殊处理编码器和逻辑回归器，对 X 和 y 进行训练
    clf = make_pipeline(
        OrdinalEncoder(encoded_missing_value=-1),  # 使用自定义的缺失值编码器
        LogisticRegression(),  # 使用逻辑回归模型
    ).fit(X, y)

    # 使用 pytest 来测试是否引发了预期的 ValueError 异常，验证混合类型列的错误信息
    with pytest.raises(ValueError, match="The column #0 contains mixed data types"):
        partial_dependence(clf, X, features=[0])
```