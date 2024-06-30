# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\tests\test_sequential.py`

```
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import CSR_CONTAINERS

# 定义一个测试函数，测试当 n_features_to_select 大于总特征数时是否抛出异常
def test_bad_n_features_to_select():
    n_features = 5
    # 创建回归数据集 X 和 y
    X, y = make_regression(n_features=n_features)
    # 创建一个顺序特征选择器，期望选择的特征数为 n_features
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_features)
    # 断言会抛出 ValueError 异常，且异常信息包含 "n_features_to_select must be < n_features"
    with pytest.raises(ValueError, match="n_features_to_select must be < n_features"):
        sfs.fit(X, y)


# 参数化测试，测试不同的特征选择方向和选择特征数的行为
@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize("n_features_to_select", (1, 5, 9, "auto"))
def test_n_features_to_select(direction, n_features_to_select):
    # 确保 n_features_to_select 被正确地应用

    n_features = 10
    # 创建回归数据集 X 和 y
    X, y = make_regression(n_features=n_features, random_state=0)
    # 创建顺序特征选择器，根据参数设置选择特征数
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features_to_select,
        direction=direction,
        cv=2,
    )
    # 对数据进行特征选择
    sfs.fit(X, y)

    if n_features_to_select == "auto":
        n_features_to_select = n_features // 2

    # 断言选择的特征数目正确
    assert sfs.get_support(indices=True).shape[0] == n_features_to_select
    # 断言 sfs 内部记录的 n_features_to_select_ 与设置的一致
    assert sfs.n_features_to_select_ == n_features_to_select
    # 断言转换后的数据形状正确
    assert sfs.transform(X).shape[1] == n_features_to_select


# 参数化测试，测试 n_features_to_select="auto" 的行为
@pytest.mark.parametrize("direction", ("forward", "backward"))
def test_n_features_to_select_auto(direction):
    """Check the behaviour of `n_features_to_select="auto"` with different
    values for the parameter `tol`.
    """

    n_features = 10
    tol = 1e-3
    # 创建回归数据集 X 和 y
    X, y = make_regression(n_features=n_features, random_state=0)
    # 创建顺序特征选择器，使用 n_features_to_select="auto"，并设置 tol 参数
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select="auto",
        tol=tol,
        direction=direction,
        cv=2,
    )
    # 对数据进行特征选择
    sfs.fit(X, y)

    max_features_to_select = n_features - 1

    # 断言选择的特征数不超过 max_features_to_select
    assert sfs.get_support(indices=True).shape[0] <= max_features_to_select
    # 断言 sfs 内部记录的 n_features_to_select_ 不超过 max_features_to_select
    assert sfs.n_features_to_select_ <= max_features_to_select
    # 断言转换后的数据形状不超过 max_features_to_select
    assert sfs.transform(X).shape[1] <= max_features_to_select
    # 断言选择的特征数目与 sfs 内部记录的 n_features_to_select_ 一致
    assert sfs.get_support(indices=True).shape[0] == sfs.n_features_to_select_


# 参数化测试，测试特征选择的停止条件
@pytest.mark.parametrize("direction", ("forward", "backward"))
def test_n_features_to_select_stopping_criterion(direction):
    """Check the behaviour stopping criterion for feature selection
    depending on the values of `n_features_to_select` and `tol`.

    When `direction` is `'forward'`, select a new features at random
    among those not currently selected in selector.support_,
    build a new version of the data that includes all the features
    in selector.support_ + this newly selected feature.
    And check that the cross-validation score of the model trained on
    this new dataset variant is lower than the model with
    the selected forward selected features or at least does not improve
    by more than the tol margin.

    When `direction` is `'backward'`, instead of adding a new feature
    to selector.support_, try to remove one of those selected features at random
    And check that the cross-validation score is either decreasing or
    not improving by more than the tol margin.
    """

    # 生成一个具有50个特征和10个信息特征的回归数据集
    X, y = make_regression(n_features=50, n_informative=10, random_state=0)

    tol = 1e-3  # 设置容差阈值

    # 初始化顺序特征选择器对象
    sfs = SequentialFeatureSelector(
        LinearRegression(),  # 使用线性回归作为基础模型
        n_features_to_select="auto",  # 自动选择特征数量
        tol=tol,  # 设置容差阈值
        direction=direction,  # 特征选择方向，可能是'forward'或'backward'
        cv=2,  # 交叉验证折数
    )
    sfs.fit(X, y)  # 在数据集上拟合顺序特征选择器
    selected_X = sfs.transform(X)  # 获取选择后的特征集

    rng = np.random.RandomState(0)  # 设置随机种子为0

    # 确定未被选择的特征候选集合
    added_candidates = list(set(range(X.shape[1])) - set(sfs.get_support(indices=True)))
    # 添加新选择的特征到选择后的特征集中
    added_X = np.hstack(
        [
            selected_X,
            (X[:, rng.choice(added_candidates)])[:, np.newaxis],
        ]
    )

    # 随机选择一个要移除的已选择特征
    removed_candidate = rng.choice(list(range(sfs.n_features_to_select_)))
    # 从选择后的特征集中移除指定的特征
    removed_X = np.delete(selected_X, removed_candidate, axis=1)

    # 计算简单交叉验证分数，使用全部特征的数据集
    plain_cv_score = cross_val_score(LinearRegression(), X, y, cv=2).mean()
    # 计算顺序特征选择后的交叉验证分数
    sfs_cv_score = cross_val_score(LinearRegression(), selected_X, y, cv=2).mean()
    # 计算添加新特征后的交叉验证分数
    added_cv_score = cross_val_score(LinearRegression(), added_X, y, cv=2).mean()
    # 计算移除特征后的交叉验证分数
    removed_cv_score = cross_val_score(LinearRegression(), removed_X, y, cv=2).mean()

    # 断言顺序特征选择后的交叉验证分数应该高于或等于全部特征的交叉验证分数
    assert sfs_cv_score >= plain_cv_score

    if direction == "forward":
        # 当特征选择方向为前向时，确保添加新特征后的交叉验证分数与顺序特征选择后的交叉验证分数的差异小于等于容差阈值
        assert (sfs_cv_score - added_cv_score) <= tol
        # 确保移除特征后的交叉验证分数与顺序特征选择后的交叉验证分数的差异大于等于容差阈值
        assert (sfs_cv_score - removed_cv_score) >= tol
    else:
        # 当特征选择方向为后向时，确保添加新特征后的交叉验证分数与顺序特征选择后的交叉验证分数的差异小于等于容差阈值
        assert (added_cv_score - sfs_cv_score) <= tol
        # 确保移除特征后的交叉验证分数与顺序特征选择后的交叉验证分数的差异小于等于容差阈值
        assert (removed_cv_score - sfs_cv_score) <= tol
@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize(
    "n_features_to_select, expected",
    (
        (0.1, 1),  # 测试选择0.1个特征时，预期结果是1个特征被选中
        (1.0, 10),  # 测试选择1.0个特征时，预期结果是10个特征被选中
        (0.5, 5),   # 测试选择0.5个特征时，预期结果是5个特征被选中
    ),
)
def test_n_features_to_select_float(direction, n_features_to_select, expected):
    # Test passing a float as n_features_to_select
    X, y = make_regression(n_features=10)  # 生成具有10个特征的回归数据集
    sfs = SequentialFeatureSelector(
        LinearRegression(),  # 使用线性回归作为基础评估器
        n_features_to_select=n_features_to_select,  # 设置要选择的特征数量
        direction=direction,  # 设置特征选择的方向（前向或后向）
        cv=2,  # 设置交叉验证的折数
    )
    sfs.fit(X, y)  # 对数据进行特征选择
    assert sfs.n_features_to_select_ == expected  # 断言实际选中的特征数量与预期一致


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize(
    "n_features_to_select, expected_selected_features",
    [
        (2, [0, 2]),  # 当选择2个特征时，预期选择的特征是0和2
        (1, [2]),     # 当选择1个特征时，预期选择的特征是2
    ],
)
def test_sanity(seed, direction, n_features_to_select, expected_selected_features):
    # Basic sanity check: 3 features, only f0 and f2 are correlated with the
    # target, f2 having a stronger correlation than f0. We expect f1 to be
    # dropped, and f2 to always be selected.

    rng = np.random.RandomState(seed)  # 使用指定种子创建随机数生成器
    n_samples = 100
    X = rng.randn(n_samples, 3)  # 生成具有3个特征的随机数据
    y = 3 * X[:, 0] - 10 * X[:, 2]  # 根据特定关系生成目标值

    sfs = SequentialFeatureSelector(
        LinearRegression(),  # 使用线性回归作为基础评估器
        n_features_to_select=n_features_to_select,  # 设置要选择的特征数量
        direction=direction,  # 设置特征选择的方向（前向或后向）
        cv=2,  # 设置交叉验证的折数
    )
    sfs.fit(X, y)  # 对数据进行特征选择
    assert_array_equal(sfs.get_support(indices=True), expected_selected_features)  # 断言实际选择的特征索引与预期一致


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_support(csr_container):
    # Make sure sparse data is supported

    X, y = make_regression(n_features=10)  # 生成具有10个特征的回归数据集
    X = csr_container(X)  # 转换数据为稀疏格式
    sfs = SequentialFeatureSelector(
        LinearRegression(),  # 使用线性回归作为基础评估器
        n_features_to_select="auto",  # 自动选择特征数量
        cv=2,  # 设置交叉验证的折数
    )
    sfs.fit(X, y)  # 对数据进行特征选择
    sfs.transform(X)  # 对数据进行特征选择变换


def test_nan_support():
    # Make sure nans are OK if the underlying estimator supports nans

    rng = np.random.RandomState(0)  # 使用指定种子创建随机数生成器
    n_samples, n_features = 40, 4
    X, y = make_regression(n_samples, n_features, random_state=0)  # 生成回归数据集
    nan_mask = rng.randint(0, 2, size=(n_samples, n_features), dtype=bool)  # 创建包含NaN的掩码
    X[nan_mask] = np.nan  # 将数据中的部分元素设为NaN

    sfs = SequentialFeatureSelector(
        HistGradientBoostingRegressor(),  # 使用梯度提升回归器作为基础评估器
        n_features_to_select="auto",  # 自动选择特征数量
        cv=2,  # 设置交叉验证的折数
    )
    sfs.fit(X, y)  # 对数据进行特征选择
    sfs.transform(X)  # 对数据进行特征选择变换

    with pytest.raises(ValueError, match="Input X contains NaN"):
        # LinearRegression does not support nans
        SequentialFeatureSelector(
            LinearRegression(),  # 使用线性回归作为基础评估器
            n_features_to_select="auto",  # 自动选择特征数量
            cv=2,  # 设置交叉验证的折数
        ).fit(X, y)
    # 创建一个管道对象，包括特征标准化和线性回归模型
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    
    # 创建一个顺序特征选择器对象，基于管道内模型选择自动特征数，并使用交叉验证将其拟合到数据上
    sfs = SequentialFeatureSelector(pipe, n_features_to_select="auto", cv=2)
    
    # 使用顺序特征选择器拟合数据
    sfs.fit(X, y)
    
    # 对输入数据进行特征变换
    sfs.transform(X)
    
    # 创建一个顺序特征选择器对象，使用线性回归模型，选择自动特征数，并使用交叉验证将其拟合到数据上
    sfs = SequentialFeatureSelector(
        LinearRegression(), n_features_to_select="auto", cv=2
    )
    
    # 创建一个新的管道对象，包括标准化和之前创建的顺序特征选择器
    pipe = make_pipeline(StandardScaler(), sfs)
    
    # 将管道对象拟合到输入数据上
    pipe.fit(X, y)
    
    # 对输入数据进行特征变换
    pipe.transform(X)
@pytest.mark.parametrize("n_features_to_select", (2, 3))
# 定义测试函数，参数化 n_features_to_select 可以为 2 或 3
def test_unsupervised_model_fit(n_features_to_select):
    # 确保没有分类标签的模型不会被验证

    # 创建一个包含四个特征的样本集合
    X, y = make_blobs(n_features=4)
    # 创建一个顺序特征选择器，使用 KMeans 作为评估器
    sfs = SequentialFeatureSelector(
        KMeans(n_init=1),
        n_features_to_select=n_features_to_select,
    )
    # 对样本集合 X 进行特征选择器的拟合
    sfs.fit(X)
    # 断言经过特征选择后的样本集合形状的特征数等于 n_features_to_select
    assert sfs.transform(X).shape[1] == n_features_to_select


@pytest.mark.parametrize("y", ("no_validation", 1j, 99.9, np.nan, 3))
# 定义测试函数，参数化 y 可以为非传统的验证标签
def test_no_y_validation_model_fit(y):
    # 确保不接受其他非传统的 y 标签

    # 创建一个包含六个特征的样本集合
    X, clusters = make_blobs(n_features=6)
    # 创建一个顺序特征选择器，使用 KMeans 作为评估器
    sfs = SequentialFeatureSelector(
        KMeans(),
        n_features_to_select=3,
    )

    # 使用 pytest 断言应该引发 TypeError 或 ValueError 异常
    with pytest.raises((TypeError, ValueError)):
        sfs.fit(X, y)


def test_forward_neg_tol_error():
    """Check that we raise an error when tol<0 and direction='forward'"""
    # 检查当 tol<0 且 direction='forward' 时是否引发错误

    # 创建一个包含十个特征的回归问题的样本集合
    X, y = make_regression(n_features=10, random_state=0)
    # 创建一个顺序特征选择器，使用 LinearRegression 作为评估器
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select="auto",
        direction="forward",
        tol=-1e-3,
    )

    # 使用 pytest 断言应该引发 ValueError 异常，并且异常信息应包含 "tol must be positive"
    with pytest.raises(ValueError, match="tol must be positive"):
        sfs.fit(X, y)


def test_backward_neg_tol():
    """Check that SequentialFeatureSelector works negative tol

    non-regression test for #25525
    """
    # 检查 SequentialFeatureSelector 在负 tol 下的工作情况

    # 创建一个包含十个特征的回归问题的样本集合
    X, y = make_regression(n_features=10, random_state=0)
    # 创建一个线性回归评估器
    lr = LinearRegression()
    # 计算原始模型得分
    initial_score = lr.fit(X, y).score(X, y)

    # 创建一个顺序特征选择器，使用 LinearRegression 作为评估器
    sfs = SequentialFeatureSelector(
        lr,
        n_features_to_select="auto",
        direction="backward",
        tol=-1e-3,
    )
    # 使用特征选择器对样本集合 X 进行拟合和转换
    Xr = sfs.fit_transform(X, y)
    # 计算新模型得分
    new_score = lr.fit(Xr, y).score(Xr, y)

    # 断言选择的特征数在 0 和 X.shape[1] 之间
    assert 0 < sfs.get_support().sum() < X.shape[1]
    # 断言新模型得分低于原始模型得分
    assert new_score < initial_score


def test_cv_generator_support():
    """Check that no exception raised when cv is generator

    non-regression test for #25957
    """
    # 检查当 cv 是生成器时不会引发异常

    # 创建一个分类问题的样本集合
    X, y = make_classification(random_state=0)

    # 创建一个分组数组
    groups = np.zeros_like(y, dtype=int)
    groups[y.size // 2 :] = 1

    # 创建一个 LeaveOneGroupOut 交叉验证器
    cv = LeaveOneGroupOut()
    splits = cv.split(X, y, groups=groups)

    # 创建一个 K 近邻分类器
    knc = KNeighborsClassifier(n_neighbors=5)

    # 创建一个顺序特征选择器，使用 K 近邻分类器作为评估器
    sfs = SequentialFeatureSelector(knc, n_features_to_select=5, cv=splits)
    # 使用特征选择器对样本集合 X 进行拟合
    sfs.fit(X, y)
```