# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_lof.py`

```
# 导入必要的库和模块
import re
from math import sqrt

import numpy as np
import pytest

# 导入 sklearn 中的相关模块和函数
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
    check_outlier_corruption,
    parametrize_with_checks,
)
from sklearn.utils.fixes import CSR_CONTAINERS

# 加载鸢尾花数据集并随机排列
rng = check_random_state(0)
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# 定义测试函数 test_lof
def test_lof(global_dtype):
    # 创建示例数据 X，最后两个样本为异常值
    X = np.asarray(
        [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [5, 3], [-4, 2]],
        dtype=global_dtype,
    )

    # 测试 LocalOutlierFactor:
    clf = neighbors.LocalOutlierFactor(n_neighbors=5)
    score = clf.fit(X).negative_outlier_factor_
    assert_array_equal(clf._fit_X, X)

    # 断言最大的异常值分数比最小的正常值分数小
    assert np.min(score[:-2]) > np.max(score[-2:])

    # 测试 predict() 方法
    clf = neighbors.LocalOutlierFactor(contamination=0.25, n_neighbors=5).fit(X)
    expected_predictions = 6 * [1] + 2 * [-1]
    assert_array_equal(clf._predict(), expected_predictions)
    assert_array_equal(clf.fit_predict(X), expected_predictions)

# 定义性能测试函数 test_lof_performance
def test_lof_performance(global_dtype):
    # 生成训练/测试数据
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2).astype(global_dtype, copy=False)
    X_train = X[:100]

    # 生成一些异常的新观测数据
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2)).astype(
        global_dtype, copy=False
    )
    X_test = np.r_[X[100:], X_outliers]
    y_test = np.array([0] * 20 + [1] * 20)

    # 为新颖性检测拟合模型
    clf = neighbors.LocalOutlierFactor(novelty=True).fit(X_train)

    # 预测分数（分数越低，越正常）
    y_pred = -clf.decision_function(X_test)

    # 检查 roc_auc 得分是否好
    assert roc_auc_score(y_test, y_pred) > 0.99

# 定义值检验函数 test_lof_values
def test_lof_values(global_dtype):
    # 创建示例训练数据
    X_train = np.asarray([[1, 1], [1, 2], [2, 1]], dtype=global_dtype)
    clf1 = neighbors.LocalOutlierFactor(
        n_neighbors=2, contamination=0.1, novelty=True
    ).fit(X_train)
    clf2 = neighbors.LocalOutlierFactor(n_neighbors=2, novelty=True).fit(X_train)
    
    # 计算预期的异常值分数
    s_0 = 2.0 * sqrt(2.0) / (1.0 + sqrt(2.0))
    s_1 = (1.0 + sqrt(2)) * (1.0 / (4.0 * sqrt(2.0)) + 1.0 / (2.0 + 2.0 * sqrt(2)))
    
    # 检查 predict() 方法的输出
    assert_allclose(-clf1.negative_outlier_factor_, [s_0, s_1, s_1])
    assert_allclose(-clf2.negative_outlier_factor_, [s_0, s_1, s_1])
    
    # 检查预测单个未在训练集中的样本的输出
    assert_allclose(-clf1.score_samples([[2.0, 2.0]]), [s_0])
    # 使用 assert_allclose 函数检查分类器 clf2 在输入 [[2.0, 2.0]] 上的打分是否接近预期的 s_0 值
    assert_allclose(-clf2.score_samples([[2.0, 2.0]]), [s_0])
    # 检查预测结果：对于已经在训练集中的样本 [[1.0, 1.0]]，使用分类器 clf1 进行打分，判断是否接近 s_1
    assert_allclose(-clf1.score_samples([[1.0, 1.0]]), [s_1])
    # 再次检查预测结果：使用分类器 clf2 对于样本 [[1.0, 1.0]] 进行打分，判断是否接近 s_1
    assert_allclose(-clf2.score_samples([[1.0, 1.0]]), [s_1])
# 测试使用预先计算的距离矩阵的 LOF（局部离群因子）
def test_lof_precomputed(global_dtype, random_state=42):
    """Tests LOF with a distance matrix."""
    # 注意：较小的样本可能导致测试成功的误报
    rng = np.random.RandomState(random_state)
    # 创建一个大小为 (10, 4) 的随机样本矩阵 X，使用全局数据类型
    X = rng.random_sample((10, 4)).astype(global_dtype, copy=False)
    # 创建一个大小为 (3, 4) 的随机样本矩阵 Y，使用全局数据类型
    Y = rng.random_sample((3, 4)).astype(global_dtype, copy=False)
    # 计算 X 内部样本之间的欧氏距离矩阵 DXX
    DXX = metrics.pairwise_distances(X, metric="euclidean")
    # 计算 Y 到 X 样本的欧氏距离矩阵 DYX
    DYX = metrics.pairwise_distances(Y, X, metric="euclidean")
    
    # 使用 LOF（局部离群因子）作为特征矩阵（n_samples by n_features）
    lof_X = neighbors.LocalOutlierFactor(n_neighbors=3, novelty=True)
    lof_X.fit(X)
    # 预测 X 内部样本的离群程度
    pred_X_X = lof_X._predict()
    # 预测 Y 样本在 X 内部样本中的离群程度
    pred_X_Y = lof_X.predict(Y)

    # 使用 LOF（局部离群因子）作为稠密距离矩阵（n_samples by n_samples）
    lof_D = neighbors.LocalOutlierFactor(
        n_neighbors=3, algorithm="brute", metric="precomputed", novelty=True
    )
    lof_D.fit(DXX)
    # 预测基于 DXX 距离矩阵的样本离群程度
    pred_D_X = lof_D._predict()
    # 预测基于 DYX 距离矩阵的样本离群程度
    pred_D_Y = lof_D.predict(DYX)

    # 断言预测结果的相似性
    assert_allclose(pred_X_X, pred_D_X)
    assert_allclose(pred_X_Y, pred_D_Y)


# 测试 n_neighbors 属性设置
def test_n_neighbors_attribute():
    X = iris.data
    # 使用 n_neighbors=500 训练 LOF（局部离群因子）模型
    clf = neighbors.LocalOutlierFactor(n_neighbors=500).fit(X)
    # 断言模型中的 n_neighbors_ 属性值
    assert clf.n_neighbors_ == X.shape[0] - 1

    # 测试 n_neighbors 超出样本数时的警告信息
    clf = neighbors.LocalOutlierFactor(n_neighbors=500)
    msg = "n_neighbors will be set to (n_samples - 1)"
    with pytest.warns(UserWarning, match=re.escape(msg)):
        clf.fit(X)
    # 再次断言模型中的 n_neighbors_ 属性值
    assert clf.n_neighbors_ == X.shape[0] - 1


# 测试 score_samples 方法
def test_score_samples(global_dtype):
    X_train = np.asarray([[1, 1], [1, 2], [2, 1]], dtype=global_dtype)
    X_test = np.asarray([[2.0, 2.0]], dtype=global_dtype)
    # 使用 LOF（局部离群因子）模型训练 X_train
    clf1 = neighbors.LocalOutlierFactor(
        n_neighbors=2, contamination=0.1, novelty=True
    ).fit(X_train)
    clf2 = neighbors.LocalOutlierFactor(n_neighbors=2, novelty=True).fit(X_train)

    # 计算 X_test 样本的 LOF 分数
    clf1_scores = clf1.score_samples(X_test)
    # 计算 X_test 样本的 LOF 决策函数值
    clf1_decisions = clf1.decision_function(X_test)

    # 计算 X_test 样本的 LOF 分数（另一种方式）
    clf2_scores = clf2.score_samples(X_test)
    # 计算 X_test 样本的 LOF 决策函数值（另一种方式）
    clf2_decisions = clf2.decision_function(X_test)

    # 断言两种计算 LOF 分数的方式得到的结果相等
    assert_allclose(
        clf1_scores,
        clf1_decisions + clf1.offset_,
    )
    assert_allclose(
        clf2_scores,
        clf2_decisions + clf2.offset_,
    )
    # 进一步断言两种 LOF 计算方式的分数结果相等
    assert_allclose(clf1_scores, clf2_scores)


# 测试 novelty=True 时的异常处理
def test_novelty_errors():
    X = iris.data

    # 检查 novelty=False 时的异常情况
    clf = neighbors.LocalOutlierFactor()
    clf.fit(X)
    # 预测、决策函数和 score_samples 在 novelty=False 时会引发 ValueError
    for method in ["predict", "decision_function", "score_samples"]:
        outer_msg = f"'LocalOutlierFactor' has no attribute '{method}'"
        inner_msg = "{} is not available when novelty=False".format(method)
        with pytest.raises(AttributeError, match=outer_msg) as exec_info:
            getattr(clf, method)

        assert isinstance(exec_info.value.__cause__, AttributeError)
        assert inner_msg in str(exec_info.value.__cause__)

    # 检查 novelty=True 时的异常情况
    clf = neighbors.LocalOutlierFactor(novelty=True)
    # 设置外部错误消息，指示'LocalOutlierFactor'类没有'fit_predict'属性
    outer_msg = "'LocalOutlierFactor' has no attribute 'fit_predict'"
    # 设置内部错误消息，指示当novelty=True时，'fit_predict'不可用
    inner_msg = "fit_predict is not available when novelty=True"
    # 使用 pytest 模块中的 raises 函数，期望捕获 AttributeError 异常，并匹配外部错误消息
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        # 使用 getattr 函数尝试获取 clf 对象中的 'fit_predict' 属性
        getattr(clf, "fit_predict")
    
    # 断言捕获的异常 exec_info 的原因（cause）是 AttributeError 类型
    assert isinstance(exec_info.value.__cause__, AttributeError)
    # 断言捕获的异常 exec_info 的原因（cause）包含预设的内部错误消息
    assert inner_msg in str(exec_info.value.__cause__)
# 检查在 novelty=True 时，训练样本的异常分数是否通过 negative_outlier_factor_ 属性仍然可访问
def test_novelty_training_scores(global_dtype):
    # 将 iris 数据转换为指定的全局数据类型
    X = iris.data.astype(global_dtype)

    # 使用 novelty=False 进行拟合
    clf_1 = neighbors.LocalOutlierFactor()
    clf_1.fit(X)
    scores_1 = clf_1.negative_outlier_factor_

    # 使用 novelty=True 进行拟合
    clf_2 = neighbors.LocalOutlierFactor(novelty=True)
    clf_2.fit(X)
    scores_2 = clf_2.negative_outlier_factor_

    # 断言两种模式下的异常分数是否近似相等
    assert_allclose(scores_1, scores_2)


# 检查预测方法的可用性取决于 novelty 值
def test_hasattr_prediction():
    X = [[1, 1], [1, 2], [2, 1]]

    # 当 novelty=True 时
    clf = neighbors.LocalOutlierFactor(novelty=True)
    clf.fit(X)
    assert hasattr(clf, "predict")
    assert hasattr(clf, "decision_function")
    assert hasattr(clf, "score_samples")
    assert not hasattr(clf, "fit_predict")

    # 当 novelty=False 时
    clf = neighbors.LocalOutlierFactor(novelty=False)
    clf.fit(X)
    assert hasattr(clf, "fit_predict")
    assert not hasattr(clf, "predict")
    assert not hasattr(clf, "decision_function")
    assert not hasattr(clf, "score_samples")


# 使用 parametrize_with_checks 对 novelty=True 的 estimator 运行通用测试
def test_novelty_true_common_tests(estimator, check):
    # 默认 LOF（novelty=False）的通用测试也会运行。
    # 这里我们为 novelty=True 的 LOF 运行这些通用测试
    check(estimator)


# 检查预测的异常值数量是否与期望的异常值数量相等
@pytest.mark.parametrize("expected_outliers", [30, 53])
def test_predicted_outlier_number(expected_outliers):
    # 预测的异常值数量应当等于期望的异常值数量，除非异常性分数存在并列情况。
    X = iris.data
    n_samples = X.shape[0]
    contamination = float(expected_outliers) / n_samples

    clf = neighbors.LocalOutlierFactor(contamination=contamination)
    y_pred = clf.fit_predict(X)

    num_outliers = np.sum(y_pred != 1)
    if num_outliers != expected_outliers:
        y_dec = clf.negative_outlier_factor_
        check_outlier_corruption(num_outliers, expected_outliers, y_dec)


# 测试 LocalOutlierFactor 是否支持 CSR 输入
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse(csr_container):
    # LocalOutlierFactor 必须支持 CSR 输入
    # TODO: 比较在密集和稀疏数据上的结果，参见：
    # https://github.com/scikit-learn/scikit-learn/pull/23585#discussion_r968388186
    X = csr_container(iris.data)

    lof = neighbors.LocalOutlierFactor(novelty=True)
    lof.fit(X)
    lof.predict(X)
    lof.score_samples(X)
    lof.decision_function(X)

    lof = neighbors.LocalOutlierFactor(novelty=False)
    lof.fit_predict(X)


# 检查当 n_neighbors == n_samples 时是否能正确抛出错误消息
def test_lof_error_n_neighbors_too_large():
    """Check that we raise a proper error message when n_neighbors == n_samples.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/17207
    """
    X = np.ones((7, 7))
    # 定义错误消息，用于断言异常抛出时的匹配条件
    msg = (
        "Expected n_neighbors < n_samples_fit, but n_neighbors = 1, "
        "n_samples_fit = 1, n_samples = 1"
    )
    # 使用 pytest 框架验证异常抛出，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        # 使用 LocalOutlierFactor 模型拟合只含有一个样本的数据集
        lof = neighbors.LocalOutlierFactor(n_neighbors=1).fit(X[:1])

    # 使用 LocalOutlierFactor 模型拟合只含有两个样本的数据集
    lof = neighbors.LocalOutlierFactor(n_neighbors=2).fit(X[:2])
    # 断言模型中的 n_samples_fit_ 属性等于拟合数据集的样本数
    assert lof.n_samples_fit_ == 2

    # 定义错误消息，用于断言异常抛出时的匹配条件
    msg = (
        "Expected n_neighbors < n_samples_fit, but n_neighbors = 2, "
        "n_samples_fit = 2, n_samples = 2"
    )
    # 使用 pytest 框架验证异常抛出，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        # 使用 LocalOutlierFactor 模型计算无监督情况下的 k 近邻
        lof.kneighbors(None, n_neighbors=2)

    # 计算数据集中每个样本到其最近邻的距离和索引
    distances, indices = lof.kneighbors(None, n_neighbors=1)
    # 断言返回的距离数组的形状符合预期
    assert distances.shape == (2, 1)
    # 断言返回的索引数组的形状符合预期
    assert indices.shape == (2, 1)

    # 定义错误消息，用于断言异常抛出时的匹配条件
    msg = (
        "Expected n_neighbors <= n_samples_fit, but n_neighbors = 3, "
        "n_samples_fit = 2, n_samples = 7"
    )
    # 使用 pytest 框架验证异常抛出，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        # 使用 LocalOutlierFactor 模型计算指定 k 近邻数时的异常情况
        lof.kneighbors(X, n_neighbors=3)

    # 计算数据集中每个样本到其最近邻的距离和索引
    (
        distances,
        indices,
    ) = lof.kneighbors(X, n_neighbors=2)
    # 断言返回的距离数组的形状符合预期
    assert distances.shape == (7, 2)
    # 断言返回的索引数组的形状符合预期
    assert indices.shape == (7, 2)
# 使用 pytest 的参数化装饰器，为 test_lof_input_dtype_preservation 函数提供多组参数
@pytest.mark.parametrize("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
@pytest.mark.parametrize("novelty", [True, False])
@pytest.mark.parametrize("contamination", [0.5, "auto"])
def test_lof_input_dtype_preservation(global_dtype, algorithm, contamination, novelty):
    """Check that the fitted attributes are stored using the data type of X."""
    # 从 iris 数据中获取特征数据 X，并按照全局数据类型 global_dtype 进行类型转换，不复制数据
    X = iris.data.astype(global_dtype, copy=False)

    # 创建 LocalOutlierFactor 模型对象 iso，设置参数包括邻居数量、算法、污染度和是否新颖
    iso = neighbors.LocalOutlierFactor(
        n_neighbors=5, algorithm=algorithm, contamination=contamination, novelty=novelty
    )
    # 使用 X 对象拟合 iso 模型
    iso.fit(X)

    # 断言 iso 模型的 negative_outlier_factor_ 属性的数据类型与 global_dtype 一致
    assert iso.negative_outlier_factor_.dtype == global_dtype

    # 对于方法 ("score_samples", "decision_function")，若 iso 对象包含该方法，则进行断言其返回的数据类型与 global_dtype 一致
    for method in ("score_samples", "decision_function"):
        if hasattr(iso, method):
            y_pred = getattr(iso, method)(X)
            assert y_pred.dtype == global_dtype


# 使用 pytest 的参数化装饰器，为 test_lof_dtype_equivalence 函数提供多组参数
@pytest.mark.parametrize("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
@pytest.mark.parametrize("novelty", [True, False])
@pytest.mark.parametrize("contamination", [0.5, "auto"])
def test_lof_dtype_equivalence(algorithm, novelty, contamination):
    """Check the equivalence of the results with 32 and 64 bits input."""

    # 创建正常样本和异常样本数据
    inliers = iris.data[:50]  # setosa 鸢尾花数据，与其他鸢尾花明显不同
    outliers = iris.data[-5:]  # virginica 将被视为异常值
    # 将输入数据 X 的精度降低到 32 位，以检查在 32 位和 64 位计算时的等效性
    X = np.concatenate([inliers, outliers], axis=0).astype(np.float32)

    # 创建 32 位精度的 lof_32 模型对象
    lof_32 = neighbors.LocalOutlierFactor(
        algorithm=algorithm, novelty=novelty, contamination=contamination
    )
    X_32 = X.astype(np.float32, copy=True)
    lof_32.fit(X_32)

    # 创建 64 位精度的 lof_64 模型对象
    lof_64 = neighbors.LocalOutlierFactor(
        algorithm=algorithm, novelty=novelty, contamination=contamination
    )
    X_64 = X.astype(np.float64, copy=True)
    lof_64.fit(X_64)

    # 断言 lof_32 和 lof_64 的 negative_outlier_factor_ 属性值非常接近
    assert_allclose(lof_32.negative_outlier_factor_, lof_64.negative_outlier_factor_)

    # 对于方法 ("score_samples", "decision_function", "predict", "fit_predict")，若 lof_32 对象包含该方法，则进行断言 lof_32 和 lof_64 返回的结果非常接近
    for method in ("score_samples", "decision_function", "predict", "fit_predict"):
        if hasattr(lof_32, method):
            y_pred_32 = getattr(lof_32, method)(X_32)
            y_pred_64 = getattr(lof_64, method)(X_64)
            assert_allclose(y_pred_32, y_pred_64, atol=0.0002)


# 无参数化装饰器，测试 LocalOutlierFactor 在训练数据中存在重复值时是否引发警告
def test_lof_duplicate_samples():
    """
    Check that LocalOutlierFactor raises a warning when duplicate values
    in the training data cause inaccurate results.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27839
    """

    # 创建随机数生成器 rng
    rng = np.random.default_rng(0)

    # 创建 x 数组，包含重复值和明显的异常值
    x = rng.permutation(
        np.hstack(
            [
                [0.1] * 1000,  # 常量数值
                np.linspace(0.1, 0.3, num=3000),
                rng.random(500) * 100,  # 明显的异常值
            ]
        )
    )
    # 将 x 转换为二维数组 X
    X = x.reshape(-1, 1)

    # 定义错误消息内容，指示重复值可能导致不准确的结果
    error_msg = (
        "Duplicate values are leading to incorrect results. "
        "Increase the number of neighbors for more accurate results."
    )
    # 创建一个局部异常因子检测器对象，设置参数：邻居数为5，异常值比例为0.1
    lof = neighbors.LocalOutlierFactor(n_neighbors=5, contamination=0.1)

    # 使用 pytest 来捕获特定的警告信息
    # 当代码块执行时，期望捕获一个 UserWarning 类型的警告，并且警告消息与 error_msg 变量内容匹配
    with pytest.warns(UserWarning, match=re.escape(error_msg)):
        # 对数据集 X 进行拟合并预测异常值
        lof.fit_predict(X)
```