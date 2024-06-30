# `D:\src\scipysrc\scikit-learn\sklearn\metrics\tests\test_score_objects.py`

```
# 导入必要的模块
import numbers  # 用于检查是否为数字类型
import pickle  # 用于序列化和反序列化对象
from copy import deepcopy  # 用于深拷贝对象
from functools import partial  # 用于创建偏函数
from unittest.mock import Mock  # 用于模拟对象

import joblib  # 用于序列化和反序列化对象
import numpy as np  # NumPy 数学库
import pytest  # 测试框架
from numpy.testing import assert_allclose  # 用于检查 NumPy 数组是否近似相等

from sklearn import config_context  # 用于配置上下文
from sklearn.base import BaseEstimator  # 基础估计器类
from sklearn.cluster import KMeans  # KMeans 聚类算法
from sklearn.datasets import (  # 数据集生成器
    load_diabetes,
    make_blobs,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge  # 线性模型
from sklearn.metrics import (  # 模型评估指标
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    check_scoring,
    f1_score,
    fbeta_score,
    get_scorer,
    get_scorer_names,
    jaccard_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.metrics import cluster as cluster_module  # 聚类相关指标
from sklearn.metrics._scorer import (  # 评估器相关类
    _check_multimetric_scoring,
    _MultimetricScorer,
    _PassthroughScorer,
    _Scorer,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split  # 模型选择和交叉验证
from sklearn.multiclass import OneVsRestClassifier  # 多类别分类器
from sklearn.neighbors import KNeighborsClassifier  # K 近邻分类器
from sklearn.pipeline import make_pipeline  # 创建管道
from sklearn.svm import LinearSVC  # 线性支持向量机
from sklearn.tests.metadata_routing_common import (  # 测试相关模块
    assert_request_is_empty,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 决策树分类器和回归器
from sklearn.utils._testing import (  # 测试工具
    assert_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping  # 元数据路由器和方法映射

REGRESSION_SCORERS = [  # 回归模型评估指标列表
    "d2_absolute_error_score",
    "explained_variance",
    "r2",
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "neg_mean_absolute_percentage_error",
    "neg_mean_squared_log_error",
    "neg_median_absolute_error",
    "neg_root_mean_squared_error",
    "neg_root_mean_squared_log_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "median_absolute_error",
    "max_error",
    "neg_mean_poisson_deviance",
    "neg_mean_gamma_deviance",
]

CLF_SCORERS = [  # 分类模型评估指标列表
    "accuracy",
    "balanced_accuracy",
    "top_k_accuracy",
    "f1",
    "f1_weighted",
    "f1_macro",
    "f1_micro",
    "roc_auc",
    "average_precision",
    "precision",
    "precision_weighted",
    "precision_macro",
    "precision_micro",
    "recall",
    "recall_weighted",
    "recall_macro",
    "recall_micro",
    "neg_log_loss",
    "neg_brier_score",
    "jaccard",
    "jaccard_weighted",
    "jaccard_macro",
    "jaccard_micro",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_weighted",
    "matthews_corrcoef",
    "positive_likelihood_ratio",
    "neg_negative_likelihood_ratio",
]

# 所有监督学习聚类评估指标（它们的行为类似于分类指标）
CLUSTER_SCORERS = [
    "adjusted_rand_score",  # 聚类评分器使用的评分指标：调整兰德指数
    "rand_score",           # 聚类评分器使用的评分指标：兰德指数
    "homogeneity_score",    # 聚类评分器使用的评分指标：同质性
    "completeness_score",   # 聚类评分器使用的评分指标：完整性
    "v_measure_score",      # 聚类评分器使用的评分指标：V-measure
    "mutual_info_score",    # 聚类评分器使用的评分指标：互信息
    "adjusted_mutual_info_score",  # 聚类评分器使用的评分指标：调整互信息
    "normalized_mutual_info_score",  # 聚类评分器使用的评分指标：归一化互信息
    "fowlkes_mallows_score",  # 聚类评分器使用的评分指标：福尔克斯-马洛指数
]

MULTILABEL_ONLY_SCORERS = [
    "precision_samples",    # 多标签评分器使用的评分指标：样本精度
    "recall_samples",       # 多标签评分器使用的评分指标：样本召回率
    "f1_samples",           # 多标签评分器使用的评分指标：样本F1分数
    "jaccard_samples",      # 多标签评分器使用的评分指标：Jaccard相似系数
]

REQUIRE_POSITIVE_Y_SCORERS = [
    "neg_mean_poisson_deviance",   # 需要正值目标 y 的评分器：负泊松偏差均值
    "neg_mean_gamma_deviance"      # 需要正值目标 y 的评分器：负伽马偏差均值
]


def _require_positive_y(y):
    """确保目标 y 严格为正值"""
    offset = abs(y.min()) + 1
    y = y + offset
    return y


def _make_estimators(X_train, y_train, y_ml_train):
    # 创建一些合理的估算器，以便测试各种评分方法
    sensible_regr = DecisionTreeRegressor(random_state=0)
    # 一些回归评分器要求严格正值的输入。
    sensible_regr.fit(X_train, _require_positive_y(y_train))
    sensible_clf = DecisionTreeClassifier(random_state=0)
    sensible_clf.fit(X_train, y_train)
    sensible_ml_clf = DecisionTreeClassifier(random_state=0)
    sensible_ml_clf.fit(X_train, y_ml_train)
    return dict(
        [(name, sensible_regr) for name in REGRESSION_SCORERS]
        + [(name, sensible_clf) for name in CLF_SCORERS]
        + [(name, sensible_clf) for name in CLUSTER_SCORERS]
        + [(name, sensible_ml_clf) for name in MULTILABEL_ONLY_SCORERS]
    )


@pytest.fixture(scope="module")
def memmap_data_and_estimators(tmp_path_factory):
    temp_folder = tmp_path_factory.mktemp("sklearn_test_score_objects")
    X, y = make_classification(n_samples=30, n_features=5, random_state=0)
    _, y_ml = make_multilabel_classification(n_samples=X.shape[0], random_state=0)
    filename = temp_folder / "test_data.pkl"
    joblib.dump((X, y, y_ml), filename)
    X_mm, y_mm, y_ml_mm = joblib.load(filename, mmap_mode="r")
    estimators = _make_estimators(X_mm, y_mm, y_ml_mm)

    yield X_mm, y_mm, y_ml_mm, estimators


class EstimatorWithFit(BaseEstimator):
    """用于测试评分验证器的虚拟估算器"""

    def fit(self, X, y):
        return self


class EstimatorWithFitAndScore:
    """用于测试评分验证器的虚拟估算器"""

    def fit(self, X, y):
        return self

    def score(self, X, y):
        return 1.0


class EstimatorWithFitAndPredict:
    """用于测试评分验证器的虚拟估算器"""

    def fit(self, X, y):
        self.y = y
        return self

    def predict(self, X):
        return self.y


class DummyScorer:
    """总是返回1的虚拟评分器"""

    def __call__(self, est, X, y):
        return 1


def test_all_scorers_repr():
    # 测试所有评分器的工作 repr
    for name in get_scorer_names():
        repr(get_scorer(name))


def check_scoring_validator_for_single_metric_usecases(scoring_validator):
    # 测试所有单指标用例的分支
    estimator = EstimatorWithFitAndScore()
    estimator.fit([[1]], [1])
    scorer = scoring_validator(estimator)
    # 断言确保 scorer 是 _PassthroughScorer 的实例
    assert isinstance(scorer, _PassthroughScorer)
    # 使用 assert_almost_equal 函数检查 scorer 对象的预测结果与实际结果的近似性
    assert_almost_equal(scorer(estimator, [[1]], [1]), 1.0)
    
    # 创建 EstimatorWithFitAndPredict 的实例 estimator，并对其进行训练
    estimator = EstimatorWithFitAndPredict()
    estimator.fit([[1]], [1])
    # 定义匹配模式用于检查 TypeError 异常信息
    pattern = (
        r"If no scoring is specified, the estimator passed should have"
        r" a 'score' method\. The estimator .* does not\."
    )
    # 使用 pytest.raises 检查是否抛出了预期的 TypeError 异常，且异常信息符合 pattern
    with pytest.raises(TypeError, match=pattern):
        scoring_validator(estimator)
    
    # 使用 scoring_validator 函数获取评分器 scorer，评分方式为 "accuracy"
    scorer = scoring_validator(estimator, scoring="accuracy")
    # 使用 assert_almost_equal 函数检查 scorer 对象的预测结果与实际结果的近似性
    assert_almost_equal(scorer(estimator, [[1]], [1]), 1.0)
    
    # 创建 EstimatorWithFit 的实例 estimator
    estimator = EstimatorWithFit()
    # 使用 scoring_validator 函数获取评分器 scorer，评分方式为 "accuracy"
    scorer = scoring_validator(estimator, scoring="accuracy")
    # 断言确保 scorer 是 _Scorer 的实例
    assert isinstance(scorer, _Scorer)
    # 断言确保 scorer 的响应方法是 "predict"
    assert scorer._response_method == "predict"
    
    # 检查 allow_none 参数在 check_scoring 函数中的行为
    if scoring_validator is check_scoring:
        # 创建 EstimatorWithFit 的实例 estimator
        estimator = EstimatorWithFit()
        # 使用 scoring_validator 函数获取评分器 scorer，同时允许其为 None
        scorer = scoring_validator(estimator, allow_none=True)
        # 断言 scorer 是 None
        assert scorer is None
# 使用 pytest.mark.parametrize 装饰器定义参数化测试，用于测试不同的评分方式
@pytest.mark.parametrize(
    "scoring",
    (
        ("accuracy",),  # 单个元素元组，评分方式为 accuracy
        ["precision"],  # 单个元素列表，评分方式为 precision
        {"acc": "accuracy", "precision": "precision"},  # 字典，映射评分名称到评分方式
        ("accuracy", "precision"),  # 多个元素元组，包含 accuracy 和 precision 评分方式
        ["precision", "accuracy"],  # 多个元素列表，包含 precision 和 accuracy 评分方式
        {
            "accuracy": make_scorer(accuracy_score),  # 字典，accuracy 评分对应的评分器
            "precision": make_scorer(precision_score),  # precision 评分对应的评分器
        },
    ),
    ids=[
        "single_tuple",  # 每种评分方式的标识符
        "single_list",
        "dict_str",
        "multi_tuple",
        "multi_list",
        "dict_callable",
    ],
)
def test_check_scoring_and_check_multimetric_scoring(scoring):
    # 调用函数检查单个评分用例的评分验证器
    check_scoring_validator_for_single_metric_usecases(check_scoring)
    # 确保 check_scoring 正确应用于构成的评分器

    # 创建一个 LinearSVC 估计器实例
    estimator = LinearSVC(random_state=0)
    # 使用示例数据拟合估计器
    estimator.fit([[1], [2], [3]], [1, 1, 0])

    # 调用函数检查多指标评分
    scorers = _check_multimetric_scoring(estimator, scoring)
    # 断言 scorers 是一个字典
    assert isinstance(scorers, dict)
    # 断言 scorers 的键按照 scoring 中的顺序排列
    assert sorted(scorers.keys()) == sorted(list(scoring))
    # 断言 scorers 中的值都是 _Scorer 类型的实例
    assert all([isinstance(scorer, _Scorer) for scorer in list(scorers.values())])
    # 断言 scorers 中每个评分器的响应方法都是 "predict"
    assert all(scorer._response_method == "predict" for scorer in scorers.values())

    # 如果 scoring 中包含 "acc" 评分方式，执行以下断言
    if "acc" in scoring:
        # 断言 acc 评分器在给定数据上的评分接近 2/3
        assert_almost_equal(
            scorers["acc"](estimator, [[1], [2], [3]], [1, 0, 0]), 2.0 / 3.0
        )
    # 如果 scoring 中包含 "accuracy" 评分方式，执行以下断言
    if "accuracy" in scoring:
        # 断言 accuracy 评分器在给定数据上的评分接近 2/3
        assert_almost_equal(
            scorers["accuracy"](estimator, [[1], [2], [3]], [1, 0, 0]), 2.0 / 3.0
        )
    # 如果 scoring 中包含 "precision" 评分方式，执行以下断言
    if "precision" in scoring:
        # 断言 precision 评分器在给定数据上的评分接近 0.5
        assert_almost_equal(
            scorers["precision"](estimator, [[1], [2], [3]], [1, 0, 0]), 0.5
        )


# 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试不同的评分方式错误情况
@pytest.mark.parametrize(
    "scoring, msg",
    [
        (
            (make_scorer(precision_score), make_scorer(accuracy_score)),
            "One or more of the elements were callables",
        ),
        ([5], "Non-string types were found"),
        ((make_scorer(precision_score),), "One or more of the elements were callables"),
        ((), "Empty list was given"),
        (("f1", "f1"), "Duplicate elements were found"),
        ({4: "accuracy"}, "Non-string types were found in the keys"),
        ({}, "An empty dict was passed"),
    ],
    ids=[
        "tuple of callables",  # 每种错误情况的标识符
        "list of int",
        "tuple of one callable",
        "empty tuple",
        "non-unique str",
        "non-string key dict",
        "empty dict",
    ],
)
def test_check_scoring_and_check_multimetric_scoring_errors(scoring, msg):
    # 确保在评分参数无效时引发错误
    # 在 test_validation.py 中测试更多奇怪的边界情况
    estimator = EstimatorWithFitAndPredict()
    estimator.fit([[1]], [1])

    # 使用 pytest 的 raises 断言，确保调用 _check_multimetric_scoring 函数时会引发 ValueError，并匹配指定的错误消息
    with pytest.raises(ValueError, match=msg):
        _check_multimetric_scoring(estimator, scoring=scoring)


# 测试 check_scoring 在 GridSearchCV 和 pipeline 上的工作情况
def test_check_scoring_gridsearchcv():
    # 确保 check_scoring 在 GridSearchCV 和 pipeline 上正常工作
    # 稍微冗余的非回归测试

    # 创建一个 GridSearchCV 实例，使用 LinearSVC 作为估计器，C 参数为 [0.1, 1]，交叉验证次数为 3
    grid = GridSearchCV(LinearSVC(), param_grid={"C": [0.1, 1]}, cv=3)
    # 使用默认的评分指标 'f1' 检查网格参数 grid，并返回评分器对象
    scorer = check_scoring(grid, scoring="f1")
    # 断言评分器确实是 _Scorer 类的实例
    assert isinstance(scorer, _Scorer)
    # 断言评分器使用的响应方法是 "predict"
    assert scorer._response_method == "predict"

    # 创建包含 LinearSVC 的管道对象
    pipe = make_pipeline(LinearSVC())
    # 使用默认的评分指标 'f1' 检查管道对象 pipe，并返回评分器对象
    scorer = check_scoring(pipe, scoring="f1")
    # 断言评分器确实是 _Scorer 类的实例
    assert isinstance(scorer, _Scorer)
    # 断言评分器使用的响应方法是 "predict"
    assert scorer._response_method == "predict"

    # 检查 cross_val_score 是否确实调用了评分器，并且除了拟合之外不对估计器做任何假设。
    # 使用 DummyScorer 作为评分指标，对 EstimatorWithFit 拟合数据 [[1], [2], [3]] 和目标 [1, 0, 1]，进行交叉验证，cv=3
    scores = cross_val_score(
        EstimatorWithFit(), [[1], [2], [3]], [1, 0, 1], scoring=DummyScorer(), cv=3
    )
    # 断言 scores 数组中的值全部为 1
    assert_array_equal(scores, 1)
@pytest.mark.parametrize(
    "scorer_name, metric",
    [
        ("f1", f1_score),  # 参数化测试用例：使用f1_score作为度量标准
        ("f1_weighted", partial(f1_score, average="weighted")),  # 使用加权平均的f1_score作为度量标准
        ("f1_macro", partial(f1_score, average="macro")),  # 使用宏平均的f1_score作为度量标准
        ("f1_micro", partial(f1_score, average="micro")),  # 使用微平均的f1_score作为度量标准
        ("precision", precision_score),  # 使用precision_score作为度量标准
        ("precision_weighted", partial(precision_score, average="weighted")),  # 使用加权平均的precision_score作为度量标准
        ("precision_macro", partial(precision_score, average="macro")),  # 使用宏平均的precision_score作为度量标准
        ("precision_micro", partial(precision_score, average="micro")),  # 使用微平均的precision_score作为度量标准
        ("recall", recall_score),  # 使用recall_score作为度量标准
        ("recall_weighted", partial(recall_score, average="weighted")),  # 使用加权平均的recall_score作为度量标准
        ("recall_macro", partial(recall_score, average="macro")),  # 使用宏平均的recall_score作为度量标准
        ("recall_micro", partial(recall_score, average="micro")),  # 使用微平均的recall_score作为度量标准
        ("jaccard", jaccard_score),  # 使用jaccard_score作为度量标准
        ("jaccard_weighted", partial(jaccard_score, average="weighted")),  # 使用加权平均的jaccard_score作为度量标准
        ("jaccard_macro", partial(jaccard_score, average="macro")),  # 使用宏平均的jaccard_score作为度量标准
        ("jaccard_micro", partial(jaccard_score, average="micro")),  # 使用微平均的jaccard_score作为度量标准
        ("top_k_accuracy", top_k_accuracy_score),  # 使用top_k_accuracy_score作为度量标准
        ("matthews_corrcoef", matthews_corrcoef),  # 使用matthews_corrcoef作为度量标准
    ],
)
def test_classification_binary_scores(scorer_name, metric):
    # 检查对于支持二元分类的分数和评分器之间的一致性。
    X, y = make_blobs(random_state=0, centers=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LinearSVC(random_state=0)
    clf.fit(X_train, y_train)

    # 计算测试集上的评分
    score = get_scorer(scorer_name)(clf, X_test, y_test)
    # 计算预期的评分
    expected_score = metric(y_test, clf.predict(X_test))
    # 断言评分与预期评分几乎相等
    assert_almost_equal(score, expected_score)


@pytest.mark.parametrize(
    "scorer_name, metric",
    [
        ("accuracy", accuracy_score),  # 参数化测试用例：使用accuracy_score作为度量标准
        ("balanced_accuracy", balanced_accuracy_score),  # 使用balanced_accuracy_score作为度量标准
        ("f1_weighted", partial(f1_score, average="weighted")),  # 使用加权平均的f1_score作为度量标准
        ("f1_macro", partial(f1_score, average="macro")),  # 使用宏平均的f1_score作为度量标准
        ("f1_micro", partial(f1_score, average="micro")),  # 使用微平均的f1_score作为度量标准
        ("precision_weighted", partial(precision_score, average="weighted")),  # 使用加权平均的precision_score作为度量标准
        ("precision_macro", partial(precision_score, average="macro")),  # 使用宏平均的precision_score作为度量标准
        ("precision_micro", partial(precision_score, average="micro")),  # 使用微平均的precision_score作为度量标准
        ("recall_weighted", partial(recall_score, average="weighted")),  # 使用加权平均的recall_score作为度量标准
        ("recall_macro", partial(recall_score, average="macro")),  # 使用宏平均的recall_score作为度量标准
        ("recall_micro", partial(recall_score, average="micro")),  # 使用微平均的recall_score作为度量标准
        ("jaccard_weighted", partial(jaccard_score, average="weighted")),  # 使用加权平均的jaccard_score作为度量标准
        ("jaccard_macro", partial(jaccard_score, average="macro")),  # 使用宏平均的jaccard_score作为度量标准
        ("jaccard_micro", partial(jaccard_score, average="micro")),  # 使用微平均的jaccard_score作为度量标准
    ],
)
def test_classification_multiclass_scores(scorer_name, metric):
    # 检查对于支持多类分类的分数和评分器之间的一致性。
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=30, random_state=0
    )

    # 使用`stratify` = y确保训练集和测试集包含所有类别
    # 使用 train_test_split 函数划分数据集为训练集和测试集，保持随机种子为0，使用类别平衡抽样策略
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, stratify=y
    )
    
    # 创建一个决策树分类器对象，设定随机种子为0
    clf = DecisionTreeClassifier(random_state=0)
    
    # 使用训练集 X_train 和标签 y_train 对分类器进行训练
    clf.fit(X_train, y_train)
    
    # 调用指定评分器名称的评分函数，评估分类器在测试集 X_test 和 y_test 上的性能得分
    score = get_scorer(scorer_name)(clf, X_test, y_test)
    
    # 计算分类器在测试集上的预期得分，通过指定的度量标准 metric 对真实标签 y_test 和预测标签 clf.predict(X_test) 进行评估
    expected_score = metric(y_test, clf.predict(X_test))
    
    # 使用 pytest 的 approx 函数检查计算得分和预期得分是否非常接近，断言它们相等
    assert score == pytest.approx(expected_score)
def test_custom_scorer_pickling():
    # 测试自定义评分器是否可以被序列化
    X, y = make_blobs(random_state=0, centers=2)
    # 使用 make_blobs 生成随机数据集 X, y，centers=2 表示生成两个中心的数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 将数据集分割为训练集和测试集
    clf = LinearSVC(random_state=0)
    # 初始化线性支持向量分类器
    clf.fit(X_train, y_train)
    # 在训练集上拟合分类器

    scorer = make_scorer(fbeta_score, beta=2)
    # 创建一个评分器，使用 F-beta 分数，beta=2
    score1 = scorer(clf, X_test, y_test)
    # 使用评分器计算分类器在测试集上的分数
    unpickled_scorer = pickle.loads(pickle.dumps(scorer))
    # 序列化和反序列化评分器对象
    score2 = unpickled_scorer(clf, X_test, y_test)
    # 使用反序列化后的评分器对象计算分类器在测试集上的分数
    assert score1 == pytest.approx(score2)
    # 断言序列化前后的分数近似相等

    # smoke test the repr:
    repr(fbeta_score)


def test_regression_scorers():
    # 测试回归评分器
    diabetes = load_diabetes()
    # 载入糖尿病数据集
    X, y = diabetes.data, diabetes.target
    # 获取数据和目标变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 将数据集分割为训练集和测试集
    clf = Ridge()
    # 初始化岭回归模型
    clf.fit(X_train, y_train)
    # 在训练集上拟合回归模型
    score1 = get_scorer("r2")(clf, X_test, y_test)
    # 使用 r2 评分器计算回归模型在测试集上的分数
    score2 = r2_score(y_test, clf.predict(X_test))
    # 计算回归模型在测试集上的 R^2 分数
    assert_almost_equal(score1, score2)
    # 断言两种方法计算得到的分数近似相等


def test_thresholded_scorers():
    # 测试需要阈值的评分器
    X, y = make_blobs(random_state=0, centers=2)
    # 使用 make_blobs 生成随机数据集 X, y，centers=2 表示生成两个中心的数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 将数据集分割为训练集和测试集
    clf = LogisticRegression(random_state=0)
    # 初始化逻辑回归模型
    clf.fit(X_train, y_train)
    # 在训练集上拟合逻辑回归模型
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    # 使用 roc_auc 评分器计算逻辑回归模型在测试集上的 ROC AUC 分数
    score2 = roc_auc_score(y_test, clf.decision_function(X_test))
    # 计算逻辑回归模型在测试集上的 ROC AUC 分数（使用 decision_function）
    score3 = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    # 计算逻辑回归模型在测试集上的 ROC AUC 分数（使用 predict_proba 的第二列）
    assert_almost_equal(score1, score2)
    # 断言两种方法计算得到的 ROC AUC 分数近似相等
    assert_almost_equal(score1, score3)
    # 断言两种方法计算得到的 ROC AUC 分数近似相等

    logscore = get_scorer("neg_log_loss")(clf, X_test, y_test)
    # 使用 neg_log_loss 评分器计算逻辑回归模型在测试集上的负对数损失
    logloss = log_loss(y_test, clf.predict_proba(X_test))
    # 计算逻辑回归模型在测试集上的对数损失
    assert_almost_equal(-logscore, logloss)
    # 断言负对数损失与对数损失的负值近似相等

    # same for an estimator without decision_function
    clf = DecisionTreeClassifier()
    # 初始化决策树分类器
    clf.fit(X_train, y_train)
    # 在训练集上拟合决策树分类器
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    # 使用 roc_auc 评分器计算决策树分类器在测试集上的 ROC AUC 分数
    score2 = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    # 计算决策树分类器在测试集上的 ROC AUC 分数（使用 predict_proba 的第二列）
    assert_almost_equal(score1, score2)
    # 断言两种方法计算得到的 ROC AUC 分数近似相等

    # test with a regressor (no decision_function)
    reg = DecisionTreeRegressor()
    # 初始化决策树回归器
    reg.fit(X_train, y_train)
    # 在训练集上拟合决策树回归器
    err_msg = "DecisionTreeRegressor has none of the following attributes"
    # 错误消息字符串
    with pytest.raises(AttributeError, match=err_msg):
        get_scorer("roc_auc")(reg, X_test, y_test)
    # 断言使用决策树回归器时会引发 AttributeError 异常，匹配错误消息字符串

    # Test that an exception is raised on more than two classes
    X, y = make_blobs(random_state=0, centers=3)
    # 使用 make_blobs 生成随机数据集 X, y，centers=3 表示生成三个中心的数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 将数据集分割为训练集和测试集
    clf.fit(X_train, y_train)
    # 在训练集上拟合分类器
    with pytest.raises(ValueError, match="multi_class must be in \\('ovo', 'ovr'\\)"):
        get_scorer("roc_auc")(clf, X_test, y_test)
    # 断言使用 ROC AUC 评分器时会引发 ValueError 异常，匹配错误消息字符串

    # test error is raised with a single class present in model
    # (predict_proba shape is not suitable for binary auc)
    X, y = make_blobs(random_state=0, centers=2)
    # 使用 make_blobs 生成随机数据集 X, y，centers=2 表示生成两个中心的数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 将数据集分割为训练集和测试集
    clf = DecisionTreeClassifier()
    # 初始化决策树分类器
    clf.fit(X_train, np.zeros_like(y_train))
    # 在训练集上拟合决策树分类器，将 y_train 替换为与之形状相同的全零数组
    # 使用 pytest 来测试代码中抛出的 ValueError 异常，检查异常消息是否包含特定的错误信息
    with pytest.raises(ValueError, match="need classifier with two classes"):
        # 调用 get_scorer 函数，使用 "roc_auc" 指标来评估分类器 clf 在测试集 X_test 上的性能
        get_scorer("roc_auc")(clf, X_test, y_test)
    
    # 用于处理概率评分器的情况
    with pytest.raises(ValueError, match="need classifier with two classes"):
        # 调用 get_scorer 函数，使用 "neg_log_loss" 指标来评估分类器 clf 在测试集 X_test 上的性能
        get_scorer("neg_log_loss")(clf, X_test, y_test)
# 定义一个测试函数，用于测试多标签指示数据的阈值评分器
def test_thresholded_scorers_multilabel_indicator_data():
    # 使用 make_multilabel_classification 生成多标签指示数据集 X 和标签 y
    X, y = make_multilabel_classification(allow_unlabeled=False, random_state=0)
    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 使用决策树分类器 DecisionTreeClassifier 进行拟合
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    # 预测测试集的概率
    y_proba = clf.predict_proba(X_test)
    # 使用 "roc_auc" 评分器计算得分 score1
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    # 计算 roc_auc_score 得分 score2
    score2 = roc_auc_score(y_test, np.vstack([p[:, -1] for p in y_proba]).T)
    # 断言 score1 和 score2 几乎相等
    assert_almost_equal(score1, score2)

    # 使用 OneVsRestClassifier 包装的决策树分类器进行拟合
    clf = OneVsRestClassifier(DecisionTreeClassifier())
    clf.fit(X_train, y_train)
    # 使用 "roc_auc" 评分器计算得分 score1
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    # 计算 roc_auc_score 得分 score2
    score2 = roc_auc_score(y_test, clf.predict_proba(X_test))
    # 断言 score1 和 score2 几乎相等
    assert_almost_equal(score1, score2)

    # 使用 OneVsRestClassifier 包装的线性支持向量机进行拟合
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    clf.fit(X_train, y_train)
    # 使用 "roc_auc" 评分器计算得分 score1
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    # 计算 roc_auc_score 得分 score2
    score2 = roc_auc_score(y_test, clf.decision_function(X_test))
    # 断言 score1 和 score2 几乎相等
    assert_almost_equal(score1, score2)


# 定义一个测试函数，用于测试有监督聚类评分器
def test_supervised_cluster_scorers():
    # 使用 make_blobs 生成聚类数据集 X 和标签 y
    X, y = make_blobs(random_state=0, centers=2)
    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 使用 KMeans 进行聚类，设置簇数为 3
    km = KMeans(n_clusters=3, n_init="auto")
    km.fit(X_train)
    # 遍历 CLUSTER_SCORERS 列表中的评分器名称
    for name in CLUSTER_SCORERS:
        # 使用当前评分器名称的评分器计算得分 score1
        score1 = get_scorer(name)(km, X_test, y_test)
        # 使用 cluster_module 中对应的评分函数计算得分 score2
        score2 = getattr(cluster_module, name)(y_test, km.predict(X_test))
        # 断言 score1 和 score2 几乎相等
        assert_almost_equal(score1, score2)


@ignore_warnings
def test_raises_on_score_list():
    # 测试当返回一个分数列表时是否引发适当的错误

    # 使用 make_blobs 生成分类数据集 X 和标签 y
    X, y = make_blobs(random_state=0)
    # 创建一个不对 F1 分数进行平均的评分器 f1_scorer_no_average
    f1_scorer_no_average = make_scorer(f1_score, average=None)
    clf = DecisionTreeClassifier()

    # 断言 cross_val_score 在使用不对 F1 分数进行平均的评分器时会引发 ValueError 错误
    with pytest.raises(ValueError):
        cross_val_score(clf, X, y, scoring=f1_scorer_no_average)

    # 创建一个 GridSearchCV 对象，使用不对 F1 分数进行平均的评分器进行网格搜索
    grid_search = GridSearchCV(
        clf, scoring=f1_scorer_no_average, param_grid={"max_depth": [1, 2]}
    )

    # 断言 grid_search 在使用不对 F1 分数进行平均的评分器时会引发 ValueError 错误
    with pytest.raises(ValueError):
        grid_search.fit(X, y)


@ignore_warnings
def test_classification_scorer_sample_weight():
    # 测试分类评分器是否支持样本权重或引发合理的错误

    # 使用 make_classification 生成分类数据集 X 和标签 y
    X, y = make_classification(random_state=0)
    # 使用 make_multilabel_classification 生成多标签分类数据集，并分割数据集
    _, y_ml = make_multilabel_classification(n_samples=X.shape[0], random_state=0)
    split = train_test_split(X, y, y_ml, random_state=0)
    X_train, X_test, y_train, y_test, y_ml_train, y_ml_test = split

    # 创建一个样本权重数组 sample_weight
    sample_weight = np.ones_like(y_test)
    sample_weight[:10] = 0

    # 获取每个指标的合理的评估器
    # 使用 _make_estimators 函数创建估算器对象，用于后续评估器的操作
    estimator = _make_estimators(X_train, y_train, y_ml_train)

    # 遍历所有可用的评分器名称
    for name in get_scorer_names():
        # 获取特定名称的评分器对象
        scorer = get_scorer(name)
        
        # 如果评分器名称在 REGRESSION_SCORERS 中，跳过回归评分器
        if name in REGRESSION_SCORERS:
            # 跳过回归评分器的处理
            continue
        
        # 如果评分器名称为 "top_k_accuracy"
        if name == "top_k_accuracy":
            # 对于二分类情况，当 k > 1 时，总会得到完美的评分
            scorer._kwargs = {"k": 1}
        
        # 如果评分器名称在 MULTILABEL_ONLY_SCORERS 中
        if name in MULTILABEL_ONLY_SCORERS:
            # 使用多标签测试集作为目标
            target = y_ml_test
        else:
            # 否则使用普通测试集作为目标
            target = y_test
        
        try:
            # 使用评分器评估估算器的预测结果，并考虑样本权重 sample_weight
            weighted = scorer(
                estimator[name], X_test, target, sample_weight=sample_weight
            )
            
            # 忽略部分样本进行评估
            ignored = scorer(estimator[name], X_test[10:], target[10:])
            
            # 不考虑样本权重进行评估
            unweighted = scorer(estimator[name], X_test, target)
            
            # 应当不会引发异常。如果 sample_weight 为 None，则应该被忽略。
            _ = scorer(estimator[name], X_test[:10], target[:10], sample_weight=None)
            
            # 断言加权评估和非加权评估结果不同，用于验证评分器的行为是否正确
            assert weighted != unweighted, (
                f"scorer {name} behaves identically when called with "
                f"sample weights: {weighted} vs {unweighted}"
            )
            
            # 断言加权评估和忽略部分样本评估结果相似，验证评分器的行为是否正确
            assert_almost_equal(
                weighted,
                ignored,
                err_msg=(
                    f"scorer {name} behaves differently "
                    "when ignoring samples and setting "
                    f"sample_weight to 0: {weighted} vs {ignored}"
                ),
            )

        except TypeError as e:
            # 断言调用评分器时是否抛出了与 sample_weight 相关的异常
            assert "sample_weight" in str(e), (
                f"scorer {name} raises unhelpful exception when called "
                f"with sample weights: {str(e)}"
            )
# 装饰器，用于忽略警告
@ignore_warnings
# 测试回归评分器是否支持样本权重或引发合理的错误
def test_regression_scorer_sample_weight():
    # 生成具有101个样本和20个特征的回归数据集，固定随机种子为0
    X, y = make_regression(n_samples=101, n_features=20, random_state=0)
    # 确保目标变量 y 是正值
    y = _require_positive_y(y)
    # 将数据集划分为训练集和测试集，随机种子为0
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 创建一个与 y_test 形状相同的全一数组作为样本权重
    sample_weight = np.ones_like(y_test)
    # 将前11个样本的权重设为0，用于测试负中位数绝对误差
    sample_weight[:11] = 0

    # 创建一个决策树回归器对象，随机种子为0，并使用训练数据拟合回归器
    reg = DecisionTreeRegressor(random_state=0)
    reg.fit(X_train, y_train)

    # 遍历所有可用的评分器名称
    for name in get_scorer_names():
        # 根据评分器名称获取评分器对象
        scorer = get_scorer(name)
        # 如果评分器不属于回归评分器列表中的评分器，则跳过
        if name not in REGRESSION_SCORERS:
            continue
        try:
            # 使用样本权重计算加权得分
            weighted = scorer(reg, X_test, y_test, sample_weight=sample_weight)
            # 忽略前11个样本后再计算得分
            ignored = scorer(reg, X_test[11:], y_test[11:])
            # 不使用样本权重计算得分
            unweighted = scorer(reg, X_test, y_test)
            # 断言加权得分与未加权得分不相等，若相等则抛出异常
            assert weighted != unweighted, (
                f"scorer {name} behaves identically when called with "
                f"sample weights: {weighted} vs {unweighted}"
            )
            # 断言加权得分与忽略前11个样本后得分近似相等，否则抛出异常
            assert_almost_equal(
                weighted,
                ignored,
                err_msg=(
                    f"scorer {name} behaves differently "
                    "when ignoring samples and setting "
                    f"sample_weight to 0: {weighted} vs {ignored}"
                ),
            )

        # 捕获 TypeError 异常，确保异常信息中包含 "sample_weight"
        except TypeError as e:
            assert "sample_weight" in str(e), (
                f"scorer {name} raises unhelpful exception when called "
                f"with sample weights: {str(e)}"
            )


# 使用参数化测试，对每个评分器名称进行测试
@pytest.mark.parametrize("name", get_scorer_names())
def test_scorer_memmap_input(name, memmap_data_and_estimators):
    # 非回归测试，用于 #6147: 部分评分函数在 memmap 数据上计算时返回单个 memmap 而非标量浮点数值。
    X_mm, y_mm, y_ml_mm, estimators = memmap_data_and_estimators

    # 如果评分器名称在要求目标变量为正值的列表中，则对 y_mm 和 y_ml_mm 进行要求正值处理
    if name in REQUIRE_POSITIVE_Y_SCORERS:
        y_mm_1 = _require_positive_y(y_mm)
        y_ml_mm_1 = _require_positive_y(y_ml_mm)
    else:
        y_mm_1, y_ml_mm_1 = y_mm, y_ml_mm

    # 对于 P/R 分数，会出现 UndefinedMetricWarning
    with ignore_warnings():
        # 获取评分器和估计器
        scorer, estimator = get_scorer(name), estimators[name]
        # 如果评分器属于多标签专用评分器，则使用 y_ml_mm_1 计算得分
        if name in MULTILABEL_ONLY_SCORERS:
            score = scorer(estimator, X_mm, y_ml_mm_1)
        else:
            score = scorer(estimator, X_mm, y_mm_1)
        # 断言得分为数字类型
        assert isinstance(score, numbers.Number), name


# 测试评分不是度量值的函数
def test_scoring_is_not_metric():
    # 确保使用 f1_score 作为评分引发 ValueError 异常
    with pytest.raises(ValueError, match="make_scorer"):
        check_scoring(LogisticRegression(), scoring=f1_score)
    # 确保使用 roc_auc_score 作为评分引发 ValueError 异常
    with pytest.raises(ValueError, match="make_scorer"):
        check_scoring(LogisticRegression(), scoring=roc_auc_score)
    # 确保使用 r2_score 作为评分引发 ValueError 异常
    with pytest.raises(ValueError, match="make_scorer"):
        check_scoring(Ridge(), scoring=r2_score)
    # 使用 pytest 的断言检查，验证调用 check_scoring 函数时是否会抛出 ValueError 异常，并检查异常消息是否包含 "make_scorer"
    with pytest.raises(ValueError, match="make_scorer"):
        # 调用 check_scoring 函数，传入 KMeans() 实例作为参数，并指定 scoring 参数为 cluster_module.adjusted_rand_score
        check_scoring(KMeans(), scoring=cluster_module.adjusted_rand_score)
    
    # 使用 pytest 的断言检查，验证调用 check_scoring 函数时是否会抛出 ValueError 异常，并检查异常消息是否包含 "make_scorer"
    with pytest.raises(ValueError, match="make_scorer"):
        # 调用 check_scoring 函数，传入 KMeans() 实例作为参数，并指定 scoring 参数为 cluster_module.rand_score
        check_scoring(KMeans(), scoring=cluster_module.rand_score)
# 使用 pytest.mark.parametrize 装饰器定义多个参数化测试用例，每个用例包含多个参数
@pytest.mark.parametrize(
    (
        "scorers,expected_predict_count,"
        "expected_predict_proba_count,expected_decision_func_count"
    ),
    [
        (
            {
                "a1": "accuracy",
                "a2": "accuracy",
                "ll1": "neg_log_loss",
                "ll2": "neg_log_loss",
                "ra1": "roc_auc",
                "ra2": "roc_auc",
            },
            1,
            1,
            1,
        ),
        (["roc_auc", "accuracy"], 1, 0, 1),
        (["neg_log_loss", "accuracy"], 1, 1, 0),
    ],
)
# 定义名为 test_multimetric_scorer_calls_method_once 的测试函数，接受多个参数化的输入
def test_multimetric_scorer_calls_method_once(
    scorers,
    expected_predict_count,
    expected_predict_proba_count,
    expected_decision_func_count,
):
    # 创建示例数据 X 和 y，其中 X 是二维数组，y 是一维数组
    X, y = np.array([[1], [1], [0], [0], [0]]), np.array([0, 1, 1, 1, 0])

    # 创建 Mock 对象 mock_est 作为估计器
    mock_est = Mock()
    # 设置 mock_est 的 _estimator_type 属性为 "classifier"
    mock_est._estimator_type = "classifier"
    
    # 创建 Mock 对象 fit_func 作为估计器的 fit 方法
    fit_func = Mock(return_value=mock_est, name="fit")
    fit_func.__name__ = "fit"
    # 创建 Mock 对象 predict_func 作为估计器的 predict 方法
    predict_func = Mock(return_value=y, name="predict")
    predict_func.__name__ = "predict"

    # 生成随机概率值作为正类的概率
    pos_proba = np.random.rand(X.shape[0])
    # 构造概率数组，每个样本的概率由 1-pos_proba 和 pos_proba 组成
    proba = np.c_[1 - pos_proba, pos_proba]
    # 创建 Mock 对象 predict_proba_func 作为估计器的 predict_proba 方法
    predict_proba_func = Mock(return_value=proba, name="predict_proba")
    predict_proba_func.__name__ = "predict_proba"
    # 创建 Mock 对象 decision_function_func 作为估计器的 decision_function 方法
    decision_function_func = Mock(return_value=pos_proba, name="decision_function")
    decision_function_func.__name__ = "decision_function"

    # 将 Mock 方法设置到 mock_est 对象中
    mock_est.fit = fit_func
    mock_est.predict = predict_func
    mock_est.predict_proba = predict_proba_func
    mock_est.decision_function = decision_function_func
    # 设置 mock_est 的 classes_ 属性为类标签的数组 [0, 1]
    mock_est.classes_ = np.array([0, 1])

    # 使用 _check_multimetric_scoring 函数获取多指标评分器的字典
    scorer_dict = _check_multimetric_scoring(LogisticRegression(), scorers)
    # 创建 _MultimetricScorer 对象 multi_scorer，使用 scorers 参数初始化
    multi_scorer = _MultimetricScorer(scorers=scorer_dict)
    # 调用 multi_scorer 对象，传入 mock_est、X 和 y 进行评估，获取结果
    results = multi_scorer(mock_est, X, y)

    # 断言：比较 scorers 字典的键集合与 results 字典的键集合，确保一致
    assert set(scorers) == set(results)  # compare dict keys

    # 断言：验证 predict_func 被调用次数是否符合预期
    assert predict_func.call_count == expected_predict_count
    # 断言：验证 predict_proba_func 被调用次数是否符合预期
    assert predict_proba_func.call_count == expected_predict_proba_count
    # 断言：验证 decision_function_func 被调用次数是否符合预期
    assert decision_function_func.call_count == expected_decision_func_count
    # 创建一个模拟的K最近邻分类器对象，设置邻居数为1
    clf = MockKNeighborsClassifier(n_neighbors=1)
    # 使用输入数据X和标签y对分类器进行训练
    clf.fit(X, y)
    
    # 检查并获取多指标评分器的字典
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    # 创建一个多指标评分器对象，传入评分器字典
    scorer = _MultimetricScorer(scorers=scorer_dict)
    # 对分类器进行评分
    scorer(clf, X, y)
    
    # 断言预测概率调用次数为1，用于验证模型的预测概率是否被调用过
    assert predict_proba_call_cnt == 1
def test_multimetric_scorer_calls_method_once_regressor_threshold():
    # 初始化预测调用计数器
    predict_called_cnt = 0

    # 定义一个 MockDecisionTreeRegressor 类，继承自 DecisionTreeRegressor
    class MockDecisionTreeRegressor(DecisionTreeRegressor):
        # 重写 predict 方法，用于统计预测调用次数
        def predict(self, X):
            nonlocal predict_called_cnt
            predict_called_cnt += 1
            return super().predict(X)

    # 创建示例数据 X 和 y
    X, y = np.array([[1], [1], [0], [0], [0]]), np.array([0, 1, 1, 1, 0])

    # 实例化 MockDecisionTreeRegressor 对象
    clf = MockDecisionTreeRegressor()
    # 使用 X, y 数据进行拟合
    clf.fit(X, y)

    # 定义评分器字典 scorers
    scorers = {"neg_mse": "neg_mean_squared_error", "r2": "r2"}
    # 调用 _check_multimetric_scoring 函数获取评分器字典
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    # 实例化 _MultimetricScorer 对象
    scorer = _MultimetricScorer(scorers=scorer_dict)
    # 使用评分器对 clf 模型进行评分
    scorer(clf, X, y)

    # 断言预测调用次数为 1
    assert predict_called_cnt == 1


def test_multimetric_scorer_sanity_check():
    # 验证评分字典返回结果与分别调用每个评分器的结果一致
    scorers = {
        "a1": "accuracy",
        "a2": "accuracy",
        "ll1": "neg_log_loss",
        "ll2": "neg_log_loss",
        "ra1": "roc_auc",
        "ra2": "roc_auc",
    }

    # 创建示例数据 X 和 y
    X, y = make_classification(random_state=0)

    # 实例化决策树分类器
    clf = DecisionTreeClassifier()
    # 使用 X, y 数据进行拟合
    clf.fit(X, y)

    # 调用 _check_multimetric_scoring 函数获取评分器字典
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    # 实例化 _MultimetricScorer 对象
    multi_scorer = _MultimetricScorer(scorers=scorer_dict)

    # 使用 multi_scorer 对象对 clf 模型进行评分
    result = multi_scorer(clf, X, y)

    # 分别调用每个评分器获取单独的分数
    separate_scores = {
        name: get_scorer(name)(clf, X, y)
        for name in ["accuracy", "neg_log_loss", "roc_auc"]
    }

    # 断言 multi_scorer 返回的结果与单独调用评分器的结果一致
    for key, value in result.items():
        score_name = scorers[key]
        assert_allclose(value, separate_scores[score_name])


@pytest.mark.parametrize("raise_exc", [True, False])
def test_multimetric_scorer_exception_handling(raise_exc):
    """Check that the calling of the `_MultimetricScorer` returns
    exception messages in the result dict for the failing scorers
    in case of `raise_exc` is `False` and if `raise_exc` is `True`,
    then the proper exception is raised.
    """
    # 定义评分器字典 scorers，包括一个会失败的评分器和一个不会失败的评分器
    scorers = {
        "failing_1": "neg_mean_squared_log_error",
        "non_failing": "neg_median_absolute_error",
        "failing_2": "neg_mean_squared_log_error",
    }

    # 创建示例数据 X 和 y
    X, y = make_classification(
        n_samples=50, n_features=2, n_redundant=0, random_state=0
    )
    # 调整 y 值，使得 neg_mean_squared_log_error 评分器会失败
    y *= -1  # neg_mean_squared_log_error 在 y 包含负值时会失败

    # 实例化决策树分类器，并使用 X, y 数据进行拟合
    clf = DecisionTreeClassifier().fit(X, y)

    # 调用 _check_multimetric_scoring 函数获取评分器字典
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    # 实例化 _MultimetricScorer 对象，根据 raise_exc 参数决定是否抛出异常
    multi_scorer = _MultimetricScorer(scorers=scorer_dict, raise_exc=raise_exc)

    # 定义期望的错误信息
    error_msg = (
        "Mean Squared Logarithmic Error cannot be used when targets contain"
        " negative values."
    )

    # 如果 raise_exc 为 True，则期望抛出 ValueError 异常，且错误信息匹配 error_msg
    if raise_exc:
        with pytest.raises(ValueError, match=error_msg):
            multi_scorer(clf, X, y)
    else:
        # 调用 multi_scorer 函数对分类器 clf 在数据集 X, y 上进行评分
        result = multi_scorer(clf, X, y)

        # 从评分结果中获取失败信息1和非失败分数
        exception_message_1 = result["failing_1"]
        score = result["non_failing"]
        # 从评分结果中获取失败信息2

        exception_message_2 = result["failing_2"]

        # 使用断言确保 exception_message_1 是字符串类型且包含指定的错误消息
        assert isinstance(exception_message_1, str) and error_msg in exception_message_1
        # 使用断言确保 score 是浮点数类型
        assert isinstance(score, float)
        # 使用断言确保 exception_message_2 是字符串类型且包含指定的错误消息
        assert isinstance(exception_message_2, str) and error_msg in exception_message_2
@pytest.mark.parametrize(
    "scorer_name, metric",
    [
        ("roc_auc_ovr", partial(roc_auc_score, multi_class="ovr")),
        ("roc_auc_ovo", partial(roc_auc_score, multi_class="ovo")),
        (
            "roc_auc_ovr_weighted",
            partial(roc_auc_score, multi_class="ovr", average="weighted"),
        ),
        (
            "roc_auc_ovo_weighted",
            partial(roc_auc_score, multi_class="ovo", average="weighted"),
        ),
    ],
)
def test_multiclass_roc_proba_scorer(scorer_name, metric):
    # 获取指定评分器函数
    scorer = get_scorer(scorer_name)
    # 创建一个多类别分类问题的数据集
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=20, random_state=0
    )
    # 使用逻辑回归模型拟合数据集
    lr = LogisticRegression().fit(X, y)
    # 预测每个样本的类别概率
    y_proba = lr.predict_proba(X)
    # 计算预期的评分值
    expected_score = metric(y, y_proba)

    # 断言评分器的评分结果与预期的评分值接近
    assert scorer(lr, X, y) == pytest.approx(expected_score)


def test_multiclass_roc_proba_scorer_label():
    # 创建一个基于标签的多类别分类问题的评分器
    scorer = make_scorer(
        roc_auc_score,
        multi_class="ovo",
        labels=[0, 1, 2],
        response_method="predict_proba",
    )
    # 创建一个多类别分类问题的数据集
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=20, random_state=0
    )
    # 使用逻辑回归模型拟合数据集
    lr = LogisticRegression().fit(X, y)
    # 预测每个样本的类别概率
    y_proba = lr.predict_proba(X)

    # 将多类别标签转换为二进制标签
    y_binary = y == 0
    # 计算预期的评分值
    expected_score = roc_auc_score(
        y_binary, y_proba, multi_class="ovo", labels=[0, 1, 2]
    )

    # 断言评分器的评分结果与预期的评分值接近
    assert scorer(lr, X, y_binary) == pytest.approx(expected_score)


@pytest.mark.parametrize(
    "scorer_name",
    ["roc_auc_ovr", "roc_auc_ovo", "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"],
)
def test_multiclass_roc_no_proba_scorer_errors(scorer_name):
    # 感知机模型没有 predict_proba 方法，期望抛出 AttributeError 异常
    scorer = get_scorer(scorer_name)
    # 创建一个多类别分类问题的数据集
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=20, random_state=0
    )
    # 使用感知机模型拟合数据集
    lr = Perceptron().fit(X, y)
    # 预期的错误信息
    msg = "Perceptron has none of the following attributes: predict_proba."
    # 断言调用评分器时抛出期望的异常
    with pytest.raises(AttributeError, match=msg):
        scorer(lr, X, y)


@pytest.fixture
def string_labeled_classification_problem():
    """Train a classifier on binary problem with string target.

    The classifier is trained on a binary classification problem where the
    minority class of interest has a string label that is intentionally not the
    greatest class label using the lexicographic order. In this case, "cancer"
    is the positive label, and `classifier.classes_` is
    `["cancer", "not cancer"]`.

    In addition, the dataset is imbalanced to better identify problems when
    using non-symmetric performance metrics such as f1-score, average precision
    and so on.

    Returns
    -------
    classifier : estimator object
        Trained classifier on the binary problem.
    X_test : ndarray of shape (n_samples, n_features)
        Data to be used as testing set in tests.
    y_test : ndarray of shape (n_samples,), dtype=object
        Binary target where labels are strings.
    """
    y_pred : ndarray of shape (n_samples,), dtype=object
        Prediction of `classifier` when predicting for `X_test`.
    y_pred_proba : ndarray of shape (n_samples, 2), dtype=np.float64
        Probabilities of `classifier` when predicting for `X_test`.
    y_pred_decision : ndarray of shape (n_samples,), dtype=np.float64
        Decision function values of `classifier` when predicting on `X_test`.
    """
    from sklearn.datasets import load_breast_cancer  # 导入 breast_cancer 数据集
    from sklearn.utils import shuffle  # 导入 shuffle 函数

    X, y = load_breast_cancer(return_X_y=True)
    # 创建一个高度不平衡的分类任务

    idx_positive = np.flatnonzero(y == 1)  # 找出类别为1的索引
    idx_negative = np.flatnonzero(y == 0)  # 找出类别为0的索引
    idx_selected = np.hstack([idx_negative, idx_positive[:25]])  # 选择少量类别0和前25个类别1的索引
    X, y = X[idx_selected], y[idx_selected]  # 根据选定的索引筛选数据集
    X, y = shuffle(X, y, random_state=42)  # 打乱数据集顺序，使用随机种子42

    # 只使用前两个特征使问题更加复杂
    X = X[:, :2]  # 仅保留前两个特征列
    y = np.array(["cancer" if c == 1 else "not cancer" for c in y], dtype=object)  # 根据类别创建标签数组

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,  # 使用标签y进行分层采样
        random_state=0,
    )
    classifier = LogisticRegression().fit(X_train, y_train)  # 训练逻辑回归分类器
    y_pred = classifier.predict(X_test)  # 预测测试集的类别
    y_pred_proba = classifier.predict_proba(X_test)  # 预测测试集的类别概率
    y_pred_decision = classifier.decision_function(X_test)  # 预测测试集的决策函数值

    return classifier, X_test, y_test, y_pred, y_pred_proba, y_pred_decision
# 定义一个测试函数，用于测试平均精度（average precision）的评分函数对于特定正类标签（pos_label）的行为
def test_average_precision_pos_label(string_labeled_classification_problem):
    # 解包传入的字符串标记分类问题的元组，获取需要的变量
    (
        clf,
        X_test,
        y_test,
        _,
        y_pred_proba,
        y_pred_decision,
    ) = string_labeled_classification_problem

    # 设定正类标签为"cancer"
    pos_label = "cancer"
    # 从预测的概率中选择第一列作为正类的概率
    y_pred_proba = y_pred_proba[:, 0]
    # 反转决策值的符号
    y_pred_decision = y_pred_decision * -1
    # 断言分类器的第一个类别是我们设定的正类标签
    assert clf.classes_[0] == pos_label

    # 使用概率估计调用评分函数，检查概率估计和决策值是否导致相同的结果
    ap_proba = average_precision_score(y_test, y_pred_proba, pos_label=pos_label)
    ap_decision_function = average_precision_score(
        y_test, y_pred_decision, pos_label=pos_label
    )
    assert ap_proba == pytest.approx(ap_decision_function)

    # 创建一个评分器，需要传递`pos_label`参数
    # 检查如果没有提供`pos_label`参数，是否会失败
    average_precision_scorer = make_scorer(
        average_precision_score,
        response_method=("decision_function", "predict_proba"),
    )
    err_msg = "pos_label=1 is not a valid label. It should be one of "
    with pytest.raises(ValueError, match=err_msg):
        average_precision_scorer(clf, X_test, y_test)

    # 否则，评分器应该给出与直接调用评分函数相同的结果
    average_precision_scorer = make_scorer(
        average_precision_score,
        response_method=("decision_function", "predict_proba"),
        pos_label=pos_label,
    )
    ap_scorer = average_precision_scorer(clf, X_test, y_test)
    assert ap_scorer == pytest.approx(ap_proba)

    # 上述评分器调用使用了`clf.decision_function`。我们将强制它使用`clf.predict_proba`。
    clf_without_predict_proba = deepcopy(clf)

    def _predict_proba(self, X):
        raise NotImplementedError

    # 将`clf.predict_proba`替换为抛出`NotImplementedError`异常的偏函数
    clf_without_predict_proba.predict_proba = partial(
        _predict_proba, clf_without_predict_proba
    )
    # 检查
    with pytest.raises(NotImplementedError):
        clf_without_predict_proba.predict_proba(X_test)

    # 使用新的评分器调用`clf_without_predict_proba`，检查是否与`ap_proba`相等
    ap_scorer = average_precision_scorer(clf_without_predict_proba, X_test, y_test)
    assert ap_scorer == pytest.approx(ap_proba)


# 定义一个测试函数，用于测试Brier得分损失（brier score loss）对于特定正类标签（pos_label）的行为
def test_brier_score_loss_pos_label(string_labeled_classification_problem):
    # 解包传入的字符串标记分类问题的元组，获取需要的变量
    clf, X_test, y_test, _, y_pred_proba, _ = string_labeled_classification_problem

    # 设定正类标签为"cancer"
    pos_label = "cancer"
    # 断言分类器的第一个类别是我们设定的正类标签
    assert clf.classes_[0] == pos_label

    # 计算使用正类标签为"cancer"的Brier得分损失
    brier_pos_cancer = brier_score_loss(y_test, y_pred_proba[:, 0], pos_label="cancer")
    # 计算预测为 "not cancer" 类别时的 Brier 评分
    brier_pos_not_cancer = brier_score_loss(
        y_test, y_pred_proba[:, 1], pos_label="not cancer"
    )
    # 断言预测为 "cancer" 和 "not cancer" 类别时的 Brier 评分近似相等
    assert brier_pos_cancer == pytest.approx(brier_pos_not_cancer)

    # 创建一个用于评估 Brier 评分的评分器
    brier_scorer = make_scorer(
        brier_score_loss,
        response_method="predict_proba",
        pos_label=pos_label,
    )
    # 断言评分器计算的 Brier 评分与预测为 "cancer" 类别时的 Brier 评分近似相等
    assert brier_scorer(clf, X_test, y_test) == pytest.approx(brier_pos_cancer)
@pytest.mark.parametrize(
    "score_func", [f1_score, precision_score, recall_score, jaccard_score]
)
# 使用参数化测试来测试多个评分函数
def test_non_symmetric_metric_pos_label(
    score_func, string_labeled_classification_problem
):
    # 检查当提供 `pos_label` 时 `_Scorer` 是否会导致正确的评分。
    # 我们检查所有支持的评分指标。
    # 注意: 最终可能会出现 "scorer tags"。
    clf, X_test, y_test, y_pred, _, _ = string_labeled_classification_problem

    pos_label = "cancer"
    assert clf.classes_[0] == pos_label

    score_pos_cancer = score_func(y_test, y_pred, pos_label="cancer")
    score_pos_not_cancer = score_func(y_test, y_pred, pos_label="not cancer")

    assert score_pos_cancer != pytest.approx(score_pos_not_cancer)

    # 创建一个针对特定 `pos_label` 的评分器对象
    scorer = make_scorer(score_func, pos_label=pos_label)
    assert scorer(clf, X_test, y_test) == pytest.approx(score_pos_cancer)


@pytest.mark.parametrize(
    "scorer",
    [
        make_scorer(
            average_precision_score,
            response_method=("decision_function", "predict_proba"),
            pos_label="xxx",
        ),
        make_scorer(brier_score_loss, response_method="predict_proba", pos_label="xxx"),
        make_scorer(f1_score, pos_label="xxx"),
    ],
    ids=["non-thresholded scorer", "probability scorer", "thresholded scorer"],
)
# 测试当传入未知的 `pos_label` 时是否会正确抛出异常
def test_scorer_select_proba_error(scorer):
    X, y = make_classification(
        n_classes=2, n_informative=3, n_samples=20, random_state=0
    )
    lr = LogisticRegression().fit(X, y)
    assert scorer._kwargs["pos_label"] not in np.unique(y).tolist()

    err_msg = "is not a valid label"
    with pytest.raises(ValueError, match=err_msg):
        scorer(lr, X, y)


def test_get_scorer_return_copy():
    # 测试 `get_scorer` 是否返回一个副本
    assert get_scorer("roc_auc") is not get_scorer("roc_auc")


def test_scorer_no_op_multiclass_select_proba():
    # 检查在多类问题上调用 `_Scorer` 不会引发异常，
    # 即使在评分时 `y_true` 是二元的情况下。
    # 在这种情况下不应调用 `_select_proba_binary`。
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=20, random_state=0
    )
    lr = LogisticRegression().fit(X, y)

    mask_last_class = y == lr.classes_[-1]
    X_test, y_test = X[~mask_last_class], y[~mask_last_class]
    assert_array_equal(np.unique(y_test), lr.classes_[:-1])

    # 创建一个多类问题的评分器对象
    scorer = make_scorer(
        roc_auc_score,
        response_method="predict_proba",
        multi_class="ovo",
        labels=lr.classes_,
    )
    scorer(lr, X_test, y_test)


@pytest.mark.parametrize("name", get_scorer_names())
# 测试只有在启用特征标志时才能使用 `set_score_request`。
def test_scorer_set_score_request_raises(name):
    scorer = get_scorer(name)
    # 使用 pytest 框架来测试代码，检查是否会抛出 RuntimeError 异常，并且异常消息匹配特定字符串
    with pytest.raises(RuntimeError, match="This method is only available"):
        # 调用 scorer 对象的 set_score_request() 方法，预期会抛出 RuntimeError 异常
        scorer.set_score_request()
# 使用 pytest.mark.usefixtures 装饰器标记测试函数，启用 SLEP006 的功能
# 并且使用 pytest.mark.parametrize 参数化测试，参数为 get_scorer_names() 返回的名字列表，ids 参数也使用 get_scorer_names() 返回的名字列表
@pytest.mark.usefixtures("enable_slep006")
@pytest.mark.parametrize("name", get_scorer_names(), ids=get_scorer_names())
# 定义测试函数 test_scorer_metadata_request，用于测试评分器的元数据请求
def test_scorer_metadata_request(name):
    """Testing metadata requests for scorers.

    This test checks many small things in a large test, to reduce the
    boilerplate required for each section.
    """
    # 获取评分器对象
    scorer = get_scorer(name)
    # 断言评分器对象具有 set_score_request 方法
    assert hasattr(scorer, "set_score_request")
    # 断言评分器对象具有 get_metadata_routing 方法
    assert hasattr(scorer, "get_metadata_routing")

    # 检查默认情况下是否未请求任何元数据
    assert_request_is_empty(scorer.get_metadata_routing())

    # 调用 set_score_request 方法设置 sample_weight=True
    weighted_scorer = scorer.set_score_request(sample_weight=True)
    # 断言 set_score_request 方法会改变当前评分器实例，而不是返回新的实例
    assert weighted_scorer is scorer

    # 确保评分器在除 `score` 方法外的其他方法上不请求任何内容，并且在 `score` 方法上请求的值是正确的
    assert_request_is_empty(weighted_scorer.get_metadata_routing(), exclude="score")
    assert (
        weighted_scorer.get_metadata_routing().score.requests["sample_weight"] is True
    )

    # 确保将评分器放入路由器时，默认情况下不会请求任何内容
    router = MetadataRouter(owner="test").add(
        scorer=get_scorer(name),
        method_mapping=MethodMapping().add(caller="score", callee="score"),
    )
    # 确保如果传递了 `sample_weight` 参数，会引发 TypeError 异常，错误信息匹配 "got unexpected argument"
    with pytest.raises(TypeError, match="got unexpected argument"):
        router.validate_metadata(params={"sample_weight": 1}, method="score")
    # 确保即使传递了 `sample_weight` 参数，也不会路由该参数
    routed_params = router.route_params(params={"sample_weight": 1}, caller="score")
    assert not routed_params.scorer.score

    # 确保将 weighted_scorer 放入路由器时请求 sample_weight 参数
    router = MetadataRouter(owner="test").add(
        scorer=weighted_scorer,
        method_mapping=MethodMapping().add(caller="score", callee="score"),
    )
    router.validate_metadata(params={"sample_weight": 1}, method="score")
    routed_params = router.route_params(params={"sample_weight": 1}, caller="score")
    # 断言 routed_params.scorer.score.keys() 的结果为 ["sample_weight"]
    assert list(routed_params.scorer.score.keys()) == ["sample_weight"]


# 使用 pytest.mark.usefixtures 装饰器标记测试函数，启用 SLEP006 的功能
@pytest.mark.usefixtures("enable_slep006")
# 定义测试函数 test_metadata_kwarg_conflict，测试用户如果同时作为构造函数参数传递元数据和在 __call__ 中传递时是否会收到正确的警告
def test_metadata_kwarg_conflict():
    """This test makes sure the right warning is raised if the user passes
    some metadata both as a constructor to make_scorer, and during __call__.
    """
    # 生成分类数据集 X, y
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=20, random_state=0
    )
    # 使用 LogisticRegression 拟合数据
    lr = LogisticRegression().fit(X, y)

    # 创建一个评分器 scorer，使用 make_scorer 函数，并设置一些参数
    scorer = make_scorer(
        roc_auc_score,
        response_method="predict_proba",
        multi_class="ovo",
        labels=lr.classes_,
    )
    # 使用 pytest.warns 断言会引发 UserWarning 警告，并且警告信息匹配 "already set as kwargs"
    with pytest.warns(UserWarning, match="already set as kwargs"):
        scorer.set_score_request(labels=True)
    # 使用 pytest 库中的 warns 函数来捕获特定警告类型的上下文
    with pytest.warns(UserWarning, match="There is an overlap"):
        # 调用 scorer 函数，并传入参数 lr, X, y, labels=lr.classes_
        # 如果发出 UserWarning 警告并且警告消息包含 "There is an overlap" 字符串，则测试通过
        scorer(lr, X, y, labels=lr.classes_)
@pytest.mark.usefixtures("enable_slep006")
# 使用 pytest 的标记，确保在测试运行时启用 slep006
def test_PassthroughScorer_set_score_request():
    """Test that _PassthroughScorer.set_score_request adds the correct metadata request
    on itself and doesn't change its estimator's routing."""
    # 创建一个 LogisticRegression 实例作为评分估算器
    est = LogisticRegression().set_score_request(sample_weight="estimator_weights")
    
    # 使用 check_scoring 函数创建一个 `_PassthroughScorer` 实例
    scorer = check_scoring(est, None)
    
    # 断言检查评分器的元数据路由请求
    assert (
        scorer.get_metadata_routing().score.requests["sample_weight"]
        == "estimator_weights"
    )

    # 修改评分器的元数据路由请求
    scorer.set_score_request(sample_weight="scorer_weights")
    
    # 再次断言检查评分器的元数据路由请求是否已更新
    assert (
        scorer.get_metadata_routing().score.requests["sample_weight"]
        == "scorer_weights"
    )

    # 确保更改 _PassthroughScorer 对象不会影响评估器本身
    assert (
        est.get_metadata_routing().score.requests["sample_weight"]
        == "estimator_weights"
    )


def test_PassthroughScorer_set_score_request_raises_without_routing_enabled():
    """Test that _PassthroughScorer.set_score_request raises if metadata routing is
    disabled."""
    # 使用 LogisticRegression 创建一个评分器实例
    scorer = check_scoring(LogisticRegression(), None)
    
    # 定义错误消息
    msg = "This method is only available when metadata routing is enabled."

    # 使用 pytest 的断言检查是否引发了预期的 RuntimeError 异常
    with pytest.raises(RuntimeError, match=msg):
        scorer.set_score_request(sample_weight="my_weights")


@pytest.mark.usefixtures("enable_slep006")
# 使用 pytest 的标记，确保在测试运行时启用 slep006
def test_multimetric_scoring_metadata_routing():
    # Test that _MultimetricScorer properly routes metadata.
    # 定义三个用于评分的函数 score1, score2, score3
    def score1(y_true, y_pred):
        return 1

    def score2(y_true, y_pred, sample_weight="test"):
        # 确保 sample_weight 未被传递
        assert sample_weight == "test"
        return 1

    def score3(y_true, y_pred, sample_weight=None):
        # 确保 sample_weight 被传递
        assert sample_weight is not None
        return 1

    # 创建一个包含三个评分器的字典
    scorers = {
        "score1": make_scorer(score1),
        "score2": make_scorer(score2).set_score_request(sample_weight=False),
        "score3": make_scorer(score3).set_score_request(sample_weight=True),
    }

    # 使用 make_classification 生成一些样本数据 X, y
    X, y = make_classification(
        n_samples=50, n_features=2, n_redundant=0, random_state=0
    )

    # 使用 DecisionTreeClassifier 训练一个分类器 clf
    clf = DecisionTreeClassifier().fit(X, y)

    # 使用 _check_multimetric_scoring 函数对分类器 clf 和评分器 scorers 进行检查
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    
    # 创建一个 _MultimetricScorer 实例
    multi_scorer = _MultimetricScorer(scorers=scorer_dict)
    
    # 这应该失败，因为未启用元数据路由，因此不支持为不同的评分器使用不同的元数据
    # TODO: 当 enable_metadata_routing 被弃用时移除
    with config_context(enable_metadata_routing=False):
        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            multi_scorer(clf, X, y, sample_weight=1)

    # 当启用了路由时，这个测试通过
    multi_scorer(clf, X, y, sample_weight=1)


def test_kwargs_without_metadata_routing_error():
    # Test that kwargs are not supported in scorers if metadata routing is not
    # enabled.
    # TODO: remove when enable_metadata_routing is deprecated
    # 定义一个名为 score 的函数，接受 y_true、y_pred 和可选参数 param，并返回 1
    def score(y_true, y_pred, param=None):
        return 1  # pragma: no cover

    # 使用 make_classification 生成一个包含 50 个样本、2个特征、无冗余特征的数据集 X 和对应的标签 y
    X, y = make_classification(
        n_samples=50, n_features=2, n_redundant=0, random_state=0
    )

    # 创建一个决策树分类器 clf，并使用生成的数据集 X 和 y 进行训练
    clf = DecisionTreeClassifier().fit(X, y)

    # 创建一个名为 scorer 的评分器，使用 score 函数作为评分方法
    scorer = make_scorer(score)

    # 使用 config_context 创建一个上下文，设置 enable_metadata_routing=False
    with config_context(enable_metadata_routing=False):
        # 在上下文内部使用 pytest.raises 捕获 ValueError 异常，确保异常消息包含指定字符串
        with pytest.raises(
            ValueError, match="is only supported if enable_metadata_routing=True"
        ):
            # 调用 scorer 对 clf、X、y 进行评分，并传入参数 param="blah"
            scorer(clf, X, y, param="blah")
# 定义一个测试函数，用于测试多标签指示器矩阵的评分器是否正常工作
def test_get_scorer_multilabel_indicator():
    """Check that our scorer deal with multi-label indicator matrices.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/26817
    """
    # 创建一个多标签分类数据集，包含72个样本，3个类别
    X, Y = make_multilabel_classification(n_samples=72, n_classes=3, random_state=0)
    # 将数据集划分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

    # 使用K最近邻分类器作为估计器，对训练集进行拟合
    estimator = KNeighborsClassifier().fit(X_train, Y_train)

    # 获取平均精度评分器，对测试集进行评分
    score = get_scorer("average_precision")(estimator, X_test, Y_test)
    # 断言评分结果大于0.8
    assert score > 0.8


# 使用pytest的参数化装饰器，定义多个参数化测试用例，验证评分器的字符串表示是否正确
@pytest.mark.parametrize(
    "scorer, expected_repr",
    [
        (
            get_scorer("accuracy"),
            "make_scorer(accuracy_score, response_method='predict')",
        ),
        (
            get_scorer("neg_log_loss"),
            (
                "make_scorer(log_loss, greater_is_better=False,"
                " response_method='predict_proba')"
            ),
        ),
        (
            get_scorer("roc_auc"),
            (
                "make_scorer(roc_auc_score, response_method="
                "('decision_function', 'predict_proba'))"
            ),
        ),
        (
            make_scorer(fbeta_score, beta=2),
            "make_scorer(fbeta_score, response_method='predict', beta=2)",
        ),
    ],
)
# 定义测试函数，检查评分器的字符串表示是否与预期一致
def test_make_scorer_repr(scorer, expected_repr):
    """Check the representation of the scorer."""
    assert repr(scorer) == expected_repr


# 使用pytest的参数化装饰器，定义多个参数化测试用例，验证make_scorer函数是否正确抛出错误
@pytest.mark.filterwarnings("ignore:.*needs_proba.*:FutureWarning")
@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        # 如果设置了needs_*，则不能同时设置response_method
        (
            {"response_method": "predict_proba", "needs_proba": True},
            ValueError,
            "You cannot set both `response_method`",
        ),
        (
            {"response_method": "predict_proba", "needs_threshold": True},
            ValueError,
            "You cannot set both `response_method`",
        ),
        # 不能同时设置needs_proba和needs_threshold
        (
            {"needs_proba": True, "needs_threshold": True},
            ValueError,
            "You cannot set both `needs_proba` and `needs_threshold`",
        ),
    ],
)
# 定义测试函数，检查make_scorer函数是否正确抛出预期的错误
def test_make_scorer_error(params, err_type, err_msg):
    """Check that `make_scorer` raises errors if the parameter used."""
    with pytest.raises(err_type, match=err_msg):
        make_scorer(lambda y_true, y_pred: 1, **params)


# TODO(1.6): remove the following test
# 这是一个待移除的测试用例，预计在1.6版本移除之后进行重构
@pytest.mark.parametrize(
    "deprecated_params, new_params, warn_msg",
    # 创建一个包含多个元组的列表，每个元组包含两个字典和一个字符串，描述了参数配置和相关提示信息
    [
        (
            # 第一个字典，包含键 "needs_proba" 为 True
            {"needs_proba": True},
            # 第二个字典，包含键 "response_method" 为 "predict_proba"
            {"response_method": "predict_proba"},
            # 提示信息，说明 "needs_threshold" 和 "needs_proba" 参数已被弃用
            "The `needs_threshold` and `needs_proba` parameter are deprecated"
        ),
        (
            # 第一个字典，包含键 "needs_proba" 为 True，键 "needs_threshold" 为 False
            {"needs_proba": True, "needs_threshold": False},
            # 第二个字典，包含键 "response_method" 为 "predict_proba"
            {"response_method": "predict_proba"},
            # 提示信息，说明 "needs_threshold" 和 "needs_proba" 参数已被弃用
            "The `needs_threshold` and `needs_proba` parameter are deprecated"
        ),
        (
            # 第一个字典，包含键 "needs_threshold" 为 True
            {"needs_threshold": True},
            # 第二个字典，包含键 "response_method" 为元组 ("decision_function", "predict_proba")
            {"response_method": ("decision_function", "predict_proba")},
            # 提示信息，说明 "needs_threshold" 和 "needs_proba" 参数已被弃用
            "The `needs_threshold` and `needs_proba` parameter are deprecated"
        ),
        (
            # 第一个字典，包含键 "needs_threshold" 为 True，键 "needs_proba" 为 False
            {"needs_threshold": True, "needs_proba": False},
            # 第二个字典，包含键 "response_method" 为元组 ("decision_function", "predict_proba")
            {"response_method": ("decision_function", "predict_proba")},
            # 提示信息，说明 "needs_threshold" 和 "needs_proba" 参数已被弃用
            "The `needs_threshold` and `needs_proba` parameter are deprecated"
        ),
        (
            # 第一个字典，包含键 "needs_threshold" 为 False，键 "needs_proba" 为 False
            {"needs_threshold": False, "needs_proba": False},
            # 第二个字典，包含键 "response_method" 为 "predict"
            {"response_method": "predict"},
            # 提示信息，说明 "needs_threshold" 和 "needs_proba" 参数已被弃用
            "The `needs_threshold` and `needs_proba` parameter are deprecated"
        ),
    ],
def test_make_scorer_deprecation(deprecated_params, new_params, warn_msg):
    """Check that we raise a deprecation warning when using `needs_proba` or
    `needs_threshold`."""
    # 生成一个包含150个样本和10个特征的分类数据集
    X, y = make_classification(n_samples=150, n_features=10, random_state=0)
    # 使用逻辑回归分类器拟合数据集
    classifier = LogisticRegression().fit(X, y)

    # 检查 `needs_proba` 的弃用警告
    with pytest.warns(FutureWarning, match=warn_msg):
        # 使用过时参数创建 ROC AUC 评分器
        deprecated_roc_auc_scorer = make_scorer(roc_auc_score, **deprecated_params)
    # 使用新参数创建 ROC AUC 评分器
    roc_auc_scorer = make_scorer(roc_auc_score, **new_params)

    # 断言使用过时评分器和新评分器得到的结果近似相等
    assert deprecated_roc_auc_scorer(classifier, X, y) == pytest.approx(
        roc_auc_scorer(classifier, X, y)
    )


@pytest.mark.parametrize("pass_estimator", [True, False])
def test_get_scorer_multimetric(pass_estimator):
    """Check that check_scoring is compatible with multi-metric configurations."""
    # 生成一个包含150个样本和10个特征的分类数据集
    X, y = make_classification(n_samples=150, n_features=10, random_state=0)
    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 创建逻辑回归分类器
    clf = LogisticRegression(random_state=0)

    if pass_estimator:
        # 如果传递了估计器，则使用原始的 check_scoring 函数
        check_scoring_ = check_scoring
    else:
        # 否则使用 check_scoring 函数的偏函数形式，传递逻辑回归分类器 clf
        check_scoring_ = partial(check_scoring, clf)

    # 使用训练集拟合分类器
    clf.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = clf.predict(X_test)
    # 对测试集进行概率预测
    y_proba = clf.predict_proba(X_test)

    # 预期的结果字典，包含不同评分指标的值
    expected_results = {
        "r2": r2_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba[:, 1]),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    # 针对集合、列表和元组等不同容器类型进行遍历
    for container in [set, list, tuple]:
        # 使用不同容器类型创建评分器
        scoring = check_scoring_(scoring=container(["r2", "roc_auc", "accuracy"]))
        # 计算评分结果
        result = scoring(clf, X_test, y_test)

        # 断言评分结果的键与预期结果的键相同
        assert result.keys() == expected_results.keys()
        # 断言每个评分结果近似等于预期结果
        for name in result:
            assert result[name] == pytest.approx(expected_results[name])

    # 定义一个双倍精度的自定义评分函数
    def double_accuracy(y_true, y_pred):
        return 2 * accuracy_score(y_true, y_pred)

    # 创建一个使用 `predict` 方法响应的自定义评分器
    custom_scorer = make_scorer(double_accuracy, response_method="predict")

    # 使用不同名称的字典评分器
    dict_scoring = check_scoring_(
        scoring={
            "my_r2": "r2",
            "my_roc_auc": "roc_auc",
            "double_accuracy": custom_scorer,
        }
    )
    # 计算字典评分结果
    dict_result = dict_scoring(clf, X_test, y_test)
    # 断言字典评分结果包含3个键
    assert len(dict_result) == 3
    # 断言字典评分结果中每个键的值近似等于预期结果中对应的值
    assert dict_result["my_r2"] == pytest.approx(expected_results["r2"])
    assert dict_result["my_roc_auc"] == pytest.approx(expected_results["roc_auc"])
    assert dict_result["double_accuracy"] == pytest.approx(
        2 * expected_results["accuracy"]
    )


def test_multimetric_scorer_repr():
    """Check repr for multimetric scorer"""
    # 创建一个包含 "accuracy" 和 "r2" 评分指标的多指标评分器
    multi_metric_scorer = check_scoring(scoring=["accuracy", "r2"])

    # 断言多指标评分器的字符串表示与预期相符
    assert str(multi_metric_scorer) == 'MultiMetricScorer("accuracy", "r2")'


def test_check_scoring_multimetric_raise_exc():
    """Test that check_scoring returns error code for a subset of scorers in
    multimetric scoring if raise_exc=False and raises otherwise."""
    # 定义一个函数 raising_scorer，用于测试评分机制的错误处理
    def raising_scorer(estimator, X, y):
        # 抛出 ValueError 异常，提示特定错误信息
        raise ValueError("That doesn't work.")

    # 使用 make_classification 生成数据集，包括特征 X 和目标 y
    X, y = make_classification(n_samples=150, n_features=10, random_state=0)
    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 使用 LogisticRegression 拟合分类器 clf
    clf = LogisticRegression().fit(X_train, y_train)

    # 定义一个评分字典 scoring，包含两个评分指标
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "raising_scorer": raising_scorer,
    }
    # 调用 check_scoring 函数，返回一个评分器 scoring_call
    scoring_call = check_scoring(estimator=clf, scoring=scoring, raise_exc=False)
    # 使用 scoring_call 对 clf 在测试集上进行评分，存储评分结果到 scores
    scores = scoring_call(clf, X_test, y_test)
    # 断言评分结果中包含字符串 "That doesn't work."
    assert "That doesn't work." in scores["raising_scorer"]

    # 应该触发一个错误
    # 再次调用 check_scoring 函数，但设置 raise_exc=True，期望引发异常
    scoring_call = check_scoring(estimator=clf, scoring=scoring, raise_exc=True)
    # 定义错误信息字符串
    err_msg = "That doesn't work."
    # 使用 pytest 的 raises 方法断言调用 scoring_call 时会引发 ValueError 异常，并匹配错误信息 err_msg
    with pytest.raises(ValueError, match=err_msg):
        scores = scoring_call(clf, X_test, y_test)
# 使用 pytest 的装饰器标记此函数为参数化测试，参数为 enable_metadata_routing，分别为 True 和 False
@pytest.mark.parametrize("enable_metadata_routing", [True, False])
# 定义测试函数，测试多指标评分器在启用和禁用元数据路由时的工作情况，当没有实际元数据传递时。
# 这是一个非回归测试，用于检查 https://github.com/scikit-learn/scikit-learn/issues/28256 的问题是否修复。
def test_metadata_routing_multimetric_metadata_routing(enable_metadata_routing):
    # 使用 make_classification 创建一个样本数为 50，特征数为 10 的分类数据集，随机种子为 0
    X, y = make_classification(n_samples=50, n_features=10, random_state=0)
    # 实例化一个自定义的 EstimatorWithFitAndPredict 类并对数据集 X, y 进行拟合
    estimator = EstimatorWithFitAndPredict().fit(X, y)

    # 创建一个 _MultimetricScorer 对象，包含一个名为 "acc" 的评分器，使用 accuracy 评分器
    multimetric_scorer = _MultimetricScorer(scorers={"acc": get_scorer("accuracy")})
    
    # 使用 config_context 上下文管理器，根据 enable_metadata_routing 的值启用或禁用元数据路由
    with config_context(enable_metadata_routing=enable_metadata_routing):
        # 调用 multimetric_scorer 对象，评估 estimator 在数据集 X, y 上的性能
        multimetric_scorer(estimator, X, y)
```