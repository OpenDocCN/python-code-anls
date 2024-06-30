# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\tests\test_weight_boosting.py`

```
# 导入所需的库和模块
import re

import numpy as np
import pytest

from sklearn import datasets  # 导入 sklearn 中的 datasets 模块
from sklearn.base import BaseEstimator, clone  # 导入基础模型和克隆函数
from sklearn.dummy import DummyClassifier, DummyRegressor  # 导入虚拟分类器和回归器
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor  # 导入 AdaBoost 相关模型
from sklearn.ensemble._weight_boosting import _samme_proba  # 导入 AdaBoost 使用的辅助函数
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.model_selection import GridSearchCV, train_test_split  # 导入网格搜索和数据集划分函数
from sklearn.svm import SVC, SVR  # 导入支持向量分类器和回归器
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入决策树分类器和回归器
from sklearn.utils import shuffle  # 导入数据洗牌函数
from sklearn.utils._mocking import NoSampleWeightWrapper  # 导入样本权重模拟包装器
from sklearn.utils._testing import (  # 导入测试函数
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
)
from sklearn.utils.fixes import (  # 导入修复函数
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
)

# 共用的随机状态
rng = np.random.RandomState(0)

# 简单的样本数据
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y_class = ["foo", "foo", "foo", 1, 1, 1]  # 测试字符串类标签
y_regr = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
y_t_class = ["foo", 1, 1]
y_t_regr = [-1, 1, 1]

# 加载鸢尾花数据集并随机排列
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data, iris.target = shuffle(iris.data, iris.target, random_state=rng)

# 加载糖尿病数据集并随机排列
diabetes = datasets.load_diabetes()
diabetes.data, diabetes.target = shuffle(
    diabetes.data, diabetes.target, random_state=rng
)


def test_samme_proba():
    # 测试 `_samme_proba` 辅助函数。

    # 定义一些示例（不良）的 `predict_proba` 输出。
    probs = np.array(
        [[1, 1e-6, 0], [0.19, 0.6, 0.2], [-999, 0.51, 0.5], [1e-6, 1, 1e-9]]
    )
    probs /= np.abs(probs.sum(axis=1))[:, np.newaxis]

    # `_samme_proba` 调用 estimator.predict_proba。
    # 创建一个模拟对象以控制返回值。
    class MockEstimator:
        def predict_proba(self, X):
            assert_array_equal(X.shape, probs.shape)
            return probs

    mock = MockEstimator()

    # 调用 `_samme_proba` 函数
    samme_proba = _samme_proba(mock, 3, np.ones_like(probs))

    # 断言输出形状与预期一致
    assert_array_equal(samme_proba.shape, probs.shape)
    # 断言输出数组中的元素都是有限的
    assert np.isfinite(samme_proba).all()

    # 确保每个样本中最小的元素正确
    assert_array_equal(np.argmin(samme_proba, axis=1), [2, 0, 0, 2])
    # 确保每个样本中最大的元素正确
    assert_array_equal(np.argmax(samme_proba, axis=1), [0, 1, 1, 1])


def test_oneclass_adaboost_proba():
    # 测试单类标签输入的 `predict_proba` 的健壮性。
    # 以响应问题 #7501
    # https://github.com/scikit-learn/scikit-learn/issues/7501
    y_t = np.ones(len(X))
    # 使用 SAMME 算法训练 AdaBoost 分类器
    clf = AdaBoostClassifier(algorithm="SAMME").fit(X, y_t)
    # 断言预测概率与预期一致
    assert_array_almost_equal(clf.predict_proba(X), np.ones((len(X), 1)))
# TODO(1.6): remove "@pytest.mark.filterwarnings" as SAMME.R will be removed
# and substituted with the SAMME algorithm as a default; also re-write test to
# only consider "SAMME"
@pytest.mark.filterwarnings("ignore:The SAMME.R algorithm")
@pytest.mark.parametrize("algorithm", ["SAMME", "SAMME.R"])
def test_classification_toy(algorithm):
    # 在一个简单的数据集上检查分类效果。
    clf = AdaBoostClassifier(algorithm=algorithm, random_state=0)
    # 使用指定的算法进行模型训练
    clf.fit(X, y_class)
    # 断言预测结果与目标分类数据一致
    assert_array_equal(clf.predict(T), y_t_class)
    # 断言分类器的类别和目标类别一致
    assert_array_equal(np.unique(np.asarray(y_t_class)), clf.classes_)
    # 断言预测概率的形状正确
    assert clf.predict_proba(T).shape == (len(T), 2)
    # 断言决策函数的形状正确
    assert clf.decision_function(T).shape == (len(T),)


def test_regression_toy():
    # 在一个简单的数据集上检查回归效果。
    clf = AdaBoostRegressor(random_state=0)
    # 使用默认参数进行模型训练
    clf.fit(X, y_regr)
    # 断言回归预测结果与目标回归数据一致
    assert_array_equal(clf.predict(T), y_t_regr)


# TODO(1.6): remove "@pytest.mark.filterwarnings" as SAMME.R will be removed
# and substituted with the SAMME algorithm as a default; also re-write test to
# only consider "SAMME"
@pytest.mark.filterwarnings("ignore:The SAMME.R algorithm")
def test_iris():
    # 在鸢尾花数据集上检查算法的一致性。
    classes = np.unique(iris.target)
    clf_samme = prob_samme = None

    for alg in ["SAMME", "SAMME.R"]:
        clf = AdaBoostClassifier(algorithm=alg)
        # 使用当前算法进行模型训练
        clf.fit(iris.data, iris.target)

        # 断言分类器的类别与目标类别一致
        assert_array_equal(classes, clf.classes_)
        # 预测概率的形状与类别数一致
        proba = clf.predict_proba(iris.data)
        if alg == "SAMME":
            clf_samme = clf
            prob_samme = proba
        assert proba.shape[1] == len(classes)
        # 断言决策函数的形状与类别数一致
        assert clf.decision_function(iris.data).shape[1] == len(classes)

        # 检查模型得分是否大于0.9
        score = clf.score(iris.data, iris.target)
        assert score > 0.9, "Failed with algorithm %s and score = %f" % (alg, score)

        # 检查是否使用了多个估计器
        assert len(clf.estimators_) > 1
        # 检查不同的随机状态（参见问题＃7408）
        assert len(set(est.random_state for est in clf.estimators_)) == len(
            clf.estimators_
        )

    # 有些微妙的回归测试：在 ae7adc880d624615a34bafdb1d75ef67051b8200 之前，
    # 对于 SAMME，predict_proba 返回 SAMME.R 的值。
    clf_samme.algorithm = "SAMME.R"
    assert_array_less(0, np.abs(clf_samme.predict_proba(iris.data) - prob_samme))


@pytest.mark.parametrize("loss", ["linear", "square", "exponential"])
def test_diabetes(loss):
    # 在糖尿病数据集上检查算法的一致性。
    reg = AdaBoostRegressor(loss=loss, random_state=0)
    # 使用指定的损失函数进行模型训练
    reg.fit(diabetes.data, diabetes.target)
    # 检查模型得分是否大于0.55
    score = reg.score(diabetes.data, diabetes.target)
    assert score > 0.55

    # 检查是否使用了多个估计器
    assert len(reg.estimators_) > 1
    # 检查不同的随机状态（参见问题＃7408）
    assert len(set(est.random_state for est in reg.estimators_)) == len(reg.estimators_)
# 为了以默认的 SAMME 算法替代 SAMME.R 算法；同时重写测试以仅考虑 "SAMME"
@pytest.mark.filterwarnings("ignore:The SAMME.R algorithm")
@pytest.mark.parametrize("algorithm", ["SAMME", "SAMME.R"])
def test_staged_predict(algorithm):
    # 检查阶段预测。
    
    # 创建一个随机数生成器对象，用于生成伪随机数
    rng = np.random.RandomState(0)
    
    # 生成与 iris 数据集目标值形状相同的随机权重数组
    iris_weights = rng.randint(10, size=iris.target.shape)
    
    # 生成与 diabetes 数据集目标值形状相同的随机权重数组
    diabetes_weights = rng.randint(10, size=diabetes.target.shape)

    # 创建 AdaBoostClassifier 分类器对象，使用给定的算法和估计器数量
    clf = AdaBoostClassifier(algorithm=algorithm, n_estimators=10)
    
    # 使用 iris 数据集拟合分类器，同时考虑样本权重
    clf.fit(iris.data, iris.target, sample_weight=iris_weights)

    # 对 iris 数据集进行预测
    predictions = clf.predict(iris.data)
    
    # 获取分类器的阶段预测结果
    staged_predictions = [p for p in clf.staged_predict(iris.data)]
    
    # 对 iris 数据集进行概率预测
    proba = clf.predict_proba(iris.data)
    
    # 获取分类器的阶段概率预测结果
    staged_probas = [p for p in clf.staged_predict_proba(iris.data)]
    
    # 计算分类器在 iris 数据集上的得分，考虑样本权重
    score = clf.score(iris.data, iris.target, sample_weight=iris_weights)
    
    # 获取分类器的阶段得分结果
    staged_scores = [
        s for s in clf.staged_score(iris.data, iris.target, sample_weight=iris_weights)
    ]

    # 断言：阶段预测结果的长度应为 10
    assert len(staged_predictions) == 10
    # 断言：预测结果应与最后一个阶段预测结果几乎相等
    assert_array_almost_equal(predictions, staged_predictions[-1])
    # 断言：阶段概率预测结果的长度应为 10
    assert len(staged_probas) == 10
    # 断言：概率预测结果应与最后一个阶段概率预测结果几乎相等
    assert_array_almost_equal(proba, staged_probas[-1])
    # 断言：阶段得分结果的长度应为 10
    assert len(staged_scores) == 10
    # 断言：得分结果应与最后一个阶段得分结果几乎相等
    assert_array_almost_equal(score, staged_scores[-1])

    # AdaBoost 回归
    
    # 创建 AdaBoostRegressor 回归器对象，使用给定的估计器数量和随机种子
    clf = AdaBoostRegressor(n_estimators=10, random_state=0)
    
    # 使用 diabetes 数据集拟合回归器，同时考虑样本权重
    clf.fit(diabetes.data, diabetes.target, sample_weight=diabetes_weights)

    # 对 diabetes 数据集进行预测
    predictions = clf.predict(diabetes.data)
    
    # 获取回归器的阶段预测结果
    staged_predictions = [p for p in clf.staged_predict(diabetes.data)]
    
    # 计算回归器在 diabetes 数据集上的得分，考虑样本权重
    score = clf.score(diabetes.data, diabetes.target, sample_weight=diabetes_weights)
    
    # 获取回归器的阶段得分结果
    staged_scores = [
        s
        for s in clf.staged_score(
            diabetes.data, diabetes.target, sample_weight=diabetes_weights
        )
    ]

    # 断言：阶段预测结果的长度应为 10
    assert len(staged_predictions) == 10
    # 断言：预测结果应与最后一个阶段预测结果几乎相等
    assert_array_almost_equal(predictions, staged_predictions[-1])
    # 断言：阶段得分结果的长度应为 10
    assert len(staged_scores) == 10
    # 断言：得分结果应与最后一个阶段得分结果几乎相等
    assert_array_almost_equal(score, staged_scores[-1])


def test_gridsearch():
    # 检查是否可以对基础树进行网格搜索。
    
    # AdaBoost 分类
    
    # 创建 AdaBoostClassifier 分类器对象，使用决策树分类器作为估计器
    boost = AdaBoostClassifier(estimator=DecisionTreeClassifier())
    
    # 定义网格搜索的参数空间
    parameters = {
        "n_estimators": (1, 2),
        "estimator__max_depth": (1, 2),
        "algorithm": ("SAMME", "SAMME.R"),
    }
    
    # 创建 GridSearchCV 对象，使用 AdaBoostClassifier 和参数空间
    clf = GridSearchCV(boost, parameters)
    
    # 使用 iris 数据集进行网格搜索
    clf.fit(iris.data, iris.target)

    # AdaBoost 回归
    
    # 创建 AdaBoostRegressor 回归器对象，使用决策树回归器作为估计器和给定的随机种子
    boost = AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=0)
    
    # 定义回归器的参数空间
    parameters = {"n_estimators": (1, 2), "estimator__max_depth": (1, 2)}
    
    # 创建 GridSearchCV 对象，使用 AdaBoostRegressor 和参数空间
    clf = GridSearchCV(boost, parameters)
    
    # 使用 diabetes 数据集进行网格搜索
    clf.fit(diabetes.data, diabetes.target)


# TODO(1.6): 移除 "@pytest.mark.filterwarnings"，因为 SAMME.R 将被移除，
# 并以 SAMME 算法作为默认；同时重写测试以仅考虑 "SAMME"
@pytest.mark.filterwarnings("ignore:The SAMME.R algorithm")
def test_pickle():
    # 检查是否可以进行 pickle 操作。
    import pickle
    # Adaboost classifier
    # 对于分类器，使用两种不同的算法：SAMME 和 SAMME.R
    for alg in ["SAMME", "SAMME.R"]:
        # 创建一个 Adaboost 分类器对象，指定算法
        obj = AdaBoostClassifier(algorithm=alg)
        # 使用 iris 数据集训练分类器
        obj.fit(iris.data, iris.target)
        # 计算分类器在训练数据上的准确率
        score = obj.score(iris.data, iris.target)
        # 将训练好的分类器对象序列化为字符串
        s = pickle.dumps(obj)

        # 从序列化的字符串中反序列化出一个新的对象
        obj2 = pickle.loads(s)
        # 断言反序列化出的对象类型与原对象类型相同
        assert type(obj2) == obj.__class__
        # 计算反序列化对象在训练数据上的准确率
        score2 = obj2.score(iris.data, iris.target)
        # 断言原对象和反序列化对象在训练数据上的准确率相同
        assert score == score2

    # Adaboost regressor
    # 创建一个 Adaboost 回归器对象，使用随机种子 0
    obj = AdaBoostRegressor(random_state=0)
    # 使用 diabetes 数据集训练回归器
    obj.fit(diabetes.data, diabetes.target)
    # 计算回归器在训练数据上的决定系数 R^2
    score = obj.score(diabetes.data, diabetes.target)
    # 将训练好的回归器对象序列化为字符串
    s = pickle.dumps(obj)

    # 从序列化的字符串中反序列化出一个新的对象
    obj2 = pickle.loads(s)
    # 断言反序列化出的对象类型与原对象类型相同
    assert type(obj2) == obj.__class__
    # 计算反序列化对象在训练数据上的决定系数 R^2
    score2 = obj2.score(diabetes.data, diabetes.target)
    # 断言原对象和反序列化对象在训练数据上的决定系数 R^2 相同
    assert score == score2
# TODO(1.6): remove "@pytest.mark.filterwarnings" as SAMME.R will be removed
# and substituted with the SAMME algorithm as a default; also re-write test to
# only consider "SAMME"
@pytest.mark.filterwarnings("ignore:The SAMME.R algorithm")
def test_importances():
    # 生成一个分类数据集，包括2000个样本，10个特征，3个信息特征，没有冗余特征和重复特征
    X, y = datasets.make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=1,
    )

    # 针对每种算法"SAMME"和"SAMME.R"分别进行测试
    for alg in ["SAMME", "SAMME.R"]:
        # 创建AdaBoost分类器对象，使用指定的算法
        clf = AdaBoostClassifier(algorithm=alg)

        # 使用数据集X, y来拟合分类器
        clf.fit(X, y)
        # 获取特征重要性
        importances = clf.feature_importances_

        # 断言特征重要性数组的长度为10
        assert importances.shape[0] == 10
        # 断言前3个特征的重要性大于或等于后面7个特征的重要性
        assert (importances[:3, np.newaxis] >= importances[3:]).all()


def test_adaboost_classifier_sample_weight_error():
    # 测试在错误的样本权重上是否能够正确引发异常
    clf = AdaBoostClassifier()
    # 期望的错误消息，使用正则表达式转义生成
    msg = re.escape("sample_weight.shape == (1,), expected (6,)")
    # 使用pytest断言，检查是否引发了预期的ValueError异常，异常消息与预期的错误消息匹配
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y_class, sample_weight=np.asarray([-1]))


def test_estimator():
    # 测试不同的估算器（estimators）
    from sklearn.ensemble import RandomForestClassifier

    # XXX doesn't work with y_class because RF doesn't support classes_
    # Shouldn't AdaBoost run a LabelBinarizer?
    # 创建AdaBoost分类器对象，基础估算器为RandomForestClassifier，使用SAMME算法
    clf = AdaBoostClassifier(RandomForestClassifier(), algorithm="SAMME")
    # 使用数据集X, y_regr来拟合分类器
    clf.fit(X, y_regr)

    # 创建AdaBoost分类器对象，基础估算器为SVC，使用SAMME算法
    clf = AdaBoostClassifier(SVC(), algorithm="SAMME")
    # 使用数据集X, y_class来拟合分类器
    clf.fit(X, y_class)

    from sklearn.ensemble import RandomForestRegressor

    # 创建AdaBoost回归器对象，基础估算器为RandomForestRegressor，使用随机状态0
    clf = AdaBoostRegressor(RandomForestRegressor(), random_state=0)
    # 使用数据集X, y_regr来拟合回归器
    clf.fit(X, y_regr)

    # 创建AdaBoost回归器对象，基础估算器为SVR，使用随机状态0
    clf = AdaBoostRegressor(SVR(), random_state=0)
    # 使用数据集X, y_regr来拟合回归器
    clf.fit(X, y_regr)

    # 检查空的离散集成在拟合时是否失败，但在预测时不会失败
    X_fail = [[1, 1], [1, 1], [1, 1], [1, 1]]
    y_fail = ["foo", "bar", 1, 2]
    # 创建AdaBoost分类器对象，基础估算器为SVC，使用SAMME算法
    clf = AdaBoostClassifier(SVC(), algorithm="SAMME")
    # 使用pytest断言，检查是否引发了预期的ValueError异常，异常消息包含"worse than random"
    with pytest.raises(ValueError, match="worse than random"):
        clf.fit(X_fail, y_fail)


def test_sample_weights_infinite():
    # 测试样本权重是否会引发无穷值警告
    msg = "Sample weights have reached infinite values"
    # 创建AdaBoost分类器对象，使用指定的参数设置
    clf = AdaBoostClassifier(n_estimators=30, learning_rate=23.0, algorithm="SAMME")
    # 使用pytest断言，检查是否引发了预期的UserWarning警告，警告消息包含msg变量定义的字符串
    with pytest.warns(UserWarning, match=msg):
        clf.fit(iris.data, iris.target)


@pytest.mark.parametrize(
    "sparse_container, expected_internal_type",
    # 使用zip函数对稀疏容器和期望的内部类型进行参数化测试
    zip(
        [
            *CSC_CONTAINERS,
            *CSR_CONTAINERS,
            *LIL_CONTAINERS,
            *COO_CONTAINERS,
            *DOK_CONTAINERS,
        ],
        CSC_CONTAINERS + 4 * CSR_CONTAINERS,
    ),
)
def test_sparse_classification(sparse_container, expected_internal_type):
    # 使用稀疏输入检查分类
    class CustomSVC(SVC):
        """继承自SVC的定制SVC变体，记录训练集的数据类型。"""
    
        def fit(self, X, y, sample_weight=None):
            """修改后的fit方法，记录训练数据类型以备后续验证。"""
            super().fit(X, y, sample_weight=sample_weight)
            self.data_type_ = type(X)
            return self
    
    # 使用make_multilabel_classification生成多标签分类数据集，包括15个样本，每个样本5个特征
    X, y = datasets.make_multilabel_classification(
        n_classes=1, n_samples=15, n_features=5, random_state=42
    )
    # 将y展平为1维数组
    y = np.ravel(y)
    
    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # 将训练集和测试集转换为稀疏格式
    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)
    
    # 使用稀疏格式训练AdaBoost分类器
    sparse_classifier = AdaBoostClassifier(
        estimator=CustomSVC(probability=True),
        random_state=1,
        algorithm="SAMME",
    ).fit(X_train_sparse, y_train)
    
    # 使用密集格式训练AdaBoost分类器
    dense_classifier = AdaBoostClassifier(
        estimator=CustomSVC(probability=True),
        random_state=1,
        algorithm="SAMME",
    ).fit(X_train, y_train)
    
    # 预测测试集数据
    sparse_clf_results = sparse_classifier.predict(X_test_sparse)
    dense_clf_results = dense_classifier.predict(X_test)
    assert_array_equal(sparse_clf_results, dense_clf_results)
    
    # decision_function
    sparse_clf_results = sparse_classifier.decision_function(X_test_sparse)
    dense_clf_results = dense_classifier.decision_function(X_test)
    assert_array_almost_equal(sparse_clf_results, dense_clf_results)
    
    # predict_log_proba
    sparse_clf_results = sparse_classifier.predict_log_proba(X_test_sparse)
    dense_clf_results = dense_classifier.predict_log_proba(X_test)
    assert_array_almost_equal(sparse_clf_results, dense_clf_results)
    
    # predict_proba
    sparse_clf_results = sparse_classifier.predict_proba(X_test_sparse)
    dense_clf_results = dense_classifier.predict_proba(X_test)
    assert_array_almost_equal(sparse_clf_results, dense_clf_results)
    
    # score
    sparse_clf_results = sparse_classifier.score(X_test_sparse, y_test)
    dense_clf_results = dense_classifier.score(X_test, y_test)
    assert_array_almost_equal(sparse_clf_results, dense_clf_results)
    
    # staged_decision_function
    sparse_clf_results = sparse_classifier.staged_decision_function(X_test_sparse)
    dense_clf_results = dense_classifier.staged_decision_function(X_test)
    for sparse_clf_res, dense_clf_res in zip(sparse_clf_results, dense_clf_results):
        assert_array_almost_equal(sparse_clf_res, dense_clf_res)
    
    # staged_predict
    sparse_clf_results = sparse_classifier.staged_predict(X_test_sparse)
    dense_clf_results = dense_classifier.staged_predict(X_test)
    for sparse_clf_res, dense_clf_res in zip(sparse_clf_results, dense_clf_results):
        assert_array_equal(sparse_clf_res, dense_clf_res)
    
    # staged_predict_proba
    sparse_clf_results = sparse_classifier.staged_predict_proba(X_test_sparse)
    # 使用密集分类器对测试集进行逐步预测概率，并返回迭代器
    dense_clf_results = dense_classifier.staged_predict_proba(X_test)
    # 遍历稀疏分类器和密集分类器的结果，确保它们几乎相等
    for sparse_clf_res, dense_clf_res in zip(sparse_clf_results, dense_clf_results):
        assert_array_almost_equal(sparse_clf_res, dense_clf_res)

    # staged_score
    # 使用稀疏分类器对稀疏测试集和标签进行逐步评分，并返回迭代器
    sparse_clf_results = sparse_classifier.staged_score(X_test_sparse, y_test)
    # 使用密集分类器对测试集和标签进行逐步评分，并返回迭代器
    dense_clf_results = dense_classifier.staged_score(X_test, y_test)
    # 遍历稀疏分类器和密集分类器的评分结果，确保它们完全相等
    for sparse_clf_res, dense_clf_res in zip(sparse_clf_results, dense_clf_results):
        assert_array_equal(sparse_clf_res, dense_clf_res)

    # 验证在训练过程中数据的稀疏性得以保持
    # 获取稀疏分类器的所有基分类器的数据类型
    types = [i.data_type_ for i in sparse_classifier.estimators_]

    # 断言所有基分类器的数据类型与预期的内部类型相同
    assert all([t == expected_internal_type for t in types])
# 使用 pytest.mark.parametrize 装饰器为 test_sparse_regression 函数参数化测试用例，依次传入稀疏容器类型和预期的内部类型
@pytest.mark.parametrize(
    "sparse_container, expected_internal_type",
    zip(
        [
            *CSC_CONTAINERS,
            *CSR_CONTAINERS,
            *LIL_CONTAINERS,
            *COO_CONTAINERS,
            *DOK_CONTAINERS,
        ],
        CSC_CONTAINERS + 4 * CSR_CONTAINERS,
    ),
)
def test_sparse_regression(sparse_container, expected_internal_type):
    # Check regression with sparse input.

    # 定义一个自定义的 SVR 变体，记录训练集的特性
    class CustomSVR(SVR):
        """SVR variant that records the nature of the training set."""

        # 改进的 fit 方法，记录数据类型以便后续验证
        def fit(self, X, y, sample_weight=None):
            """Modification on fit caries data type for later verification."""
            super().fit(X, y, sample_weight=sample_weight)
            self.data_type_ = type(X)
            return self

    # 生成一个具有特定属性的回归数据集
    X, y = datasets.make_regression(
        n_samples=15, n_features=50, n_targets=1, random_state=42
    )

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 将训练集和测试集转换为稀疏格式
    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)

    # 在稀疏格式上训练 AdaBoost 回归器
    sparse_regressor = AdaBoostRegressor(estimator=CustomSVR(), random_state=1).fit(
        X_train_sparse, y_train
    )

    # 在稠密格式上训练 AdaBoost 回归器
    dense_regressor = AdaBoostRegressor(estimator=CustomSVR(), random_state=1).fit(
        X_train, y_train
    )

    # 预测稀疏格式的结果
    sparse_regr_results = sparse_regressor.predict(X_test_sparse)
    # 预测稠密格式的结果
    dense_regr_results = dense_regressor.predict(X_test)
    # 断言两种格式的预测结果近似相等
    assert_array_almost_equal(sparse_regr_results, dense_regr_results)

    # staged_predict 方法的预测
    sparse_regr_results = sparse_regressor.staged_predict(X_test_sparse)
    dense_regr_results = dense_regressor.staged_predict(X_test)
    # 逐个断言 staged_predict 结果的近似相等性
    for sparse_regr_res, dense_regr_res in zip(sparse_regr_results, dense_regr_results):
        assert_array_almost_equal(sparse_regr_res, dense_regr_res)

    # 提取每个 AdaBoost 回归器的内部数据类型
    types = [i.data_type_ for i in sparse_regressor.estimators_]

    # 断言所有回归器的内部数据类型与预期的内部类型相同
    assert all([t == expected_internal_type for t in types])


# 测试 AdaBoostRegressor 在不使用样本权重的情况下是否正常工作
def test_sample_weight_adaboost_regressor():
    """
    AdaBoostRegressor should work without sample_weights in the base estimator
    The random weighted sampling is done internally in the _boost method in
    AdaBoostRegressor.
    """

    # 定义一个简单的基础估计器类 DummyEstimator
    class DummyEstimator(BaseEstimator):
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(X.shape[0])

    # 创建 AdaBoostRegressor 对象并拟合数据
    boost = AdaBoostRegressor(DummyEstimator(), n_estimators=3)
    boost.fit(X, y_regr)
    # 断言估计器权重的长度与估计器错误的长度相等
    assert len(boost.estimator_weights_) == len(boost.estimator_errors_)


# 测试 AdaBoostClassifier 是否能处理多维度数据矩阵
def test_multidimensional_X():
    """
    Check that the AdaBoost estimators can work with n-dimensional
    data matrix
    """
    # 设置随机种子
    rng = np.random.RandomState(0)

    # 生成一个 3 维数据矩阵 X
    X = rng.randn(51, 3, 3)
    # 随机选择类别标签 yc
    yc = rng.choice([0, 1], 51)
    # 随机生成回归目标值 yr
    yr = rng.randn(51)

    # 使用 DummyClassifier 作为基础分类器，创建 AdaBoostClassifier 对象
    boost = AdaBoostClassifier(
        DummyClassifier(strategy="most_frequent"), algorithm="SAMME"
    )
    # 拟合数据
    boost.fit(X, yc)
    # 进行预测
    boost.predict(X)
    # 进行概率预测
    boost.predict_proba(X)
    # 使用AdaBoostRegressor创建一个AdaBoost回归器对象，使用DummyRegressor作为基础估计器
    boost = AdaBoostRegressor(DummyRegressor())
    # 使用训练数据X和对应的目标值yr来拟合（训练）AdaBoost回归器
    boost.fit(X, yr)
    # 对输入数据X进行预测，并返回预测结果
    boost.predict(X)
# TODO(1.6): remove "@pytest.mark.filterwarnings" as SAMME.R will be removed
# and substituted with the SAMME algorithm as a default; also re-write test to
# only consider "SAMME"
# 使用pytest的mark功能，忽略指定警告，以便测试仅考虑"SAMME"算法而非"SAMME.R"
@pytest.mark.filterwarnings("ignore:The SAMME.R algorithm")
# 参数化测试函数，使用两种算法："SAMME"和"SAMME.R"
@pytest.mark.parametrize("algorithm", ["SAMME", "SAMME.R"])
def test_adaboostclassifier_without_sample_weight(algorithm):
    # 加载鸢尾花数据集
    X, y = iris.data, iris.target
    # 使用DummyClassifier包装器创建无样本权重的估计器
    estimator = NoSampleWeightWrapper(DummyClassifier())
    # 创建AdaBoostClassifier分类器，指定估计器和算法
    clf = AdaBoostClassifier(estimator=estimator, algorithm=algorithm)
    # 错误消息模板，用于验证是否支持样本权重
    err_msg = "{} doesn't support sample_weight".format(estimator.__class__.__name__)
    # 使用pytest的raises断言，检查是否引发了指定错误
    with pytest.raises(ValueError, match=err_msg):
        # 训练分类器
        clf.fit(X, y)


def test_adaboostregressor_sample_weight():
    # 检查是否给定权重会影响计算的错误
    rng = np.random.RandomState(42)
    # 生成输入特征
    X = np.linspace(0, 100, num=1000)
    # 生成输出标签，并引入随机噪声
    y = (0.8 * X + 0.2) + (rng.rand(X.shape[0]) * 0.0001)
    X = X.reshape(-1, 1)

    # 添加一个任意的异常值
    X[-1] *= 10
    y[-1] = 10000

    # 设置随机状态以确保引导样本将使用异常值
    regr_no_outlier = AdaBoostRegressor(
        estimator=LinearRegression(), n_estimators=1, random_state=0
    )
    # 克隆不含异常值的回归模型
    regr_with_weight = clone(regr_no_outlier)
    # 克隆含异常值的回归模型
    regr_with_outlier = clone(regr_no_outlier)

    # 训练3个模型：
    # - 一个包含异常值的模型
    # - 一个不包含异常值的模型
    # - 一个包含异常值但样本权重为零的模型
    regr_with_outlier.fit(X, y)
    regr_no_outlier.fit(X[:-1], y[:-1])
    sample_weight = np.ones_like(y)
    sample_weight[-1] = 0
    regr_with_weight.fit(X, y, sample_weight=sample_weight)

    # 计算各模型在不含异常值数据上的评分
    score_with_outlier = regr_with_outlier.score(X[:-1], y[:-1])
    score_no_outlier = regr_no_outlier.score(X[:-1], y[:-1])
    score_with_weight = regr_with_weight.score(X[:-1], y[:-1])

    # 使用断言验证各模型的评分关系
    assert score_with_outlier < score_no_outlier
    assert score_with_outlier < score_with_weight
    assert score_no_outlier == pytest.approx(score_with_weight)


# TODO(1.6): remove "@pytest.mark.filterwarnings" as SAMME.R will be removed
# and substituted with the SAMME algorithm as a default; also re-write test to
# only consider "SAMME"
# 使用pytest的mark功能，忽略指定警告，以便测试仅考虑"SAMME"算法而非"SAMME.R"
@pytest.mark.filterwarnings("ignore:The SAMME.R algorithm")
# 参数化测试函数，使用两种算法："SAMME"和"SAMME.R"
@pytest.mark.parametrize("algorithm", ["SAMME", "SAMME.R"])
def test_adaboost_consistent_predict(algorithm):
    # 检查predict_proba和predict是否给出一致的结果
    # 回归测试用例，验证：
    # https://github.com/scikit-learn/scikit-learn/issues/14084
    # 加载手写数字数据集并分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        *datasets.load_digits(return_X_y=True), random_state=42
    )
    # 创建AdaBoostClassifier分类器，指定算法和随机状态
    model = AdaBoostClassifier(algorithm=algorithm, random_state=42)
    # 训练分类器
    model.fit(X_train, y_train)

    # 使用断言验证预测概率和预测结果的一致性
    assert_array_equal(
        np.argmax(model.predict_proba(X_test), axis=1), model.predict(X_test)
    )


@pytest.mark.parametrize(
    "model, X, y",
    [
        # 使用 AdaBoostClassifier 模型进行分类，训练数据为 iris 数据集的特征数据，标签为 iris 数据集的目标值
        (AdaBoostClassifier(), iris.data, iris.target),
        # 使用 AdaBoostRegressor 模型进行回归，训练数据为 diabetes 数据集的特征数据，标签为 diabetes 数据集的目标值
        (AdaBoostRegressor(), diabetes.data, diabetes.target),
    ],
# 定义一个测试函数，用于检查 Adaboost 模型在使用负权重时是否能正确抛出 ValueError 异常
def test_adaboost_negative_weight_error(model, X, y):
    # 创建一个与 y 相同形状的权重数组，初始值为 1
    sample_weight = np.ones_like(y)
    # 将最后一个样本的权重设置为 -10，用于测试负值权重的情况
    sample_weight[-1] = -10

    # 定义错误消息，用于匹配抛出的 ValueError 异常信息
    err_msg = "Negative values in data passed to `sample_weight`"
    # 使用 pytest 检查是否抛出指定异常，并匹配错误消息
    with pytest.raises(ValueError, match=err_msg):
        model.fit(X, y, sample_weight=sample_weight)


# 定义一个测试函数，用于检查 Adaboost 模型在使用极小权重时是否产生 NaN 的特征重要性
def test_adaboost_numerically_stable_feature_importance_with_small_weights():
    """Check that we don't create NaN feature importance with numerically
    instable inputs.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20320
    """
    # 使用随机数生成器创建数据集 X 和标签 y
    rng = np.random.RandomState(42)
    X = rng.normal(size=(1000, 10))
    y = rng.choice([0, 1], size=1000)
    # 创建一个极小的权重数组，避免数值不稳定性
    sample_weight = np.ones_like(y) * 1e-263
    # 创建一个决策树分类器
    tree = DecisionTreeClassifier(max_depth=10, random_state=12)
    # 创建 Adaboost 分类器，使用决策树分类器作为基分类器
    ada_model = AdaBoostClassifier(
        estimator=tree, n_estimators=20, algorithm="SAMME", random_state=12
    )
    # 使用极小权重训练 Adaboost 模型
    ada_model.fit(X, y, sample_weight=sample_weight)
    # 断言特征重要性数组中没有 NaN 值
    assert np.isnan(ada_model.feature_importances_).sum() == 0


# 根据版本规划的 TODO，此测试函数是为了检查 Adaboost 模型在不同算法下的决策函数对称性约束
@pytest.mark.filterwarnings("ignore:The SAMME.R algorithm")
@pytest.mark.parametrize("algorithm", ["SAMME", "SAMME.R"])
def test_adaboost_decision_function(algorithm, global_random_seed):
    """Check that the decision function respects the symmetric constraint for weak
    learners.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/26520
    """
    # 定义分类数
    n_classes = 3
    # 使用 make_classification 生成数据集 X 和标签 y
    X, y = datasets.make_classification(
        n_classes=n_classes, n_clusters_per_class=1, random_state=global_random_seed
    )
    # 创建 Adaboost 分类器，包含一个基分类器
    clf = AdaBoostClassifier(
        n_estimators=1, random_state=global_random_seed, algorithm=algorithm
    ).fit(X, y)

    # 获取决策函数的预测得分
    y_score = clf.decision_function(X)
    # 断言每个样本的得分和为 0，以一定的容差检查
    assert_allclose(y_score.sum(axis=1), 0, atol=1e-8)

    if algorithm == "SAMME":
        # 对于单个学习器，预期决策函数的取值在 {1, - 1 / (n_classes - 1)} 中
        assert set(np.unique(y_score)) == {1, -1 / (n_classes - 1)}

    # 对 staged_decision_function 也进行相同的检查，因为只有一个学习器
    for y_score in clf.staged_decision_function(X):
        assert_allclose(y_score.sum(axis=1), 0, atol=1e-8)

        if algorithm == "SAMME":
            # 对于单个学习器，预期决策函数的取值在 {1, - 1 / (n_classes - 1)} 中
            assert set(np.unique(y_score)) == {1, -1 / (n_classes - 1)}

    # 增加基分类器数量为 5，并重新训练 Adaboost 模型
    clf.set_params(n_estimators=5).fit(X, y)

    y_score = clf.decision_function(X)
    # 再次检查每个样本的得分和为 0
    assert_allclose(y_score.sum(axis=1), 0, atol=1e-8)

    for y_score in clf.staged_decision_function(X):
        # 对 staged_decision_function 再次检查每个样本的得分和为 0
        assert_allclose(y_score.sum(axis=1), 0, atol=1e-8)


# 根据版本规划的 TODO，此测试函数是为了检查废弃的 SAMME.R 算法
def test_deprecated_samme_r_algorithm():
    # 创建一个 Adaboost 分类器，只包含一个基分类器
    adaboost_clf = AdaBoostClassifier(n_estimators=1)
    # 使用 pytest 的 warns 上下文管理器捕获特定的 FutureWarning 警告
    with pytest.warns(
        FutureWarning,  # 指定警告类型为 FutureWarning
        match=re.escape("The SAMME.R algorithm (the default) is deprecated"),  # 指定匹配的警告消息内容，使用 re.escape 来确保特殊字符被正确转义
    ):
        # 使用 AdaBoost 分类器（adaboost_clf）拟合训练数据 X 和标签 y_class
        adaboost_clf.fit(X, y_class)
```