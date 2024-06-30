# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\tests\test_voting.py`

```
# 导入所需的库和模块
import re  # 导入 re 模块用于正则表达式操作
import numpy as np  # 导入 numpy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from sklearn import datasets  # 导入 sklearn 中的 datasets 模块
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # 导入 sklearn 中的基础类和函数
from sklearn.datasets import make_multilabel_classification  # 导入 sklearn 中生成多标签分类数据的函数
from sklearn.dummy import DummyRegressor  # 导入 sklearn 中的虚拟回归器
from sklearn.ensemble import (  # 导入 sklearn 中的集成模型
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.exceptions import NotFittedError  # 导入 sklearn 中的未拟合错误类
from sklearn.linear_model import LinearRegression, LogisticRegression  # 导入 sklearn 中的线性回归和逻辑回归模型
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split  # 导入 sklearn 中的模型选择和交叉验证函数
from sklearn.multiclass import OneVsRestClassifier  # 导入 sklearn 中的一对多分类器
from sklearn.naive_bayes import GaussianNB  # 导入 sklearn 中的高斯朴素贝叶斯模型
from sklearn.neighbors import KNeighborsClassifier  # 导入 sklearn 中的K近邻分类器
from sklearn.preprocessing import StandardScaler  # 导入 sklearn 中的标准化预处理函数
from sklearn.svm import SVC  # 导入 sklearn 中的支持向量分类器
from sklearn.tests.metadata_routing_common import (  # 导入 sklearn 测试中的元数据公共函数
    ConsumingClassifier,
    ConsumingRegressor,
    _Registry,
    check_recorded_metadata,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入 sklearn 中的决策树分类器和回归器
from sklearn.utils._testing import (  # 导入 sklearn 测试工具函数
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)

# 加载示例数据集
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target  # 从 iris 数据集中选择部分特征作为 X，选择标签作为 y
# 使用标准化方法对 X 进行预处理，以解决逻辑回归的收敛警告
X_scaled = StandardScaler().fit_transform(X)

# 加载糖尿病数据集
X_r, y_r = datasets.load_diabetes(return_X_y=True)


@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            {"estimators": []},
            "Invalid 'estimators' attribute, 'estimators' should be a non-empty list",
        ),
        (
            {"estimators": [("lr", LogisticRegression())], "weights": [1, 2]},
            "Number of `estimators` and weights must be equal",
        ),
    ],
)
def test_voting_classifier_estimator_init(params, err_msg):
    # 创建 VotingClassifier 对象并使用 pytest 来断言是否会抛出 ValueError 异常
    ensemble = VotingClassifier(**params)
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)


def test_predictproba_hardvoting():
    # 创建 VotingClassifier 对象，设置 voting='hard'
    eclf = VotingClassifier(
        estimators=[("lr1", LogisticRegression()), ("lr2", LogisticRegression())],
        voting="hard",
    )

    inner_msg = "predict_proba is not available when voting='hard'"
    outer_msg = "'VotingClassifier' has no attribute 'predict_proba'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        eclf.predict_proba
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)

    # 验证 VotingClassifier 对象是否有 'predict_proba' 属性
    assert not hasattr(eclf, "predict_proba")
    eclf.fit(X_scaled, y)
    assert not hasattr(eclf, "predict_proba")


def test_notfitted():
    # 创建 VotingClassifier 和 VotingRegressor 对象
    eclf = VotingClassifier(
        estimators=[("lr1", LogisticRegression()), ("lr2", LogisticRegression())],
        voting="soft",
    )
    ereg = VotingRegressor([("dr", DummyRegressor())])
    # 定义错误消息模板，指示模型未拟合
    msg = (
        "This %s instance is not fitted yet. Call 'fit'"
        " with appropriate arguments before using this estimator."
    )
    # 使用 pytest 检查 VotingClassifier 是否抛出 NotFittedError 异常，并匹配特定错误消息
    with pytest.raises(NotFittedError, match=msg % "VotingClassifier"):
        # 调用 VotingClassifier 实例的 predict 方法，预期会抛出未拟合异常
        eclf.predict(X)
    # 同上，检查 VotingClassifier 是否抛出 NotFittedError 异常，并匹配特定错误消息
    with pytest.raises(NotFittedError, match=msg % "VotingClassifier"):
        # 调用 VotingClassifier 实例的 predict_proba 方法，预期会抛出未拟合异常
        eclf.predict_proba(X)
    # 同上，检查 VotingClassifier 是否抛出 NotFittedError 异常，并匹配特定错误消息
    with pytest.raises(NotFittedError, match=msg % "VotingClassifier"):
        # 调用 VotingClassifier 实例的 transform 方法，预期会抛出未拟合异常
        eclf.transform(X)
    # 同上，检查 VotingRegressor 是否抛出 NotFittedError 异常，并匹配特定错误消息
    with pytest.raises(NotFittedError, match=msg % "VotingRegressor"):
        # 调用 VotingRegressor 实例的 predict 方法，预期会抛出未拟合异常
        ereg.predict(X_r)
    # 同上，检查 VotingRegressor 是否抛出 NotFittedError 异常，并匹配特定错误消息
    with pytest.raises(NotFittedError, match=msg % "VotingRegressor"):
        # 调用 VotingRegressor 实例的 transform 方法，预期会抛出未拟合异常
        ereg.transform(X_r)
# 检查在 iris 数据集上使用多数标签分类的测试函数
def test_majority_label_iris(global_random_seed):
    """Check classification by majority label on dataset iris."""
    # 创建逻辑回归分类器，使用liblinear求解器，设置随机种子
    clf1 = LogisticRegression(solver="liblinear", random_state=global_random_seed)
    # 创建随机森林分类器，设置10棵树，设置随机种子
    clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed)
    # 创建高斯朴素贝叶斯分类器
    clf3 = GaussianNB()
    # 创建投票分类器，包含逻辑回归、随机森林和高斯朴素贝叶斯分类器，采用硬投票策略
    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="hard"
    )
    # 使用交叉验证计算分类器的准确率得分
    scores = cross_val_score(eclf, X, y, scoring="accuracy")

    # 断言分类器的平均准确率大于等于0.9
    assert scores.mean() >= 0.9


# 检查在平局情况下，投票分类器选择较小类标签的测试函数
def test_tie_situation():
    """Check voting classifier selects smaller class label in tie situation."""
    # 创建逻辑回归分类器，设置随机种子和liblinear求解器
    clf1 = LogisticRegression(random_state=123, solver="liblinear")
    # 创建随机森林分类器，设置随机种子
    clf2 = RandomForestClassifier(random_state=123)
    # 创建投票分类器，包含逻辑回归和随机森林分类器，采用硬投票策略
    eclf = VotingClassifier(estimators=[("lr", clf1), ("rf", clf2)], voting="hard")
    # 断言逻辑回归分类器在X数据集上对第73个样本的预测为2
    assert clf1.fit(X, y).predict(X)[73] == 2
    # 断言随机森林分类器在X数据集上对第73个样本的预测为1
    assert clf2.fit(X, y).predict(X)[73] == 1
    # 断言投票分类器在X数据集上对第73个样本的预测为1
    assert eclf.fit(X, y).predict(X)[73] == 1


# 检查在 iris 数据集上使用平均概率分类的测试函数
def test_weights_iris(global_random_seed):
    """Check classification by average probabilities on dataset iris."""
    # 创建逻辑回归分类器，设置随机种子
    clf1 = LogisticRegression(random_state=global_random_seed)
    # 创建随机森林分类器，设置10棵树，设置随机种子
    clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed)
    # 创建高斯朴素贝叶斯分类器
    clf3 = GaussianNB()
    # 创建投票分类器，包含逻辑回归、随机森林和高斯朴素贝叶斯分类器，采用软投票策略，并设置权重
    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="soft",
        weights=[1, 2, 10],
    )
    # 使用交叉验证计算分类器的准确率得分
    scores = cross_val_score(eclf, X_scaled, y, scoring="accuracy")
    # 断言分类器的平均准确率大于等于0.9
    assert scores.mean() >= 0.9


# 检查在糖尿病数据集上使用加权平均回归预测的测试函数
def test_weights_regressor():
    """Check weighted average regression prediction on diabetes dataset."""
    # 创建均值策略的虚拟回归器
    reg1 = DummyRegressor(strategy="mean")
    # 创建中位数策略的虚拟回归器
    reg2 = DummyRegressor(strategy="median")
    # 创建分位数策略为0.2的虚拟回归器
    reg3 = DummyRegressor(strategy="quantile", quantile=0.2)
    # 创建投票回归器，包含均值、中位数和分位数策略的虚拟回归器，并设置权重
    ereg = VotingRegressor(
        [("mean", reg1), ("median", reg2), ("quantile", reg3)], weights=[1, 2, 10]
    )

    # 将糖尿病数据集划分为训练集和测试集
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_r, y_r, test_size=0.25
    )

    # 分别对训练集进行拟合和预测
    reg1_pred = reg1.fit(X_r_train, y_r_train).predict(X_r_test)
    reg2_pred = reg2.fit(X_r_train, y_r_train).predict(X_r_test)
    reg3_pred = reg3.fit(X_r_train, y_r_train).predict(X_r_test)
    ereg_pred = ereg.fit(X_r_train, y_r_train).predict(X_r_test)

    # 计算加权平均预测值
    avg = np.average(
        np.asarray([reg1_pred, reg2_pred, reg3_pred]), axis=0, weights=[1, 2, 10]
    )
    # 断言投票回归器的预测结果与加权平均值接近
    assert_almost_equal(ereg_pred, avg, decimal=2)

    # 创建没有权重的投票回归器
    ereg_weights_none = VotingRegressor(
        [("mean", reg1), ("median", reg2), ("quantile", reg3)], weights=None
    )
    # 创建权重相等的投票回归器
    ereg_weights_equal = VotingRegressor(
        [("mean", reg1), ("median", reg2), ("quantile", reg3)], weights=[1, 1, 1]
    )
    # 分别对训练集进行拟合
    ereg_weights_none.fit(X_r_train, y_r_train)
    ereg_weights_equal.fit(X_r_train, y_r_train)
    # 分别预测测试集的结果
    ereg_none_pred = ereg_weights_none.predict(X_r_test)
    ereg_equal_pred = ereg_weights_equal.predict(X_r_test)
    # 断言没有权重和权重相等的预测结果近似
    assert_almost_equal(ereg_none_pred, ereg_equal_pred, decimal=2)


# 待完成：关于玩具问题的预测测试函数
def test_predict_on_toy_problem(global_random_seed):
    """Manually check predicted class labels for toy dataset."""
    # 创建逻辑回归分类器，使用全局随机种子作为随机状态
    clf1 = LogisticRegression(random_state=global_random_seed)
    # 创建随机森林分类器，包含10棵树，使用全局随机种子作为随机状态
    clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed)
    # 创建高斯朴素贝叶斯分类器
    clf3 = GaussianNB()

    # 创建特征矩阵 X
    X = np.array(
        [[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2], [2.1, 1.4], [3.1, 2.3]]
    )

    # 创建目标变量 y
    y = np.array([1, 1, 1, 2, 2, 2])

    # 验证逻辑回归分类器是否正确预测 X 的类标签，并与预期结果进行比较
    assert_array_equal(clf1.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])
    # 验证随机森林分类器是否正确预测 X 的类标签，并与预期结果进行比较
    assert_array_equal(clf2.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])
    # 验证高斯朴素贝叶斯分类器是否正确预测 X 的类标签，并与预期结果进行比较
    assert_array_equal(clf3.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])

    # 创建硬投票集成分类器，包含逻辑回归、随机森林和高斯朴素贝叶斯分类器，权重均为1
    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="hard",
        weights=[1, 1, 1],
    )
    # 验证硬投票集成分类器是否正确预测 X 的类标签，并与预期结果进行比较
    assert_array_equal(eclf.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])

    # 创建软投票集成分类器，包含逻辑回归、随机森林和高斯朴素贝叶斯分类器，权重均为1
    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="soft",
        weights=[1, 1, 1],
    )
    # 验证软投票集成分类器是否正确预测 X 的类标签，并与预期结果进行比较
    assert_array_equal(eclf.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])
# 定义一个测试函数，用于在简单数据集上计算预测概率
def test_predict_proba_on_toy_problem():
    # 创建逻辑回归、随机森林和高斯朴素贝叶斯分类器实例
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    # 定义一个包含四个样本的二维数组作为特征矩阵
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    # 定义一个包含四个样本的一维数组作为目标变量
    y = np.array([1, 1, 2, 2])

    # 预测结果示例数组 clf1_res, clf2_res, clf3_res
    clf1_res = np.array(
        [
            [0.59790391, 0.40209609],
            [0.57622162, 0.42377838],
            [0.50728456, 0.49271544],
            [0.40241774, 0.59758226],
        ]
    )

    clf2_res = np.array([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]])

    clf3_res = np.array(
        [[0.9985082, 0.0014918], [0.99845843, 0.00154157], [0.0, 1.0], [0.0, 1.0]]
    )

    # 计算加权平均值用于断言
    t00 = (2 * clf1_res[0][0] + clf2_res[0][0] + clf3_res[0][0]) / 4
    t11 = (2 * clf1_res[1][1] + clf2_res[1][1] + clf3_res[1][1]) / 4
    t21 = (2 * clf1_res[2][1] + clf2_res[2][1] + clf3_res[2][1]) / 4
    t31 = (2 * clf1_res[3][1] + clf2_res[3][1] + clf3_res[3][1]) / 4

    # 创建投票分类器对象，使用 soft 投票方式和特定权重
    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="soft",
        weights=[2, 1, 1],
    )
    # 对投票分类器进行拟合，并预测概率
    eclf_res = eclf.fit(X, y).predict_proba(X)

    # 断言结果
    assert_almost_equal(t00, eclf_res[0][0], decimal=1)
    assert_almost_equal(t11, eclf_res[1][1], decimal=1)
    assert_almost_equal(t21, eclf_res[2][1], decimal=1)
    assert_almost_equal(t31, eclf_res[3][1], decimal=1)

    # 检查当 voting='hard' 时，predict_proba 方法不可用的情况
    inner_msg = "predict_proba is not available when voting='hard'"
    outer_msg = "'VotingClassifier' has no attribute 'predict_proba'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        eclf = VotingClassifier(
            estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="hard"
        )
        # 对投票分类器进行拟合，并尝试调用 predict_proba 方法
        eclf.fit(X, y).predict_proba(X)

    # 断言捕获的异常信息和期望的内部消息
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)


# 定义一个测试函数，用于检查多标签分类时是否会引发错误
def test_multilabel():
    # 生成多标签分类的数据集
    X, y = make_multilabel_classification(
        n_classes=2, n_labels=1, allow_unlabeled=False, random_state=123
    )
    # 创建一个 OneVsRestClassifier 包装的支持向量分类器实例
    clf = OneVsRestClassifier(SVC(kernel="linear"))

    # 创建投票分类器对象，使用 hard 投票方式
    eclf = VotingClassifier(estimators=[("ovr", clf)], voting="hard")

    # 尝试对数据集进行拟合，检查是否引发 NotImplementedError
    try:
        eclf.fit(X, y)
    except NotImplementedError:
        return


# 定义一个测试函数，用于检查投票分类器是否支持网格搜索
def test_gridsearch():
    # 创建三个不同的分类器实例
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1, n_estimators=3)
    clf3 = GaussianNB()
    # 创建投票分类器对象，使用 soft 投票方式
    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft"
    )

    # 定义网格搜索的参数空间
    params = {
        "lr__C": [1.0, 100.0],
        "voting": ["soft", "hard"],
        "weights": [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]],
    }

    # 创建 GridSearchCV 对象，对投票分类器进行参数优化
    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=2)
    # 在数据集上进行网格搜索
    grid.fit(X_scaled, y)


# 定义一个测试函数，用于检查投票分类器在简单数据集上的并行拟合能力
def test_parallel_fit(global_random_seed):
    # 检查投票分类器在并行后端上的工作表现，但函数体内容缺失
    # 创建逻辑回归分类器，使用全局随机种子作为随机状态
    clf1 = LogisticRegression(random_state=global_random_seed)
    # 创建随机森林分类器，使用10棵树和全局随机种子作为随机状态
    clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed)
    # 创建高斯朴素贝叶斯分类器
    clf3 = GaussianNB()

    # 创建输入特征矩阵 X 和目标向量 y
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])

    # 创建软投票分类器 eclf1，包含逻辑回归、随机森林和高斯朴素贝叶斯分类器，使用单个线程
    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft", n_jobs=1
    ).fit(X, y)
    
    # 创建软投票分类器 eclf2，包含逻辑回归、随机森林和高斯朴素贝叶斯分类器，使用两个线程
    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft", n_jobs=2
    ).fit(X, y)

    # 断言 eclf1 和 eclf2 对相同输入 X 的预测结果相等
    assert_array_equal(eclf1.predict(X), eclf2.predict(X))
    # 断言 eclf1 和 eclf2 对相同输入 X 的预测概率近似相等
    assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))
@ignore_warnings(category=FutureWarning)
# 使用装饰器忽略 FutureWarning 类别的警告

def test_sample_weight(global_random_seed):
    """Tests sample_weight parameter of VotingClassifier"""
    # 测试 VotingClassifier 的 sample_weight 参数

    clf1 = LogisticRegression(random_state=global_random_seed)
    # 使用全局随机种子创建 LogisticRegression 分类器

    clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed)
    # 使用全局随机种子创建 RandomForestClassifier 分类器，设定树的数量为 10

    clf3 = SVC(probability=True, random_state=global_random_seed)
    # 使用全局随机种子创建 SVC 分类器，启用概率估计

    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("svc", clf3)], voting="soft"
    ).fit(X_scaled, y, sample_weight=np.ones((len(y),)))
    # 创建 VotingClassifier 对象 eclf1，包含三个基分类器，使用软投票策略，传入样本权重

    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("svc", clf3)], voting="soft"
    ).fit(X_scaled, y)
    # 创建 VotingClassifier 对象 eclf2，包含三个基分类器，使用软投票策略，未传入样本权重

    assert_array_equal(eclf1.predict(X_scaled), eclf2.predict(X_scaled))
    # 断言：比较 eclf1 和 eclf2 在 X_scaled 上的预测结果是否一致

    assert_array_almost_equal(
        eclf1.predict_proba(X_scaled), eclf2.predict_proba(X_scaled)
    )
    # 断言：比较 eclf1 和 eclf2 在 X_scaled 上的预测概率是否接近

    sample_weight = np.random.RandomState(global_random_seed).uniform(size=(len(y),))
    # 使用全局随机种子创建均匀分布的样本权重数组

    eclf3 = VotingClassifier(estimators=[("lr", clf1)], voting="soft")
    # 创建 VotingClassifier 对象 eclf3，只包含一个基分类器 lr，使用软投票策略

    eclf3.fit(X_scaled, y, sample_weight)
    # 对 eclf3 应用样本权重进行训练

    clf1.fit(X_scaled, y, sample_weight)
    # 对 clf1 单独应用样本权重进行训练

    assert_array_equal(eclf3.predict(X_scaled), clf1.predict(X_scaled))
    # 断言：比较 eclf3 和 clf1 在 X_scaled 上的预测结果是否一致

    assert_array_almost_equal(
        eclf3.predict_proba(X_scaled), clf1.predict_proba(X_scaled)
    )
    # 断言：比较 eclf3 和 clf1 在 X_scaled 上的预测概率是否接近

    # 检查当 sample_weight 不被支持时是否会引发错误
    clf4 = KNeighborsClassifier()
    # 创建 KNeighborsClassifier 分类器对象 clf4

    eclf3 = VotingClassifier(
        estimators=[("lr", clf1), ("svc", clf3), ("knn", clf4)], voting="soft"
    )
    # 创建 VotingClassifier 对象 eclf3，包含三个基分类器，使用软投票策略

    msg = "Underlying estimator KNeighborsClassifier does not support sample weights."
    # 错误消息：底层的 KNeighborsClassifier 分类器不支持样本权重

    with pytest.raises(TypeError, match=msg):
        eclf3.fit(X_scaled, y, sample_weight)

    # 断言：使用 pytest 检查是否引发 TypeError 错误，错误消息应为 msg

    # 检查 _fit_single_estimator 是否会引发正确的错误
    # 如果错误与 sample_weight 无关，则应引发原始错误
    class ClassifierErrorFit(ClassifierMixin, BaseEstimator):
        def fit(self, X_scaled, y, sample_weight):
            raise TypeError("Error unrelated to sample_weight.")

    clf = ClassifierErrorFit()
    # 创建 ClassifierErrorFit 分类器对象 clf

    with pytest.raises(TypeError, match="Error unrelated to sample_weight"):
        clf.fit(X_scaled, y, sample_weight=sample_weight)
    # 断言：使用 pytest 检查是否引发 TypeError 错误，错误消息应为 "Error unrelated to sample_weight"

def test_sample_weight_kwargs():
    """Check that VotingClassifier passes sample_weight as kwargs"""
    # 检查 VotingClassifier 是否将 sample_weight 作为关键字参数传递

    class MockClassifier(ClassifierMixin, BaseEstimator):
        """Mock Classifier to check that sample_weight is received as kwargs"""

        def fit(self, X, y, *args, **sample_weight):
            assert "sample_weight" in sample_weight
            # 断言：确保 sample_weight 在关键字参数中

    clf = MockClassifier()
    # 创建 MockClassifier 分类器对象 clf

    eclf = VotingClassifier(estimators=[("mock", clf)], voting="soft")
    # 创建 VotingClassifier 对象 eclf，包含一个基分类器 mock，使用软投票策略

    # 不应引发错误
    eclf.fit(X, y, sample_weight=np.ones((len(y),)))
    # 使用均匀权重对 eclf 进行训练

def test_voting_classifier_set_params(global_random_seed):
    # 检查设置底层估计器时输出的等价性
    clf1 = LogisticRegression(random_state=global_random_seed)
    # 使用全局随机种子创建 LogisticRegression 分类器
    # 创建一个随机森林分类器，设置了决策树数量为10，随机数种子为全局随机种子，最大深度为None
    clf2 = RandomForestClassifier(
        n_estimators=10, random_state=global_random_seed, max_depth=None
    )
    # 创建一个高斯朴素贝叶斯分类器
    clf3 = GaussianNB()

    # 创建一个软投票集成分类器eclf1，包含逻辑回归分类器clf1和随机森林分类器clf2，权重分别为1和2，使用X_scaled和y进行拟合
    eclf1 = VotingClassifier(
        [("lr", clf1), ("rf", clf2)], voting="soft", weights=[1, 2]
    ).fit(X_scaled, y)
    
    # 创建另一个软投票集成分类器eclf2，包含逻辑回归分类器clf1和高斯朴素贝叶斯分类器clf3，权重分别为1和2
    eclf2 = VotingClassifier(
        [("lr", clf1), ("nb", clf3)], voting="soft", weights=[1, 2]
    )
    # 将eclf2的nb分类器参数设置为clf2，并使用X_scaled和y进行拟合
    eclf2.set_params(nb=clf2).fit(X_scaled, y)

    # 断言eclf1对X_scaled的预测结果与eclf2相同
    assert_array_equal(eclf1.predict(X_scaled), eclf2.predict(X_scaled))
    # 断言eclf1对X_scaled的预测概率与eclf2相同
    assert_array_almost_equal(
        eclf1.predict_proba(X_scaled), eclf2.predict_proba(X_scaled)
    )
    # 断言eclf2的第一个分类器（lr）的参数与clf1相同
    assert eclf2.estimators[0][1].get_params() == clf1.get_params()
    # 断言eclf2的第二个分类器（nb）的参数与clf2相同
    assert eclf2.estimators[1][1].get_params() == clf2.get_params()
# 定义一个测试函数，用于验证 VotingClassifier 的一些功能
def test_set_estimator_drop():
    # VotingClassifier 的 set_params 方法应该能够将估算器设置为 "drop"
    # 测试预测功能
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=123)
    clf3 = GaussianNB()
    
    # 创建 VotingClassifier 对象 eclf1，并使用 fit 方法进行训练
    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("nb", clf3)],
        voting="hard",
        weights=[1, 0, 0.5],
    ).fit(X, y)

    # 创建 VotingClassifier 对象 eclf2，并使用 set_params 方法将 rf 设置为 "drop"，然后使用 fit 方法进行训练
    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("nb", clf3)],
        voting="hard",
        weights=[1, 1, 0.5],
    )
    eclf2.set_params(rf="drop").fit(X, y)

    # 断言预测结果的一致性
    assert_array_equal(eclf1.predict(X), eclf2.predict(X))

    # 断言 rf 估算器被成功设置为 "drop"
    assert dict(eclf2.estimators)["rf"] == "drop"
    # 断言 eclf2 的估算器列表长度为 2
    assert len(eclf2.estimators_) == 2
    # 断言 eclf2 中所有估算器均为 LogisticRegression 或 GaussianNB 的实例
    assert all(
        isinstance(est, (LogisticRegression, GaussianNB)) for est in eclf2.estimators_
    )
    # 断言 eclf2 的参数中 rf 被设置为 "drop"
    assert eclf2.get_params()["rf"] == "drop"

    # 使用 soft voting 方式重新设置 VotingClassifier 并进行训练
    eclf1.set_params(voting="soft").fit(X, y)
    eclf2.set_params(voting="soft").fit(X, y)

    # 断言预测结果的一致性
    assert_array_equal(eclf1.predict(X), eclf2.predict(X))
    # 断言预测概率的一致性
    assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))

    # 测试所有估算器都被移除的情况
    msg = "All estimators are dropped. At least one is required"
    with pytest.raises(ValueError, match=msg):
        # 使用 set_params 方法将 lr, rf, nb 全部设置为 "drop" 并进行训练，预期会引发 ValueError
        eclf2.set_params(lr="drop", rf="drop", nb="drop").fit(X, y)

    # 测试 soft voting 的 transform 方法
    X1 = np.array([[1], [2]])
    y1 = np.array([1, 2])
    eclf1 = VotingClassifier(
        estimators=[("rf", clf2), ("nb", clf3)],
        voting="soft",
        weights=[0, 0.5],
        flatten_transform=False,
    ).fit(X1, y1)

    eclf2 = VotingClassifier(
        estimators=[("rf", clf2), ("nb", clf3)],
        voting="soft",
        weights=[1, 0.5],
        flatten_transform=False,
    )
    # 使用 set_params 方法将 rf 设置为 "drop" 并进行训练
    eclf2.set_params(rf="drop").fit(X1, y1)

    # 断言 transform 方法的输出是否符合预期
    assert_array_almost_equal(
        eclf1.transform(X1),
        np.array([[[0.7, 0.3], [0.3, 0.7]], [[1.0, 0.0], [0.0, 1.0]]]),
    )
    assert_array_almost_equal(eclf2.transform(X1), np.array([[[1.0, 0.0], [0.0, 1.0]]]))

    # 将 voting 设置为 hard 并重新设置 VotingClassifier 并进行比较
    eclf1.set_params(voting="hard")
    eclf2.set_params(voting="hard")
    assert_array_equal(eclf1.transform(X1), np.array([[0, 0], [1, 1]]))
    assert_array_equal(eclf2.transform(X1), np.array([[0], [1]]))


def test_estimator_weights_format(global_random_seed):
    # 测试 estimator weights 的输入格式，可以是列表或数组
    clf1 = LogisticRegression(random_state=global_random_seed)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed)
    
    # 使用列表作为 weights 的输入方式创建 VotingClassifier 对象 eclf1
    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2)], weights=[1, 2], voting="soft"
    )
    # 使用数组作为 weights 的输入方式创建 VotingClassifier 对象 eclf2
    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2)], weights=np.array((1, 2)), voting="soft"
    )
    # 使用 X_scaled 和 y 进行训练
    eclf1.fit(X_scaled, y)
    eclf2.fit(X_scaled, y)
    # 断言两个 VotingClassifier 对象的预测概率是否相近
    assert_array_almost_equal(
        eclf1.predict_proba(X_scaled), eclf2.predict_proba(X_scaled)
    )
    """Check transform method of VotingClassifier on toy dataset."""
    # 定义三个不同的分类器对象
    clf1 = LogisticRegression(random_state=global_random_seed)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed)
    clf3 = GaussianNB()

    # 创建示例数据集 X 和对应的标签 y
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])

    # 创建三个投票分类器对象，分别进行拟合
    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft"
    ).fit(X, y)

    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="soft",
        flatten_transform=True,
    ).fit(X, y)

    eclf3 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="soft",
        flatten_transform=False,
    ).fit(X, y)

    # 检查转换方法的输出形状是否符合预期
    assert_array_equal(eclf1.transform(X).shape, (4, 6))
    assert_array_equal(eclf2.transform(X).shape, (4, 6))
    assert_array_equal(eclf3.transform(X).shape, (3, 4, 2))
    assert_array_almost_equal(eclf1.transform(X), eclf2.transform(X))
    assert_array_almost_equal(
        eclf3.transform(X).swapaxes(0, 1).reshape((4, 6)), eclf2.transform(X)
    )
@pytest.mark.parametrize(
    "X, y, voter",
    [
        (
            X,  # 特征数据
            y,  # 目标数据
            VotingClassifier(  # 使用投票分类器
                [
                    ("lr", LogisticRegression()),  # 逻辑回归作为一个估计器
                    ("rf", RandomForestClassifier(n_estimators=5)),  # 随机森林作为一个估计器，包含5棵树
                ]
            ),
        ),
        (
            X_r,  # 特征数据
            y_r,  # 目标数据
            VotingRegressor(  # 使用投票回归器
                [
                    ("lr", LinearRegression()),  # 线性回归作为一个估计器
                    ("rf", RandomForestRegressor(n_estimators=5)),  # 随机森林回归作为一个估计器，包含5棵树
                ]
            ),
        ),
    ],
)
def test_none_estimator_with_weights(X, y, voter):
    # 检查是否可以将估计器设置为'drop'并传递一些权重
    # 回归测试，用于解决https://github.com/scikit-learn/scikit-learn/issues/13777中的问题
    voter = clone(voter)  # 克隆投票器
    # 标准化解决逻辑回归引发的收敛警告
    X_scaled = StandardScaler().fit_transform(X)  # 对特征数据进行标准化
    voter.fit(X_scaled, y, sample_weight=np.ones(y.shape))  # 使用样本权重拟合投票器
    voter.set_params(lr="drop")  # 设置逻辑回归估计器为'drop'
    voter.fit(X_scaled, y, sample_weight=np.ones(y.shape))  # 再次使用样本权重拟合投票器
    y_pred = voter.predict(X_scaled)  # 预测目标数据
    assert y_pred.shape == y.shape  # 断言预测结果的形状与目标数据一致


@pytest.mark.parametrize(
    "est",
    [
        VotingRegressor(  # 投票回归器
            estimators=[
                ("lr", LinearRegression()),  # 线性回归作为一个估计器
                ("tree", DecisionTreeRegressor(random_state=0)),  # 决策树回归作为一个估计器，使用随机种子0
            ]
        ),
        VotingClassifier(  # 投票分类器
            estimators=[
                ("lr", LogisticRegression(random_state=0)),  # 逻辑回归作为一个估计器，使用随机种子0
                ("tree", DecisionTreeClassifier(random_state=0)),  # 决策树分类作为一个估计器，使用随机种子0
            ]
        ),
    ],
    ids=["VotingRegressor", "VotingClassifier"],  # 为每个测试案例指定标识符
)
def test_n_features_in(est):
    X = [[1, 2], [3, 4], [5, 6]]  # 特征数据
    y = [0, 1, 2]  # 目标数据

    assert not hasattr(est, "n_features_in_")  # 断言投票器没有属性'n_features_in_'
    est.fit(X, y)  # 拟合投票器
    assert est.n_features_in_ == 2  # 断言投票器的'n_features_in_'属性为2


@pytest.mark.parametrize(
    "estimator",
    [
        VotingRegressor(  # 投票回归器
            estimators=[
                ("lr", LinearRegression()),  # 线性回归作为一个估计器
                ("rf", RandomForestRegressor(random_state=123)),  # 随机森林回归作为一个估计器，使用随机种子123
            ],
            verbose=True,  # 启用详细信息
        ),
        VotingClassifier(  # 投票分类器
            estimators=[
                ("lr", LogisticRegression(random_state=123)),  # 逻辑回归作为一个估计器，使用随机种子123
                ("rf", RandomForestClassifier(random_state=123)),  # 随机森林分类器作为一个估计器，使用随机种子123
            ],
            verbose=True,  # 启用详细信息
        ),
    ],
)
def test_voting_verbose(estimator, capsys):
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])  # 特征数据
    y = np.array([1, 1, 2, 2])  # 目标数据

    pattern = (
        r"\[Voting\].*\(1 of 2\) Processing lr, total=.*\n"  # 匹配模式，用于验证输出是否符合预期
        r"\[Voting\].*\(2 of 2\) Processing rf, total=.*\n$"  # 匹配模式，用于验证输出是否符合预期
    )

    estimator.fit(X, y)  # 拟合投票器
    assert re.match(pattern, capsys.readouterr()[0])  # 断言输出符合预期的模式


def test_get_features_names_out_regressor():
    """Check get_feature_names_out output for regressor."""
    X = [[1, 2], [3, 4], [5, 6]]  # 特征数据
    y = [0, 1, 2]  # 目标数据
    # 创建一个投票回归器，包含三个子模型：线性回归器、决策树回归器和一个无效的"drop"模型
    voting = VotingRegressor(
        estimators=[
            ("lr", LinearRegression()),  # 使用线性回归器作为子模型"lr"
            ("tree", DecisionTreeRegressor(random_state=0)),  # 使用决策树回归器作为子模型"tree"
            ("ignore", "drop"),  # 无效模型，不参与投票
        ]
    )
    # 使用输入数据 X 和目标数据 y 对投票回归器进行训练
    voting.fit(X, y)

    # 获取投票回归器输出的特征名称列表
    names_out = voting.get_feature_names_out()
    # 预期的特征名称列表，包括投票回归器中线性回归器和决策树回归器的输出
    expected_names = ["votingregressor_lr", "votingregressor_tree"]
    # 断言获取的特征名称列表与预期的特征名称列表相等
    assert_array_equal(names_out, expected_names)
# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
@pytest.mark.parametrize(
    "kwargs, expected_names",
    [  # 参数化测试数据集，包含两组测试参数和期望结果
        (
            {"voting": "soft", "flatten_transform": True},  # 第一组参数和期望结果
            [
                "votingclassifier_lr0",
                "votingclassifier_lr1",
                "votingclassifier_lr2",
                "votingclassifier_tree0",
                "votingclassifier_tree1",
                "votingclassifier_tree2",
            ],
        ),
        (
            {"voting": "hard"},  # 第二组参数和期望结果
            ["votingclassifier_lr", "votingclassifier_tree"],
        ),
    ],
)
def test_get_features_names_out_classifier(kwargs, expected_names):
    """Check get_feature_names_out for classifier for different settings."""
    X = [[1, 2], [3, 4], [5, 6], [1, 1.2]]  # 创建特征矩阵 X
    y = [0, 1, 2, 0]  # 创建标签向量 y

    # 创建 VotingClassifier 对象，传入 LogisticRegression 和 DecisionTreeClassifier 作为估计器
    voting = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(random_state=0)),
            ("tree", DecisionTreeClassifier(random_state=0)),
        ],
        **kwargs,  # 使用传入的参数
    )
    voting.fit(X, y)  # 使用 X 和 y 拟合模型
    X_trans = voting.transform(X)  # 对 X 进行变换
    names_out = voting.get_feature_names_out()  # 获取输出特征名称列表

    assert X_trans.shape[1] == len(expected_names)  # 断言变换后的特征数量与期望的名称数量相等
    assert_array_equal(names_out, expected_names)  # 断言获取的特征名称列表与期望的名称列表相等


def test_get_features_names_out_classifier_error():
    """Check that error is raised when voting="soft" and flatten_transform=False."""
    X = [[1, 2], [3, 4], [5, 6]]  # 创建特征矩阵 X
    y = [0, 1, 2]  # 创建标签向量 y

    # 创建 VotingClassifier 对象，传入 LogisticRegression 和 DecisionTreeClassifier 作为估计器
    # 设置 voting="soft" 和 flatten_transform=False
    voting = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(random_state=0)),
            ("tree", DecisionTreeClassifier(random_state=0)),
        ],
        voting="soft",
        flatten_transform=False,
    )
    voting.fit(X, y)  # 使用 X 和 y 拟合模型

    # 断言捕获 ValueError，并检查错误消息是否包含特定文本
    msg = (
        "get_feature_names_out is not supported when `voting='soft'` and "
        "`flatten_transform=False`"
    )
    with pytest.raises(ValueError, match=msg):
        voting.get_feature_names_out()


# Metadata Routing Tests
# ======================


@pytest.mark.parametrize(
    "Estimator, Child",
    [(VotingClassifier, ConsumingClassifier), (VotingRegressor, ConsumingRegressor)],
)
def test_routing_passed_metadata_not_supported(Estimator, Child):
    """Test that the right error message is raised when metadata is passed while
    not supported when `enable_metadata_routing=False`."""

    X = np.array([[0, 1], [2, 2], [4, 6]])  # 创建特征矩阵 X
    y = [1, 2, 3]  # 创建标签向量 y

    # 断言捕获 ValueError，并检查错误消息是否包含特定文本
    with pytest.raises(
        ValueError, match="is only supported if enable_metadata_routing=True"
    ):
        Estimator(["clf", Child()]).fit(X, y, sample_weight=[1, 1, 1], metadata="a")


@pytest.mark.usefixtures("enable_slep006")
@pytest.mark.parametrize(
    "Estimator, Child",
    [(VotingClassifier, ConsumingClassifier), (VotingRegressor, ConsumingRegressor)],
)
def test_get_metadata_routing_without_fit(Estimator, Child):
    # Test that metadata_routing() doesn't raise when called before fit.
    est = Estimator([("sub_est", Child())])  # 创建估计器对象
    est.get_metadata_routing()  # 调用获取元数据路由方法


@pytest.mark.usefixtures("enable_slep006")
@pytest.mark.parametrize(
    "Estimator, Child",
    # 创建一个包含两个元组的列表，每个元组包含两个类（VotingClassifier 和 ConsumingClassifier）。
    # 这些类用于分类任务。另一个元组包含两个类（VotingRegressor 和 ConsumingRegressor），
    # 用于回归任务。
    [(VotingClassifier, ConsumingClassifier), (VotingRegressor, ConsumingRegressor)],
# 使用 pytest.mark.parametrize 装饰器标记参数化测试用例，参数为 "prop"，可以依次取值 "sample_weight" 和 "metadata"
@pytest.mark.parametrize("prop", ["sample_weight", "metadata"])
# 定义测试函数 test_metadata_routing_for_voting_estimators，测试元数据在投票估计器中的路由是否正确
def test_metadata_routing_for_voting_estimators(Estimator, Child, prop):
    """Test that metadata is routed correctly for Voting*."""
    # 创建示例数据 X 和 y
    X = np.array([[0, 1], [2, 2], [4, 6]])
    y = [1, 2, 3]
    # 设置样本权重和元数据
    sample_weight, metadata = [1, 1, 1], "a"

    # 创建估计器实例 est，包含两个子估计器
    est = Estimator(
        [
            (
                "sub_est1",
                Child(registry=_Registry()).set_fit_request(**{prop: True}),
            ),
            (
                "sub_est2",
                Child(registry=_Registry()).set_fit_request(**{prop: True}),
            ),
        ]
    )

    # 使用 est.fit 进行拟合，根据不同的 prop 设置样本权重或元数据
    est.fit(X, y, **{prop: sample_weight if prop == "sample_weight" else metadata})

    # 遍历 est 中的每个子估计器，根据 prop 设置不同的 kwargs
    for estimator in est.estimators:
        if prop == "sample_weight":
            kwargs = {prop: sample_weight}
        else:
            kwargs = {prop: metadata}
        # 访问子估计器的 registry 属性
        registry = estimator[1].registry
        # 断言 registry 的长度不为 0
        assert len(registry)
        # 遍历 registry 中的每个子估计器，检查记录的元数据
        for sub_est in registry:
            check_recorded_metadata(obj=sub_est, method="fit", parent="fit", **kwargs)


# 使用 pytest.mark.usefixtures 装饰器启用 slep006
@pytest.mark.usefixtures("enable_slep006")
# 参数化测试函数，参数为 Estimator 和 Child，分别为 VotingClassifier 与 ConsumingClassifier，以及 VotingRegressor 与 ConsumingRegressor
@pytest.mark.parametrize(
    "Estimator, Child",
    [(VotingClassifier, ConsumingClassifier), (VotingRegressor, ConsumingRegressor)],
)
# 定义测试函数 test_metadata_routing_error_for_voting_estimators，测试当未请求元数据时是否会引发正确的错误
def test_metadata_routing_error_for_voting_estimators(Estimator, Child):
    """Test that the right error is raised when metadata is not requested."""
    # 创建示例数据 X 和 y
    X = np.array([[0, 1], [2, 2], [4, 6]])
    y = [1, 2, 3]
    # 设置样本权重和元数据
    sample_weight, metadata = [1, 1, 1], "a"

    # 创建估计器实例 est，包含一个子估计器
    est = Estimator([("sub_est", Child())])

    # 定义错误消息
    error_message = (
        "[sample_weight, metadata] are passed but are not explicitly set as requested"
        f" or not requested for {Child.__name__}.fit"
    )

    # 使用 pytest.raises 检查是否引发 ValueError，并匹配错误消息
    with pytest.raises(ValueError, match=re.escape(error_message)):
        est.fit(X, y, sample_weight=sample_weight, metadata=metadata)


# End of Metadata Routing Tests
# =============================
```