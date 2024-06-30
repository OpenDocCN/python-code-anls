# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_multiclass.py`

```
# 从 re 模块中导入 escape 函数，用于正则表达式的特殊字符转义
from re import escape

# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 pytest 库
import pytest
# 导入 scipy.sparse 模块并重命名为 sp
import scipy.sparse as sp
# 从 numpy.testing 模块中导入 assert_allclose 函数
from numpy.testing import assert_allclose

# 导入 sklearn 库中的多个模块和类
from sklearn import datasets, svm
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Perceptron,
    Ridge,
    SGDClassifier,
)
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import (
    check_array,
    shuffle,
)
# 导入 sklearn.utils._mocking 模块中的 CheckingClassifier 类
from sklearn.utils._mocking import CheckingClassifier
# 导入 sklearn.utils._testing 模块中的 assert_almost_equal 和 assert_array_equal 函数
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
# 导入 sklearn.utils.fixes 模块中的容器相关常量
from sklearn.utils.fixes import (
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
)
# 导入 sklearn.utils.multiclass 模块中的 check_classification_targets 和 type_of_target 函数
from sklearn.utils.multiclass import check_classification_targets, type_of_target

# 设置 pytest 的标记以忽略特定的 FutureWarning
msg = "The default value for `force_alpha` will change"
pytestmark = pytest.mark.filterwarnings(f"ignore:{msg}:FutureWarning")

# 加载鸢尾花数据集
iris = datasets.load_iris()
# 设置随机数生成器的种子，并生成一个随机排列的索引数组
rng = np.random.RandomState(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
# 确定类别的数量
n_classes = 3


def test_ovr_exceptions():
    # 创建一个 OneVsRestClassifier 对象，使用 LinearSVC 作为基分类器
    ovr = OneVsRestClassifier(LinearSVC(random_state=0))

    # 测试在没有拟合的情况下进行预测是否会引发 NotFittedError 异常
    with pytest.raises(NotFittedError):
        ovr.predict([])

    # 测试多输出数据是否会引发 ValueError 异常，提示不支持标签二值化
    msg = "Multioutput target data is not supported with label binarization"
    with pytest.raises(ValueError, match=msg):
        X = np.array([[1, 0], [0, 1]])
        y = np.array([[1, 2], [3, 1]])
        OneVsRestClassifier(MultinomialNB()).fit(X, y)

    with pytest.raises(ValueError, match=msg):
        X = np.array([[1, 0], [0, 1]])
        y = np.array([[1.5, 2.4], [3.1, 0.8]])
        OneVsRestClassifier(MultinomialNB()).fit(X, y)


def test_check_classification_targets():
    # 测试 check_classification_targets 函数的行为，验证目标类型的正确性
    y = np.array([0.0, 1.1, 2.0, 3.0])
    msg = type_of_target(y)
    with pytest.raises(ValueError, match=msg):
        check_classification_targets(y)


def test_ovr_fit_predict():
    # 创建一个 OneVsRestClassifier 对象，使用 LinearSVC 作为基分类器
    ovr = OneVsRestClassifier(LinearSVC(random_state=0))
    # 对鸢尾花数据集进行拟合和预测
    pred = ovr.fit(iris.data, iris.target).predict(iris.data)
    # 断言分类器的数量等于类别数量
    assert len(ovr.estimators_) == n_classes

    # 创建一个普通的 LinearSVC 分类器
    clf = LinearSVC(random_state=0)
    # 对鸢尾花数据集进行拟合和预测
    pred2 = clf.fit(iris.data, iris.target).predict(iris.data)
    # 断言：验证两个预测数组的平均值是否相等
    assert np.mean(iris.target == pred) == np.mean(iris.target == pred2)
    
    # 创建一个多类别分类器，使用朴素贝叶斯模型（MultinomialNB），支持predict_proba方法
    ovr = OneVsRestClassifier(MultinomialNB())
    
    # 使用iris数据集进行拟合，并对数据集进行预测
    pred = ovr.fit(iris.data, iris.target).predict(iris.data)
    
    # 断言：验证预测准确率是否大于0.65
    assert np.mean(iris.target == pred) > 0.65
# 定义测试函数，用于测试 OneVsRestClassifier 的 partial_fit 方法
def test_ovr_partial_fit():
    # 将 iris 数据集中的特征和目标标签进行随机打乱
    X, y = shuffle(iris.data, iris.target, random_state=0)
    # 创建 OneVsRestClassifier 对象，使用 MultinomialNB 作为基础分类器
    ovr = OneVsRestClassifier(MultinomialNB())
    # 对前100个样本进行部分拟合
    ovr.partial_fit(X[:100], y[:100], np.unique(y))
    # 对剩余样本进行部分拟合
    ovr.partial_fit(X[100:], y[100:])
    # 使用拟合好的模型进行预测
    pred = ovr.predict(X)
    # 创建另一个 OneVsRestClassifier 对象，用于比较测试
    ovr2 = OneVsRestClassifier(MultinomialNB())
    # 对所有数据进行拟合并预测
    pred2 = ovr2.fit(X, y).predict(X)

    # 断言预测结果的近似相等
    assert_almost_equal(pred, pred2)
    # 断言分类器的个数与目标标签的唯一值个数相等
    assert len(ovr.estimators_) == len(np.unique(y))
    # 断言预测结果的平均准确率大于0.65
    assert np.mean(y == pred) > 0.65

    # 使用 SGDClassifier 测试部分拟合时，小批量数据不包含所有类别的情况
    X = np.abs(np.random.randn(14, 2))
    y = [1, 1, 1, 1, 2, 3, 3, 0, 0, 2, 3, 1, 2, 3]

    # 创建 OneVsRestClassifier 对象，使用 SGDClassifier 作为基础分类器
    ovr = OneVsRestClassifier(
        SGDClassifier(max_iter=1, tol=None, shuffle=False, random_state=0)
    )
    # 对前7个样本进行部分拟合
    ovr.partial_fit(X[:7], y[:7], np.unique(y))
    # 对剩余样本进行部分拟合
    ovr.partial_fit(X[7:], y[7:])
    # 使用拟合好的模型进行预测
    pred = ovr.predict(X)
    # 创建另一个 OneVsRestClassifier 对象，用于比较测试
    ovr1 = OneVsRestClassifier(
        SGDClassifier(max_iter=1, tol=None, shuffle=False, random_state=0)
    )
    # 对所有数据进行拟合并预测
    pred1 = ovr1.fit(X, y).predict(X)
    # 断言两种方法的预测结果平均准确率相等
    assert np.mean(pred == y) == np.mean(pred1 == y)

    # 测试如果基础估计器不支持 partial_fit 方法，断言不存在 partial_fit 方法
    ovr = OneVsRestClassifier(SVC())
    assert not hasattr(ovr, "partial_fit")


def test_ovr_partial_fit_exceptions():
    # 创建 OneVsRestClassifier 对象，使用 MultinomialNB 作为基础分类器
    ovr = OneVsRestClassifier(MultinomialNB())
    X = np.abs(np.random.randn(14, 2))
    y = [1, 1, 1, 1, 2, 3, 3, 0, 0, 2, 3, 1, 2, 3]
    # 对前7个样本进行部分拟合
    ovr.partial_fit(X[:7], y[:7], np.unique(y))
    # 测试如果在第一次部分拟合中出现新的类别，会抛出 ValueError 异常
    y1 = [5] + y[7:-1]
    msg = r"Mini-batch contains \[.+\] while classes must be subset of \[.+\]"
    with pytest.raises(ValueError, match=msg):
        ovr.partial_fit(X=X[7:], y=y1)


def test_ovr_ovo_regressor():
    # 测试回归器的 OneVsRest 和 OneVsOne 模式，这些模型没有 decision_function
    # 创建 OneVsRestClassifier 对象，使用 DecisionTreeRegressor 作为基础回归器
    ovr = OneVsRestClassifier(DecisionTreeRegressor())
    # 对 iris 数据集进行拟合并预测
    pred = ovr.fit(iris.data, iris.target).predict(iris.data)
    # 断言分类器的数量与类别数量相等
    assert len(ovr.estimators_) == n_classes
    # 断言预测结果的唯一值与类别标签相等
    assert_array_equal(np.unique(pred), [0, 1, 2])
    # 断言预测结果的平均准确率高于0.9
    assert np.mean(pred == iris.target) > 0.9

    # 创建 OneVsOneClassifier 对象，使用 DecisionTreeRegressor 作为基础回归器
    ovr = OneVsOneClassifier(DecisionTreeRegressor())
    # 对 iris 数据集进行拟合并预测
    pred = ovr.fit(iris.data, iris.target).predict(iris.data)
    # 断言分类器的数量与两两组合的类别数量相等
    assert len(ovr.estimators_) == n_classes * (n_classes - 1) / 2
    # 断言预测结果的唯一值与类别标签相等
    assert_array_equal(np.unique(pred), [0, 1, 2])
    # 断言预测结果的平均准确率高于0.9
    assert np.mean(pred == iris.target) > 0.9


@pytest.mark.parametrize(
    "sparse_container",
    CSR_CONTAINERS + CSC_CONTAINERS + COO_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS,
)
def test_ovr_fit_predict_sparse(sparse_container):
    # 创建基础分类器对象
    base_clf = MultinomialNB(alpha=1)

    # 生成多标签分类数据集
    X, Y = datasets.make_multilabel_classification(
        n_samples=100,
        n_features=20,
        n_classes=5,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )

    # 取前80个样本作为训练集
    X_train, Y_train = X[:80], Y[:80]
    # 将数据集 X 的后 20% 划分为测试集 X_test
    X_test = X[80:]

    # 使用 OneVsRestClassifier 将 base_clf 作为基分类器，对训练集 X_train, Y_train 进行拟合
    clf = OneVsRestClassifier(base_clf).fit(X_train, Y_train)
    
    # 对测试集 X_test 进行预测，得到预测结果 Y_pred
    Y_pred = clf.predict(X_test)

    # 使用 sparse_container 将 Y_train 转换为稀疏格式，再次使用 OneVsRestClassifier 进行拟合
    clf_sprs = OneVsRestClassifier(base_clf).fit(X_train, sparse_container(Y_train))
    
    # 对稀疏格式的 Y_pred_sprs 进行预测，得到预测结果 Y_pred_sprs
    Y_pred_sprs = clf_sprs.predict(X_test)

    # 断言 clf 模型是多标签分类器
    assert clf.multilabel_
    
    # 断言 Y_pred_sprs 是稀疏矩阵
    assert sp.issparse(Y_pred_sprs)
    
    # 断言 Y_pred_sprs 转换为普通数组后与 Y_pred 相等
    assert_array_equal(Y_pred_sprs.toarray(), Y_pred)

    # 测试 predict_proba 方法
    Y_proba = clf_sprs.predict_proba(X_test)

    # 若预测概率大于 0.5，则 pred 被赋予相应的标签
    pred = Y_proba > 0.5
    
    # 断言 pred 与 Y_pred_sprs 转换为普通数组后相等
    assert_array_equal(pred, Y_pred_sprs.toarray())

    # 测试 decision_function 方法
    clf = svm.SVC()
    
    # 使用 OneVsRestClassifier 将 SVC 作为基分类器对稀疏格式的 Y_train 进行拟合
    clf_sprs = OneVsRestClassifier(clf).fit(X_train, sparse_container(Y_train))
    
    # 将 decision_function 的结果大于 0 的部分转换为整数，得到 dec_pred
    dec_pred = (clf_sprs.decision_function(X_test) > 0).astype(int)
    
    # 断言 dec_pred 与 clf_sprs.predict(X_test) 转换为普通数组后相等
    assert_array_equal(dec_pred, clf_sprs.predict(X_test).toarray())
# 测试 `OneVsRestClassifier` 在总是存在或不存在的类的情况下的工作情况。
# 创建一个包含 10 行、2列的全一矩阵，并将前五行的所有列置零
X = np.ones((10, 2))
X[:5, :] = 0

# 创建一个指示矩阵，其中两个特征总是开启。
# 作为列表的列表，它会是：[[int(i >= 5), 2, 3] for i in range(10)]
y = np.zeros((10, 3))
y[5:, 0] = 1
y[:, 1] = 1
y[:, 2] = 1

# 使用 LogisticRegression 作为基础分类器，构建 One-vs-Rest 分类器
ovr = OneVsRestClassifier(LogisticRegression())

# 设定警告消息，检查是否出现特定警告
msg = r"Label .+ is present in all training examples"
with pytest.warns(UserWarning, match=msg):
    # 使用 X 和 y 训练分类器
    ovr.fit(X, y)

# 对 X 进行预测
y_pred = ovr.predict(X)
# 断言预测结果与真实标签 y 相等
assert_array_equal(np.array(y_pred), np.array(y))

# 获取决策函数的预测结果
y_pred = ovr.decision_function(X)
# 断言决策函数的预测结果中只有两个类别，值为 1
assert np.unique(y_pred[:, -2:]) == 1

# 获取概率预测结果
y_pred = ovr.predict_proba(X)
# 断言最后一列的概率预测结果都为 1
assert_array_equal(y_pred[:, -1], np.ones(X.shape[0]))

# 将 y 的一个标签总是不存在
y = np.zeros((10, 2))
y[5:, 0] = 1  # 变量标签

# 使用 LogisticRegression 作为基础分类器，构建 One-vs-Rest 分类器
ovr = OneVsRestClassifier(LogisticRegression())

# 设定警告消息，检查是否出现特定警告
msg = r"Label not 1 is present in all training examples"
with pytest.warns(UserWarning, match=msg):
    # 使用 X 和 y 训练分类器
    ovr.fit(X, y)

# 获取概率预测结果
y_pred = ovr.predict_proba(X)
# 断言最后一列的概率预测结果都为 0
assert_array_equal(y_pred[:, -1], np.zeros(X.shape[0]))
    # 定义一个函数，用于执行测试
    def conduct_test(base_clf, test_predict_proba=False):
        # 使用 OneVsRestClassifier 将 base_clf 转化为多类分类器，并在训练集 X, y 上拟合
        clf = OneVsRestClassifier(base_clf).fit(X, y)
        # 断言分类器 clf 的类别集合与预期的类别集合 classes 相同
        assert set(clf.classes_) == classes
        # 对输入 [[0, 0, 4]] 进行预测
        y_pred = clf.predict(np.array([[0, 0, 4]]))[0]
        # 断言预测结果为 ["eggs"]
        assert_array_equal(y_pred, ["eggs"])
        # 如果 base_clf 具有 decision_function 方法
        if hasattr(base_clf, "decision_function"):
            # 获取分类器 clf 的决策函数值 dec
            dec = clf.decision_function(X)
            # 断言 dec 的形状为 (5,)
            assert dec.shape == (5,)

        # 如果需要测试预测概率
        if test_predict_proba:
            # 定义测试数据 X_test
            X_test = np.array([[0, 0, 4]])
            # 获取 X_test 的预测概率
            probabilities = clf.predict_proba(X_test)
            # 断言概率数组的长度为 2
            assert 2 == len(probabilities[0])
            # 断言预测概率最高的类别与 clf.predict(X_test) 的结果相同
            assert clf.classes_[np.argmax(probabilities, axis=1)] == clf.predict(X_test)

        # 测试将输入作为标签指示矩阵的情况
        # 使用 OneVsRestClassifier 将 base_clf 转化为多类分类器，并在训练集 X, Y 上拟合
        clf = OneVsRestClassifier(base_clf).fit(X, Y)
        # 对输入 [[3, 0, 0]] 进行预测
        y_pred = clf.predict([[3, 0, 0]])[0]
        # 断言预测结果为 1
        assert y_pred == 1

    # 对于给定的分类器列表，依次调用 conduct_test 函数
    for base_clf in (
        LinearSVC(random_state=0),
        LinearRegression(),
        Ridge(),
        ElasticNet(),
    ):
        conduct_test(base_clf)

    # 对于给定的分类器列表，依次调用 conduct_test 函数，并测试预测概率
    for base_clf in (MultinomialNB(), SVC(probability=True), LogisticRegression()):
        conduct_test(base_clf, test_predict_proba=True)
# 测试多标签分类的功能

def test_ovr_multilabel():
    # 定义一个玩具数据集，特征直接对应标签
    X = np.array([[0, 4, 5], [0, 5, 0], [3, 3, 3], [4, 0, 6], [6, 0, 0]])
    y = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]])

    # 对每个基分类器进行测试
    for base_clf in (
        MultinomialNB(),
        LinearSVC(random_state=0),
        LinearRegression(),
        Ridge(),
        ElasticNet(),
        Lasso(alpha=0.5),
    ):
        # 创建一个 OneVsRestClassifier 对象并拟合数据
        clf = OneVsRestClassifier(base_clf).fit(X, y)
        
        # 预测一个新的样本
        y_pred = clf.predict([[0, 4, 4]])[0]
        
        # 断言预测结果与期望结果相等
        assert_array_equal(y_pred, [0, 1, 1])
        
        # 断言分类器支持多标签分类
        assert clf.multilabel_


def test_ovr_fit_predict_svc():
    # 创建一个 OneVsRestClassifier 对象，使用 SVM 作为基分类器
    ovr = OneVsRestClassifier(svm.SVC())
    
    # 对鸢尾花数据集进行拟合
    ovr.fit(iris.data, iris.target)
    
    # 断言估计器的数量等于标签的数量
    assert len(ovr.estimators_) == 3
    
    # 断言分类器在数据集上的得分大于 0.9
    assert ovr.score(iris.data, iris.target) > 0.9


def test_ovr_multilabel_dataset():
    # 创建一个 MultinomialNB 的基分类器
    base_clf = MultinomialNB(alpha=1)
    
    # 对多标签分类数据集进行迭代测试
    for au, prec, recall in zip((True, False), (0.51, 0.66), (0.51, 0.80)):
        # 生成多标签分类数据集
        X, Y = datasets.make_multilabel_classification(
            n_samples=100,
            n_features=20,
            n_classes=5,
            n_labels=2,
            length=50,
            allow_unlabeled=au,
            random_state=0,
        )
        
        # 划分训练集和测试集
        X_train, Y_train = X[:80], Y[:80]
        X_test, Y_test = X[80:], Y[80:]
        
        # 创建一个 OneVsRestClassifier 对象并拟合训练集
        clf = OneVsRestClassifier(base_clf).fit(X_train, Y_train)
        
        # 对测试集进行预测
        Y_pred = clf.predict(X_test)

        # 断言分类器支持多标签分类
        assert clf.multilabel_
        
        # 断言精确度满足预期值
        assert_almost_equal(
            precision_score(Y_test, Y_pred, average="micro"), prec, decimal=2
        )
        
        # 断言召回率满足预期值
        assert_almost_equal(
            recall_score(Y_test, Y_pred, average="micro"), recall, decimal=2
        )


def test_ovr_multilabel_predict_proba():
    # 创建一个 MultinomialNB 的基分类器
    base_clf = MultinomialNB(alpha=1)
    # 循环遍历两次，分别对应 allow_unlabeled 为 False 和 True 的情况
    for au in (False, True):
        # 使用 make_multilabel_classification 生成多标签分类数据集
        X, Y = datasets.make_multilabel_classification(
            n_samples=100,          # 样本数
            n_features=20,          # 特征数
            n_classes=5,            # 类别数
            n_labels=3,             # 每个样本的标签数
            length=50,              # 每个样本的平均标签长度
            allow_unlabeled=au,     # 是否允许有未标记的样本
            random_state=0,         # 随机数种子，用于重现结果
        )
        # 划分训练集和测试集
        X_train, Y_train = X[:80], Y[:80]
        X_test = X[80:]
        
        # 使用 OneVsRestClassifier 对基础分类器进行多标签分类训练
        clf = OneVsRestClassifier(base_clf).fit(X_train, Y_train)

        # 创建一个只有 decision_function 方法的估算器
        decision_only = OneVsRestClassifier(svm.SVR()).fit(X_train, Y_train)
        # 断言 decision_only 没有 predict_proba 方法
        assert not hasattr(decision_only, "predict_proba")

        # 根据参数决定是否禁用 predict_proba 的估算器
        decision_only = OneVsRestClassifier(svm.SVC(probability=False))
        assert not hasattr(decision_only, "predict_proba")
        decision_only.fit(X_train, Y_train)
        assert not hasattr(decision_only, "predict_proba")
        assert hasattr(decision_only, "decision_function")

        # 在拟合后可以启用 predict_proba 的估算器
        gs = GridSearchCV(
            svm.SVC(probability=False),  # 使用 SVM 分类器，禁用概率估算
            param_grid={"probability": [True]}  # 参数网格，用于启用概率估算
        )
        proba_after_fit = OneVsRestClassifier(gs)
        assert not hasattr(proba_after_fit, "predict_proba")
        proba_after_fit.fit(X_train, Y_train)
        assert hasattr(proba_after_fit, "predict_proba")

        # 使用训练好的分类器进行预测和概率估算
        Y_pred = clf.predict(X_test)
        Y_proba = clf.predict_proba(X_test)

        # 预测结果为概率大于 0.5 的标签
        pred = Y_proba > 0.5
        assert_array_equal(pred, Y_pred)
# 测试单标签预测概率的 OneVsRestClassifier

def test_ovr_single_label_predict_proba():
    # 创建一个带有 Laplace 平滑的 MultinomialNB 分类器
    base_clf = MultinomialNB(alpha=1)
    # 载入鸢尾花数据集的特征和标签
    X, Y = iris.data, iris.target
    # 取前 80 个样本作为训练集
    X_train, Y_train = X[:80], Y[:80]
    # 取第 80 个之后的样本作为测试集
    X_test = X[80:]
    # 创建 OneVsRestClassifier 对象，使用 MultinomialNB 分类器进行训练
    clf = OneVsRestClassifier(base_clf).fit(X_train, Y_train)

    # 创建只包含决策函数的估计器
    decision_only = OneVsRestClassifier(svm.SVR()).fit(X_train, Y_train)
    # 断言 decision_only 对象没有 predict_proba 方法
    assert not hasattr(decision_only, "predict_proba")

    # 对测试集进行预测
    Y_pred = clf.predict(X_test)
    # 获取测试集样本的预测概率
    Y_proba = clf.predict_proba(X_test)

    # 断言每个样本的预测概率之和为 1.0
    assert_almost_equal(Y_proba.sum(axis=1), 1.0)
    # 根据最大预测概率分配标签
    pred = Y_proba.argmax(axis=1)
    # 断言预测的标签与 Y_pred 一致
    assert not (pred - Y_pred).any()


# 测试多标签决策函数的 OneVsRestClassifier

def test_ovr_multilabel_decision_function():
    # 生成多标签分类的合成数据集
    X, Y = datasets.make_multilabel_classification(
        n_samples=100,
        n_features=20,
        n_classes=5,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    # 取前 80 个样本作为训练集
    X_train, Y_train = X[:80], Y[:80]
    # 取第 80 个之后的样本作为测试集
    X_test = X[80:]
    # 创建 OneVsRestClassifier 对象，使用 SVM 分类器进行训练
    clf = OneVsRestClassifier(svm.SVC()).fit(X_train, Y_train)
    # 断言决策函数的输出是否与预测结果一致
    assert_array_equal(
        (clf.decision_function(X_test) > 0).astype(int), clf.predict(X_test)
    )


# 测试单标签决策函数的 OneVsRestClassifier

def test_ovr_single_label_decision_function():
    # 生成合成数据集
    X, Y = datasets.make_classification(n_samples=100, n_features=20, random_state=0)
    # 取前 80 个样本作为训练集
    X_train, Y_train = X[:80], Y[:80]
    # 取第 80 个之后的样本作为测试集
    X_test = X[80:]
    # 创建 OneVsRestClassifier 对象，使用 SVM 分类器进行训练
    clf = OneVsRestClassifier(svm.SVC()).fit(X_train, Y_train)
    # 断言决策函数的输出是否与预测结果一致
    assert_array_equal(clf.decision_function(X_test).ravel() > 0, clf.predict(X_test))


# 测试 OneVsRestClassifier 的网格搜索

def test_ovr_gridsearch():
    # 创建一个线性支持向量分类器的 OneVsRestClassifier
    ovr = OneVsRestClassifier(LinearSVC(random_state=0))
    # 定义参数 C 的候选值列表
    Cs = [0.1, 0.5, 0.8]
    # 创建网格搜索对象，搜索最佳的参数组合
    cv = GridSearchCV(ovr, {"estimator__C": Cs})
    # 在鸢尾花数据集上进行网格搜索
    cv.fit(iris.data, iris.target)
    # 获取最佳的 C 值
    best_C = cv.best_estimator_.estimators_[0].C
    # 断言最佳的 C 值在候选列表中
    assert best_C in Cs


# 测试带有管道的 OneVsRestClassifier

def test_ovr_pipeline():
    # 创建只包含决策树分类器的管道
    clf = Pipeline([("tree", DecisionTreeClassifier())])
    # 创建 OneVsRestClassifier 对象，使用 clf 管道进行训练
    ovr_pipe = OneVsRestClassifier(clf)
    ovr_pipe.fit(iris.data, iris.target)
    # 创建只包含决策树分类器的 OneVsRestClassifier 对象，直接训练
    ovr = OneVsRestClassifier(DecisionTreeClassifier())
    ovr.fit(iris.data, iris.target)
    # 断言两种方式得到的预测结果一致
    assert_array_equal(ovr.predict(iris.data), ovr_pipe.predict(iris.data))


# 测试 OneVsOneClassifier 的异常情况

def test_ovo_exceptions():
    # 创建 OneVsOneClassifier 对象，使用线性支持向量分类器进行训练
    ovo = OneVsOneClassifier(LinearSVC(random_state=0))
    # 断言对未拟合的模型进行预测会抛出 NotFittedError 异常
    with pytest.raises(NotFittedError):
        ovo.predict([])


# 测试 OneVsOneClassifier 在列表数据上的拟合

def test_ovo_fit_on_list():
    # 创建 OneVsOneClassifier 对象，使用线性支持向量分类器进行训练
    ovo = OneVsOneClassifier(LinearSVC(random_state=0))
    # 从数组形式的数据进行拟合和预测
    prediction_from_array = ovo.fit(iris.data, iris.target).predict(iris.data)
    # 将鸢尾花数据集转换为列表形式
    iris_data_list = [list(a) for a in iris.data]
    # 从列表形式的数据进行拟合和预测
    prediction_from_list = ovo.fit(iris_data_list, list(iris.target)).predict(
        iris_data_list
    )
    # 使用断言检查两个数组或列表的内容是否完全相等
    assert_array_equal(prediction_from_array, prediction_from_list)
def test_ovo_fit_predict():
    # 使用 LinearSVC 实现的 OneVsOne 分类器
    ovo = OneVsOneClassifier(LinearSVC(random_state=0))
    # 使用 iris 数据训练分类器，并预测结果
    ovo.fit(iris.data, iris.target).predict(iris.data)
    # 检查分类器中的估计器数量是否符合 OneVsOne 的要求
    assert len(ovo.estimators_) == n_classes * (n_classes - 1) / 2

    # 使用 MultinomialNB 实现的 OneVsOne 分类器
    ovo = OneVsOneClassifier(MultinomialNB())
    # 使用 iris 数据训练分类器，并预测结果
    ovo.fit(iris.data, iris.target).predict(iris.data)
    # 检查分类器中的估计器数量是否符合 OneVsOne 的要求
    assert len(ovo.estimators_) == n_classes * (n_classes - 1) / 2


def test_ovo_partial_fit_predict():
    # 加载 iris 数据集
    temp = datasets.load_iris()
    X, y = temp.data, temp.target

    # 使用 MultinomialNB 实现的 OneVsOne 分类器，进行部分拟合
    ovo1 = OneVsOneClassifier(MultinomialNB())
    ovo1.partial_fit(X[:100], y[:100], np.unique(y))
    ovo1.partial_fit(X[100:], y[100:])
    pred1 = ovo1.predict(X)

    # 使用 MultinomialNB 实现的 OneVsOne 分类器，进行完全拟合
    ovo2 = OneVsOneClassifier(MultinomialNB())
    ovo2.fit(X, y)
    pred2 = ovo2.predict(X)
    # 检查分类器中的估计器数量是否符合 OneVsOne 的要求
    assert len(ovo1.estimators_) == n_classes * (n_classes - 1) / 2
    # 检查预测精度是否大于 0.65
    assert np.mean(y == pred1) > 0.65
    # 检查两种不同拟合方式得到的预测结果是否近似相等
    assert_almost_equal(pred1, pred2)

    # 测试当小批量拥有二元目标类别时
    ovo1 = OneVsOneClassifier(MultinomialNB())
    ovo1.partial_fit(X[:60], y[:60], np.unique(y))
    ovo1.partial_fit(X[60:], y[60:])
    pred1 = ovo1.predict(X)
    ovo2 = OneVsOneClassifier(MultinomialNB())
    pred2 = ovo2.fit(X, y).predict(X)
    # 检查两种不同拟合方式得到的预测结果是否近似相等
    assert_almost_equal(pred1, pred2)
    # 检查分类器中的估计器数量是否等于类别数
    assert len(ovo1.estimators_) == len(np.unique(y))
    # 检查预测精度是否大于 0.65
    assert np.mean(y == pred1) > 0.65

    # 部分拟合，随机数据 X 和 y
    ovo = OneVsOneClassifier(MultinomialNB())
    X = np.random.rand(14, 2)
    y = [1, 1, 2, 3, 3, 0, 0, 4, 4, 4, 4, 4, 2, 2]
    ovo.partial_fit(X[:7], y[:7], [0, 1, 2, 3, 4])
    ovo.partial_fit(X[7:], y[7:])
    pred = ovo.predict(X)
    ovo2 = OneVsOneClassifier(MultinomialNB())
    pred2 = ovo2.fit(X, y).predict(X)
    # 检查两种不同拟合方式得到的预测结果是否近似相等
    assert_almost_equal(pred, pred2)

    # 当小批量数据不包含全部类别时，应该引发错误
    ovo = OneVsOneClassifier(MultinomialNB())
    error_y = [0, 1, 2, 3, 4, 5, 2]
    message_re = escape(
        "Mini-batch contains {0} while it must be subset of {1}".format(
            np.unique(error_y), np.unique(y)
        )
    )
    # 检查是否引发 ValueError 错误，并检查错误消息是否匹配预期
    with pytest.raises(ValueError, match=message_re):
        ovo.partial_fit(X[:7], error_y, np.unique(y))

    # 测试当估计器不支持 partial_fit 时
    ovr = OneVsOneClassifier(SVC())
    # 检查是否没有 partial_fit 属性
    assert not hasattr(ovr, "partial_fit")


def test_ovo_decision_function():
    n_samples = iris.data.shape[0]

    # 创建 LinearSVC 实现的 OneVsOne 分类器
    ovo_clf = OneVsOneClassifier(LinearSVC(random_state=0))
    # 以二元方式拟合 iris 数据集
    ovo_clf.fit(iris.data, iris.target == 0)
    decisions = ovo_clf.decision_function(iris.data)
    # 检查决策函数输出的形状
    assert decisions.shape == (n_samples,)

    # 以多类方式拟合 iris 数据集
    ovo_clf.fit(iris.data, iris.target)
    decisions = ovo_clf.decision_function(iris.data)
    # 检查决策函数输出的形状
    assert decisions.shape == (n_samples, n_classes)
    # 检查决策函数的最大投票类别是否与预测结果一致
    assert_array_equal(decisions.argmax(axis=1), ovo_clf.predict(iris.data))

    # 计算投票结果
    votes = np.zeros((n_samples, n_classes))

    k = 0
    # 遍历所有可能的类别对，进行一对一（OvO）分类
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # 使用第 k 个分类器对整个 iris 数据集进行预测
            pred = ovo_clf.estimators_[k].predict(iris.data)
            # 根据预测结果更新投票统计：
            # 如果预测为类别 0，则在第 i 类别的投票数加一
            votes[pred == 0, i] += 1
            # 如果预测为类别 1，则在第 j 类别的投票数加一
            votes[pred == 1, j] += 1
            # 切换到下一个分类器
            k += 1

    # 提取投票结果并验证其与决策结果的一致性
    assert_array_equal(votes, np.round(decisions))

    # 对每个类别进行验证：
    for class_idx in range(n_classes):
        # 每个样本对于每个类别的投票仅可能有 3 种可能的投票结果，
        # 因为只有 3 对不同的类别组合，因此只有 3 种不同的二元分类器。
        # 因此，基于投票结果排序预测会得到大多数情况下的平局预测：
        assert set(votes[:, class_idx]).issubset(set([0.0, 1.0, 2.0]))

        # 另一方面，OvO 决策函数能够解决大部分的平局情况，
        # 因为它结合了投票计数和二元分类器的聚合置信水平来计算聚合决策函数。
        # Iris 数据集包含 150 个样本，有一些重复。OvO 决策能够解决大部分的平局情况：
        assert len(np.unique(decisions[:, class_idx])) > 146
def test_ovo_gridsearch():
    # 使用线性支持向量机作为基础分类器进行一对一分类器的初始化
    ovo = OneVsOneClassifier(LinearSVC(random_state=0))
    # 定义参数候选列表
    Cs = [0.1, 0.5, 0.8]
    # 使用网格搜索进行参数优化
    cv = GridSearchCV(ovo, {"estimator__C": Cs})
    # 在iris数据集上进行训练
    cv.fit(iris.data, iris.target)
    # 获取最佳分类器的参数C值
    best_C = cv.best_estimator_.estimators_[0].C
    # 断言最佳参数C值在候选列表中
    assert best_C in Cs


def test_ovo_ties():
    # 测试使用决策函数来解决平局情况，而不是默认选择最小的标签
    X = np.array([[1, 2], [2, 1], [-2, 1], [-2, -1]])
    y = np.array([2, 0, 1, 2])
    # 初始化一对一分类器，基础分类器为感知机
    multi_clf = OneVsOneClassifier(Perceptron(shuffle=False, max_iter=4, tol=None))
    # 使用训练数据进行拟合和预测
    ovo_prediction = multi_clf.fit(X, y).predict(X)
    # 获取决策函数的输出值
    ovo_decision = multi_clf.decision_function(X)

    # 按照顺序0-1, 0-2, 1-2，使用决策函数计算票数和标准化置信度之和来消除平局
    votes = np.round(ovo_decision)
    normalized_confidences = ovo_decision - votes

    # 对于第一个数据点，每个类别有一票
    assert_array_equal(votes[0, :], 1)
    # 对于其余数据点，没有平局，预测结果为argmax
    assert_array_equal(np.argmax(votes[1:], axis=1), ovo_prediction[1:])
    # 对于平局的情况，预测结果应为具有最高分数的类别
    assert ovo_prediction[0] == normalized_confidences[0].argmax()


def test_ovo_ties2():
    # 测试平局情况下不仅仅是前两个标签能够获胜
    X = np.array([[1, 2], [2, 1], [-2, 1], [-2, -1]])
    y_ref = np.array([2, 0, 1, 2])

    # 循环标签，以便每个标签都能胜出一次
    for i in range(3):
        y = (y_ref + i) % 3
        # 初始化一对一分类器，基础分类器为感知机
        multi_clf = OneVsOneClassifier(Perceptron(shuffle=False, max_iter=4, tol=None))
        # 使用训练数据进行拟合和预测
        ovo_prediction = multi_clf.fit(X, y).predict(X)
        # 断言第一个数据点的预测结果
        assert ovo_prediction[0] == i % 3


def test_ovo_string_y():
    # 测试一对一分类器对字符串标签的编码不会出错
    X = np.eye(4)
    y = np.array(["a", "b", "c", "d"])

    # 初始化一对一分类器，基础分类器为线性支持向量机
    ovo = OneVsOneClassifier(LinearSVC())
    # 使用训练数据进行拟合
    ovo.fit(X, y)
    # 断言预测结果与原始标签一致
    assert_array_equal(y, ovo.predict(X))


def test_ovo_one_class():
    # 测试一对一分类器处理只有一个类别的情况会报错
    X = np.eye(4)
    y = np.array(["a"] * 4)

    # 初始化一对一分类器，基础分类器为线性支持向量机
    ovo = OneVsOneClassifier(LinearSVC())
    msg = "when only one class"
    # 使用pytest断言捕获异常并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        ovo.fit(X, y)


def test_ovo_float_y():
    # 测试一对一分类器对浮点数标签会报错
    X = iris.data
    y = iris.data[:, 0]

    # 初始化一对一分类器，基础分类器为线性支持向量机
    ovo = OneVsOneClassifier(LinearSVC())
    msg = "Unknown label type"
    # 使用pytest断言捕获异常并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        ovo.fit(X, y)


def test_ecoc_exceptions():
    # 测试输出码分类器在未拟合情况下预测会抛出异常
    ecoc = OutputCodeClassifier(LinearSVC(random_state=0))
    # 使用pytest断言捕获预期异常
    with pytest.raises(NotFittedError):
        ecoc.predict([])


def test_ecoc_fit_predict():
    # 测试实现了decision_function的分类器
    ecoc = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
    # 使用iris数据集进行拟合和预测
    ecoc.fit(iris.data, iris.target).predict(iris.data)
    # 断言：确保 OutputCodeClassifier 中的 estimators_ 数量等于类别数乘以 2
    assert len(ecoc.estimators_) == n_classes * 2
    
    # 创建一个实现了 predict_proba 的分类器
    ecoc = OutputCodeClassifier(MultinomialNB(), code_size=2, random_state=0)
    # 使用 iris 数据集训练该 OutputCodeClassifier，并对 iris 数据集进行预测
    ecoc.fit(iris.data, iris.target).predict(iris.data)
    # 断言：再次确保 OutputCodeClassifier 中的 estimators_ 数量等于类别数乘以 2
    assert len(ecoc.estimators_) == n_classes * 2
def test_ecoc_gridsearch():
    # 创建一个 OutputCodeClassifier 对象，使用 LinearSVC 作为基分类器，设置随机种子为0
    ecoc = OutputCodeClassifier(LinearSVC(random_state=0), random_state=0)
    # 定义待搜索的参数列表 Cs
    Cs = [0.1, 0.5, 0.8]
    # 创建一个 GridSearchCV 对象，使用 ecoc 作为分类器，搜索最佳的 estimator__C 参数
    cv = GridSearchCV(ecoc, {"estimator__C": Cs})
    # 在 iris 数据集上进行拟合
    cv.fit(iris.data, iris.target)
    # 获取最佳模型的第一个分类器的 C 参数值
    best_C = cv.best_estimator_.estimators_[0].C
    # 断言最佳的 C 值在 Cs 列表中
    assert best_C in Cs


def test_ecoc_float_y():
    # 测试 OCC 是否能够正确处理浮点数标签
    X = iris.data
    y = iris.data[:, 0]

    # 创建一个 OutputCodeClassifier 对象
    ovo = OutputCodeClassifier(LinearSVC())
    # 定义预期的错误消息
    msg = "Unknown label type"
    # 使用 pytest 检查是否会引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        ovo.fit(X, y)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_ecoc_delegate_sparse_base_estimator(csc_container):
    # 非回归测试，用于检查稀疏基分类器的问题修复
    X, y = iris.data, iris.target
    # 将 X 转换为稀疏格式
    X_sp = csc_container(X)

    # 创建一个不支持稀疏输入的基分类器
    base_estimator = CheckingClassifier(
        check_X=check_array,
        check_X_params={"ensure_2d": True, "accept_sparse": False},
    )
    # 创建 OutputCodeClassifier 对象，使用上面定义的 base_estimator
    ecoc = OutputCodeClassifier(base_estimator, random_state=0)

    # 使用 pytest 检查是否会引发 TypeError 异常，并匹配预期的错误消息
    with pytest.raises(TypeError, match="Sparse data was passed"):
        ecoc.fit(X_sp, y)

    # 在非稀疏数据上拟合 ecoc 对象
    ecoc.fit(X, y)
    # 使用 pytest 检查是否会引发 TypeError 异常，并匹配预期的错误消息
    with pytest.raises(TypeError, match="Sparse data was passed"):
        ecoc.predict(X_sp)

    # 烟雾测试，验证在支持稀疏输入时是否能正常工作
    ecoc = OutputCodeClassifier(LinearSVC(random_state=0))
    ecoc.fit(X_sp, y).predict(X_sp)
    # 断言 ecoc.estimators_ 的长度为 4
    assert len(ecoc.estimators_) == 4


def test_pairwise_indices():
    # 使用预计算的核进行 SVC 分类器的测试
    clf_precomputed = svm.SVC(kernel="precomputed")
    X, y = iris.data, iris.target

    # 创建一个 OneVsOneClassifier 对象
    ovr_false = OneVsOneClassifier(clf_precomputed)
    # 计算线性核
    linear_kernel = np.dot(X, X.T)
    # 在线性核上拟合 OneVsOneClassifier 对象
    ovr_false.fit(linear_kernel, y)

    # 获取分类器数量
    n_estimators = len(ovr_false.estimators_)
    # 获取配对指数
    precomputed_indices = ovr_false.pairwise_indices_

    # 遍历 precomputed_indices，断言其形状符合预期
    for idx in precomputed_indices:
        assert (
            idx.shape[0] * n_estimators / (n_estimators - 1) == linear_kernel.shape[0]
        )


def test_pairwise_n_features_in():
    """Check the n_features_in_ attributes of the meta and base estimators

    When the training data is a regular design matrix, everything is intuitive.
    However, when the training data is a precomputed kernel matrix, the
    multiclass strategy can resample the kernel matrix of the underlying base
    estimator both row-wise and column-wise and this has a non-trivial impact
    on the expected value for the n_features_in_ of both the meta and the base
    estimators.
    """
    X, y = iris.data, iris.target

    # 移除最后一个样本以使类别不完全平衡，并使测试更加有趣
    assert y[-1] == 0
    X = X[:-1]
    y = y[:-1]

    # 直接在设计矩阵上拟合时，验证 X 的形状为 (149, 4)
    assert X.shape == (149, 4)

    # 使用线性核的 SVC 拟合 clf_notprecomputed 对象
    clf_notprecomputed = svm.SVC(kernel="linear").fit(X, y)
    # 断言 clf_notprecomputed 的 n_features_in_ 属性为 4
    assert clf_notprecomputed.n_features_in_ == 4

    # 使用 OneVsRestClassifier 在 clf_notprecomputed 上拟合 ovr_notprecomputed 对象
    ovr_notprecomputed = OneVsRestClassifier(clf_notprecomputed).fit(X, y)
    # 确保未预计算模型的输入特征数为4
    assert ovr_notprecomputed.n_features_in_ == 4
    # 确保每个二元分类器的输入特征数为4
    for est in ovr_notprecomputed.estimators_:
        assert est.n_features_in_ == 4

    # 使用OneVsOneClassifier对未预计算的分类器进行拟合
    ovo_notprecomputed = OneVsOneClassifier(clf_notprecomputed).fit(X, y)
    # 确保OneVsOneClassifier拟合后的输入特征数为4
    assert ovo_notprecomputed.n_features_in_ == 4
    # 确保OneVsOneClassifier拟合后的类别数为3
    assert ovo_notprecomputed.n_classes_ == 3
    # 确保OneVsOneClassifier拟合后的分类器数量为3
    assert len(ovo_notprecomputed.estimators_) == 3
    # 确保每个二元分类器的输入特征数为4
    for est in ovo_notprecomputed.estimators_:
        assert est.n_features_in_ == 4

    # 使用预计算的核矩阵进行SVM分类器的拟合
    K = X @ X.T
    # 确保核矩阵的形状为(149, 149)
    assert K.shape == (149, 149)

    # 使用SVM分类器（核为预计算的）对核矩阵进行拟合
    clf_precomputed = svm.SVC(kernel="precomputed").fit(K, y)
    # 确保SVM分类器拟合后的输入特征数为149
    assert clf_precomputed.n_features_in_ == 149

    # 使用OneVsRestClassifier对预计算核的SVM分类器进行拟合
    ovr_precomputed = OneVsRestClassifier(clf_precomputed).fit(K, y)
    # 确保OneVsRestClassifier拟合后的输入特征数为149
    assert ovr_precomputed.n_features_in_ == 149
    # 确保OneVsRestClassifier拟合后的类别数为3
    assert ovr_precomputed.n_classes_ == 3
    # 确保OneVsRestClassifier拟合后的分类器数量为3
    assert len(ovr_precomputed.estimators_) == 3
    # 确保每个二元分类器的输入特征数为149
    for est in ovr_precomputed.estimators_:
        assert est.n_features_in_ == 149

    # 对于OvO和预计算核一起使用的情况尤为有趣：
    # 内部上，对于每个二元分类器，OvO会丢弃不涉及的类别样本。由于我们使用了预计算的核，
    # 它还会删除核矩阵中匹配的列，因此结果中的特征数会减少。
    #
    # 由于类别0有49个样本，类别1和2各有50个样本，单个OvO二元分类器会使用形状为(99, 99)或(100, 100)的子核矩阵。
    # 使用OneVsOneClassifier对预计算核的SVM分类器进行拟合
    ovo_precomputed = OneVsOneClassifier(clf_precomputed).fit(K, y)
    # 确保OneVsOneClassifier拟合后的输入特征数为149
    assert ovo_precomputed.n_features_in_ == 149
    # 确保OneVsOneClassifier拟合后的类别数为3
    assert ovr_precomputed.n_classes_ == 3
    # 确保OneVsOneClassifier拟合后的分类器数量为3
    assert len(ovr_precomputed.estimators_) == 3
    # 确保类别0 vs 类别1的二元分类器输入特征数为99
    assert ovo_precomputed.estimators_[0].n_features_in_ == 99
    # 确保类别0 vs 类别2的二元分类器输入特征数为99
    assert ovo_precomputed.estimators_[1].n_features_in_ == 99
    # 确保类别1 vs 类别2的二元分类器输入特征数为100
    assert ovo_precomputed.estimators_[2].n_features_in_ == 100
@pytest.mark.parametrize(
    "MultiClassClassifier", [OneVsRestClassifier, OneVsOneClassifier]
)
# 定义测试函数test_pairwise_tag，使用pytest的参数化功能分别测试OneVsRestClassifier和OneVsOneClassifier
def test_pairwise_tag(MultiClassClassifier):
    # 创建一个使用预计算核的SVM分类器对象
    clf_precomputed = svm.SVC(kernel="precomputed")
    # 创建一个未使用预计算核的SVM分类器对象
    clf_notprecomputed = svm.SVC()

    # 使用MultiClassClassifier构造一个OneVsRest分类器对象
    ovr_false = MultiClassClassifier(clf_notprecomputed)
    # 断言ovr_false对象的_get_tags()方法返回的字典中不包含"pairwise"键
    assert not ovr_false._get_tags()["pairwise"]

    # 使用MultiClassClassifier构造一个OneVsRest分类器对象，使用预计算核的SVM分类器
    ovr_true = MultiClassClassifier(clf_precomputed)
    # 断言ovr_true对象的_get_tags()方法返回的字典中包含"pairwise"键
    assert ovr_true._get_tags()["pairwise"]


@pytest.mark.parametrize(
    "MultiClassClassifier", [OneVsRestClassifier, OneVsOneClassifier]
)
# 定义测试函数test_pairwise_cross_val_score，使用pytest的参数化功能分别测试OneVsRestClassifier和OneVsOneClassifier
def test_pairwise_cross_val_score(MultiClassClassifier):
    # 创建一个使用预计算核的SVM分类器对象
    clf_precomputed = svm.SVC(kernel="precomputed")
    # 创建一个使用线性核的SVM分类器对象
    clf_notprecomputed = svm.SVC(kernel="linear")

    # 加载鸢尾花数据集的特征和目标值
    X, y = iris.data, iris.target

    # 使用MultiClassClassifier构造一个使用线性核的OneVsRest分类器对象
    multiclass_clf_notprecomputed = MultiClassClassifier(clf_notprecomputed)
    # 使用MultiClassClassifier构造一个使用预计算核的OneVsRest分类器对象
    multiclass_clf_precomputed = MultiClassClassifier(clf_precomputed)

    # 计算特征X的线性核矩阵
    linear_kernel = np.dot(X, X.T)
    # 使用交叉验证评估使用线性核分类器的性能
    score_not_precomputed = cross_val_score(
        multiclass_clf_notprecomputed, X, y, error_score="raise"
    )
    # 使用交叉验证评估使用预计算核分类器的性能
    score_precomputed = cross_val_score(
        multiclass_clf_precomputed, linear_kernel, y, error_score="raise"
    )
    # 断言预计算核分类器和线性核分类器的交叉验证得分数组相等
    assert_array_equal(score_precomputed, score_not_precomputed)


@pytest.mark.parametrize(
    "MultiClassClassifier", [OneVsRestClassifier, OneVsOneClassifier]
)
# FIXME: we should move this test in `estimator_checks` once we are able
# to construct meta-estimator instances
# 定义测试函数test_support_missing_values，使用pytest的参数化功能分别测试OneVsRestClassifier和OneVsOneClassifier
def test_support_missing_values(MultiClassClassifier):
    # 创建随机数生成器对象
    rng = np.random.RandomState(42)
    # 加载鸢尾花数据集的特征和目标值
    X, y = iris.data, iris.target
    # 复制特征数据X，避免修改原始数据
    X = np.copy(X)
    # 创建一个掩码，模拟缺失值情况
    mask = rng.choice([1, 0], X.shape, p=[0.1, 0.9]).astype(bool)
    X[mask] = np.nan
    # 创建一个包含简单填充器和逻辑回归的管道对象
    lr = make_pipeline(SimpleImputer(), LogisticRegression(random_state=rng))

    # 使用MultiClassClassifier构造一个分类器对象，用于拟合X和y，并计算分数
    MultiClassClassifier(lr).fit(X, y).score(X, y)


@pytest.mark.parametrize("make_y", [np.ones, np.zeros])
# 定义测试函数test_constant_int_target，使用参数化功能测试常数目标值的情况
def test_constant_int_target(make_y):
    """Check that constant y target does not raise.

    Non-regression test for #21869
    """
    # 创建一个全为1的特征矩阵X
    X = np.ones((10, 2))
    # 创建一个全为0或全为1的目标值y
    y = make_y((10, 1), dtype=np.int32)
    # 创建一个逻辑回归作为基本分类器的OneVsRest分类器对象
    ovr = OneVsRestClassifier(LogisticRegression())

    # 使用X和y拟合分类器
    ovr.fit(X, y)
    # 预测X的概率
    y_pred = ovr.predict_proba(X)
    # 创建一个期望的概率矩阵
    expected = np.zeros((X.shape[0], 2))
    expected[:, 0] = 1
    # 断言预测的概率与期望的概率矩阵相等
    assert_allclose(y_pred, expected)


# 定义测试函数test_ovo_consistent_binary_classification，检查OvO分类器在二元分类问题上的一致性
def test_ovo_consistent_binary_classification():
    """Check that ovo is consistent with binary classifier.

    Non-regression test for #13617.
    """
    # 加载乳腺癌数据集的特征和目标值
    X, y = load_breast_cancer(return_X_y=True)

    # 创建一个K最近邻分类器对象
    clf = KNeighborsClassifier(n_neighbors=8, weights="distance")
    # 创建一个OvO分类器对象，使用K最近邻分类器作为基础分类器
    ovo = OneVsOneClassifier(clf)

    # 使用原始数据拟合K最近邻分类器和OvO分类器
    clf.fit(X, y)
    ovo.fit(X, y)

    # 断言使用K最近邻分类器和OvO分类器预测的结果相等
    assert_array_equal(clf.predict(X), ovo.predict(X))


# 定义测试函数test_multiclass_estimator_attribute_error，检查当最终估计器属性错误时是否引发正确的AttributeError异常
def test_multiclass_estimator_attribute_error():
    """Check that we raise the proper AttributeError when the final estimator
    iris = datasets.load_iris()

    # 使用 scikit-learn 提供的 load_iris() 函数加载经典的鸢尾花数据集
    # 这个数据集包含了鸢尾花的测量数据以及它们的分类标签

    clf = OneVsRestClassifier(estimator=LogisticRegression(random_state=42))

    # 创建一个 OneVsRestClassifier 分类器，使用 LogisticRegression 作为基分类器
    # LogisticRegression 并未实现 'partial_fit' 方法，因此应该会引发 AttributeError 异常

    outer_msg = "This 'OneVsRestClassifier' has no attribute 'partial_fit'"
    inner_msg = "'LogisticRegression' object has no attribute 'partial_fit'"

    # 使用 pytest 模块的 raises 函数来验证 clf.partial_fit(iris.data, iris.target) 是否抛出 AttributeError 异常
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        clf.partial_fit(iris.data, iris.target)

    # 断言异常信息的类型为 AttributeError
    assert isinstance(exec_info.value.__cause__, AttributeError)

    # 断言异常的原因字符串中应包含 inner_msg
    assert inner_msg in str(exec_info.value.__cause__)
```