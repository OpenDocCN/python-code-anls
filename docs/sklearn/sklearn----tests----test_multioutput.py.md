# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_multioutput.py`

```
# 导入必要的库
import re
import numpy as np
import pytest
from joblib import cpu_count

# 导入 sklearn 相关模块和类
from sklearn import datasets
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import (
    load_linnerud,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    StackingRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveClassifier,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import jaccard_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import (
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
    RegressorChain,
)
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import (
    BSR_CONTAINERS,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
)

# 定义测试函数：测试多目标回归
def test_multi_target_regression():
    # 生成具有三个目标变量的回归数据集
    X, y = datasets.make_regression(n_targets=3, random_state=0)
    # 划分训练集和测试集
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    # 创建一个与 y_test 维度相同的零矩阵作为参考结果
    references = np.zeros_like(y_test)
    # 对每个目标变量执行回归
    for n in range(3):
        # 使用梯度提升回归器进行训练和预测
        rgr = GradientBoostingRegressor(random_state=0)
        rgr.fit(X_train, y_train[:, n])
        references[:, n] = rgr.predict(X_test)

    # 使用多输出回归器进行训练和预测
    rgr = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    rgr.fit(X_train, y_train)
    y_pred = rgr.predict(X_test)

    # 断言预测结果与参考结果的近似性
    assert_almost_equal(references, y_pred)


# 定义测试函数：测试支持部分拟合的多目标回归
def test_multi_target_regression_partial_fit():
    # 生成具有三个目标变量的回归数据集
    X, y = datasets.make_regression(n_targets=3, random_state=0)
    # 划分训练集和测试集
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    # 创建一个与 y_test 维度相同的零矩阵作为参考结果
    references = np.zeros_like(y_test)
    half_index = 25
    # 对每个目标变量执行支持部分拟合的随机梯度下降回归
    for n in range(3):
        sgr = SGDRegressor(random_state=0, max_iter=5)
        sgr.partial_fit(X_train[:half_index], y_train[:half_index, n])
        sgr.partial_fit(X_train[half_index:], y_train[half_index:, n])
        references[:, n] = sgr.predict(X_test)

    # 使用多输出回归器进行支持部分拟合的随机梯度下降回归
    sgr = MultiOutputRegressor(SGDRegressor(random_state=0, max_iter=5))
    sgr.partial_fit(X_train[:half_index], y_train[:half_index])
    sgr.partial_fit(X_train[half_index:], y_train[half_index:])

    y_pred = sgr.predict(X_test)

    # 断言预测结果与参考结果的近似性
    assert_almost_equal(references, y_pred)
    # 断言多输出回归器不支持 Lasso 模型的部分拟合
    assert not hasattr(MultiOutputRegressor(Lasso), "partial_fit")


# 定义测试函数：测试单目标回归
def test_multi_target_regression_one_target():
    # 测试多目标回归是否会引发异常
    # 使用 sklearn.datasets 中的 make_regression 方法生成回归数据集 X 和 y
    X, y = datasets.make_regression(n_targets=1, random_state=0)
    # 创建一个 MultiOutputRegressor 对象，使用 GradientBoostingRegressor 作为基础回归器
    rgr = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    # 定义错误消息字符串
    msg = "at least two dimensions"
    # 使用 pytest 的 pytest.raises 方法来捕获 ValueError 异常，并验证其错误消息与预期相符
    with pytest.raises(ValueError, match=msg):
        # 对 MultiOutputRegressor 对象进行拟合，期望此处引发 ValueError 异常
        rgr.fit(X, y)
# 使用 pytest 的参数化装饰器标记这个测试函数，为不同的稀疏矩阵容器执行多目标稀疏回归测试
@pytest.mark.parametrize(
    "sparse_container",
    CSR_CONTAINERS           # Compressed Sparse Row (CSR) 格式的稀疏矩阵容器
    + CSC_CONTAINERS         # Compressed Sparse Column (CSC) 格式的稀疏矩阵容器
    + COO_CONTAINERS         # Coordinate (COO) 格式的稀疏矩阵容器
    + LIL_CONTAINERS         # List of Lists (LIL) 格式的稀疏矩阵容器
    + DOK_CONTAINERS         # Dictionary of Keys (DOK) 格式的稀疏矩阵容器
    + BSR_CONTAINERS,        # Block Sparse Row (BSR) 格式的稀疏矩阵容器
)
# 测试函数，用于测试多目标稀疏回归
def test_multi_target_sparse_regression(sparse_container):
    # 生成一个具有3个目标变量的回归数据集
    X, y = datasets.make_regression(n_targets=3, random_state=0)
    # 划分训练集和测试集
    X_train, y_train = X[:50], y[:50]
    X_test = X[50:]

    # 初始化多输出回归器，基础估计器为 Lasso
    rgr = MultiOutputRegressor(Lasso(random_state=0))
    rgr_sparse = MultiOutputRegressor(Lasso(random_state=0))

    # 在训练数据上拟合回归器
    rgr.fit(X_train, y_train)
    # 使用稀疏矩阵容器在训练数据上拟合稀疏多输出回归器
    rgr_sparse.fit(sparse_container(X_train), y_train)

    # 断言预测值与稀疏多输出回归器的预测值几乎相等
    assert_almost_equal(
        rgr.predict(X_test), rgr_sparse.predict(sparse_container(X_test))
    )


# 测试多目标样本权重的 API 支持情况
def test_multi_target_sample_weights_api():
    # 样本数据和目标值
    X = [[1, 2, 3], [4, 5, 6]]
    y = [[3.141, 2.718], [2.718, 3.141]]
    w = [0.8, 0.6]

    # 初始化多输出回归器，基础估计器为 OrthogonalMatchingPursuit
    rgr = MultiOutputRegressor(OrthogonalMatchingPursuit())
    msg = "does not support sample weights"
    # 使用 pytest 的断言上下文来验证是否会抛出 ValueError 异常，并匹配特定消息
    with pytest.raises(ValueError, match=msg):
        rgr.fit(X, y, w)

    # 如果基础估计器支持样本权重，则不应该引发异常
    rgr = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    rgr.fit(X, y, w)


# 测试使用部分拟合的多目标样本权重
def test_multi_target_sample_weight_partial_fit():
    # 加权回归器
    X = [[1, 2, 3], [4, 5, 6]]
    y = [[3.141, 2.718], [2.718, 3.141]]
    w = [2.0, 1.0]
    rgr_w = MultiOutputRegressor(SGDRegressor(random_state=0, max_iter=5))
    # 使用部分拟合方法拟合加权回归器
    rgr_w.partial_fit(X, y, w)

    # 使用不同权重的加权回归器
    w = [2.0, 2.0]
    rgr = MultiOutputRegressor(SGDRegressor(random_state=0, max_iter=5))
    # 使用部分拟合方法拟合加权回归器
    rgr.partial_fit(X, y, w)

    # 断言两个回归器在预测第一个样本的第一个目标变量时不相等
    assert rgr.predict(X)[0][0] != rgr_w.predict(X)[0][0]


# 测试多目标样本权重的影响
def test_multi_target_sample_weights():
    # 加权回归器
    Xw = [[1, 2, 3], [4, 5, 6]]
    yw = [[3.141, 2.718], [2.718, 3.141]]
    w = [2.0, 1.0]
    rgr_w = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    # 使用加权数据拟合加权回归器
    rgr_w.fit(Xw, yw, w)

    # 使用未加权数据，但有重复样本的数据拟合回归器
    X = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]
    y = [[3.141, 2.718], [3.141, 2.718], [2.718, 3.141]]
    rgr = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    rgr.fit(X, y)

    # 测试数据集
    X_test = [[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]
    # 断言两个回归器在测试数据集上的预测几乎相等
    assert_almost_equal(rgr.predict(X_test), rgr_w.predict(X_test))


# 导入数据集
iris = datasets.load_iris()
# 创建多目标分类的目标变量，通过随机化排序和连接 y
X = iris.data
y1 = iris.target
y2 = shuffle(y1, random_state=1)
y3 = shuffle(y1, random_state=2)
y = np.column_stack((y1, y2, y3))
n_samples, n_features = X.shape
n_outputs = y.shape[1]
n_classes = len(np.unique(y1))
classes = list(map(np.unique, (y1, y2, y3)))


# 测试多输出分类的部分拟合并行性
def test_multi_output_classification_partial_fit_parallelism():
    # 初始化随机梯度下降分类器
    sgd_linear_clf = SGDClassifier(loss="log_loss", random_state=1, max_iter=5)
    # 初始化多输出分类器，使用4个并行任务
    mor = MultiOutputClassifier(sgd_linear_clf, n_jobs=4)
    # 使用部分拟合方法拟合多输出分类器
    mor.partial_fit(X, y, classes)
    # 获取第一个估计器
    est1 = mor.estimators_[0]
    # 再次使用部分拟合方法拟合多输出分类器
    mor.partial_fit(X, y)
    # 获取第一个估计器
    est2 = mor.estimators_[0]
    # 检查当前计算机的 CPU 核心数量是否大于 1
    if cpu_count() > 1:
        # 并行处理需要这个条件成立，以确保实现的合理性
        assert est1 is not est2
# 检查 MultiOutputClassifier 是否具有 predict_proba 方法
def test_hasattr_multi_output_predict_proba():
    # 默认的 SGDClassifier 使用 loss='hinge'，不提供 predict_proba 方法
    sgd_linear_clf = SGDClassifier(random_state=1, max_iter=5)
    multi_target_linear = MultiOutputClassifier(sgd_linear_clf)
    multi_target_linear.fit(X, y)
    # 断言 multi_target_linear 对象没有 predict_proba 方法
    assert not hasattr(multi_target_linear, "predict_proba")

    # 当 predict_proba 方法存在的情况
    sgd_linear_clf = SGDClassifier(loss="log_loss", random_state=1, max_iter=5)
    multi_target_linear = MultiOutputClassifier(sgd_linear_clf)
    multi_target_linear.fit(X, y)
    # 断言 multi_target_linear 对象有 predict_proba 方法
    assert hasattr(multi_target_linear, "predict_proba")


# 检查 predict_proba 方法是否正常运行
def test_multi_output_predict_proba():
    sgd_linear_clf = SGDClassifier(random_state=1, max_iter=5)
    param = {"loss": ("hinge", "log_loss", "modified_huber")}

    # 内部函数用于自定义评分
    def custom_scorer(estimator, X, y):
        if hasattr(estimator, "predict_proba"):
            return 1.0
        else:
            return 0.0

    grid_clf = GridSearchCV(
        sgd_linear_clf,
        param_grid=param,
        scoring=custom_scorer,
        cv=3,
        error_score="raise",
    )
    multi_target_linear = MultiOutputClassifier(grid_clf)
    multi_target_linear.fit(X, y)

    multi_target_linear.predict_proba(X)

    # SGDClassifier 默认使用 loss='hinge'，这是非概率性损失函数；
    # 因此它不提供 predict_proba 方法
    sgd_linear_clf = SGDClassifier(random_state=1, max_iter=5)
    multi_target_linear = MultiOutputClassifier(sgd_linear_clf)
    multi_target_linear.fit(X, y)

    inner2_msg = "probability estimates are not available for loss='hinge'"
    inner1_msg = "'SGDClassifier' has no attribute 'predict_proba'"
    outer_msg = "'MultiOutputClassifier' has no attribute 'predict_proba'"
    # 使用 pytest 检查在调用 predict_proba 时抛出的 AttributeError 异常
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        multi_target_linear.predict_proba(X)

    # 断言异常的原因是 AttributeError 类型
    assert isinstance(exec_info.value.__cause__, AttributeError)
    # 断言异常的原因中包含 inner1_msg 消息
    assert inner1_msg in str(exec_info.value.__cause__)

    # 断言异常的原因的原因是 AttributeError 类型
    assert isinstance(exec_info.value.__cause__.__cause__, AttributeError)
    # 断言异常的原因的原因中包含 inner2_msg 消息
    assert inner2_msg in str(exec_info.value.__cause__.__cause__)


def test_multi_output_classification_partial_fit():
    # 测试 MultiOutputClassifier 是否正确初始化基础估算器并进行拟合
    # 断言 predict 方法按预期工作

    sgd_linear_clf = SGDClassifier(loss="log_loss", random_state=1, max_iter=5)
    multi_target_linear = MultiOutputClassifier(sgd_linear_clf)

    # 使用 partial_fit 训练 multi_target_linear 并获取预测
    half_index = X.shape[0] // 2
    multi_target_linear.partial_fit(X[:half_index], y[:half_index], classes=classes)

    first_predictions = multi_target_linear.predict(X)
    # 断言预测结果的形状符合预期
    assert (n_samples, n_outputs) == first_predictions.shape

    multi_target_linear.partial_fit(X[half_index:], y[half_index:])
    # 对输入数据 X 进行多目标线性模型的预测，将结果保存在 second_predictions 中
    second_predictions = multi_target_linear.predict(X)
    # 使用断言确保 second_predictions 的形状与预期的样本数和输出数相匹配
    assert (n_samples, n_outputs) == second_predictions.shape

    # 使用部分拟合方法对线性分类器进行训练，并在第一次和第二次拟合后使用断言检查预测结果是否相等
    for i in range(3):
        # 克隆具有相同状态的线性分类器对象
        sgd_linear_clf = clone(sgd_linear_clf)
        # 对前半部分数据 X[:half_index] 和标签 y[:half_index, i] 进行部分拟合
        sgd_linear_clf.partial_fit(
            X[:half_index], y[:half_index, i], classes=classes[i]
        )
        # 使用断言确保在第一次部分拟合后的预测结果与 first_predictions[:, i] 相等
        assert_array_equal(sgd_linear_clf.predict(X), first_predictions[:, i])
        # 对后半部分数据 X[half_index:] 和标签 y[half_index:, i] 进行第二次部分拟合
        sgd_linear_clf.partial_fit(X[half_index:], y[half_index:, i])
        # 使用断言确保在第二次部分拟合后的预测结果与 second_predictions[:, i] 相等
        assert_array_equal(sgd_linear_clf.predict(X), second_predictions[:, i])
# 测试函数，验证在未传递第一个类别的情况下是否会引发异常
def test_multi_output_classification_partial_fit_no_first_classes_exception():
    # 创建一个随机状态为1、最大迭代次数为5的逻辑损失SGD分类器
    sgd_linear_clf = SGDClassifier(loss="log_loss", random_state=1, max_iter=5)
    # 创建一个多输出分类器，使用上面创建的SGD分类器作为基础估计器
    multi_target_linear = MultiOutputClassifier(sgd_linear_clf)
    # 预期抛出异常的消息
    msg = "classes must be passed on the first call to partial_fit."
    # 使用 pytest 断言来验证是否抛出 ValueError 异常，并且异常消息匹配预期的消息
    with pytest.raises(ValueError, match=msg):
        multi_target_linear.partial_fit(X, y)


# 测试多输出分类器的功能
def test_multi_output_classification():
    # 创建一个包含10棵树的随机森林分类器，随机状态为1
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    # 创建一个多输出分类器，使用上面创建的随机森林分类器作为基础估计器
    multi_target_forest = MultiOutputClassifier(forest)

    # 使用训练数据 X, y 来训练多输出分类器
    multi_target_forest.fit(X, y)

    # 使用训练好的多输出分类器进行预测
    predictions = multi_target_forest.predict(X)
    # 验证预测结果的形状是否符合预期
    assert (n_samples, n_outputs) == predictions.shape

    # 使用多输出分类器预测每个输出的概率
    predict_proba = multi_target_forest.predict_proba(X)

    # 验证预测概率的列表长度是否等于输出数量
    assert len(predict_proba) == n_outputs
    # 验证每个类别概率数组的形状是否符合预期
    for class_probabilities in predict_proba:
        assert (n_samples, n_classes) == class_probabilities.shape

    # 验证利用预测概率计算的最有可能类别是否与预测结果一致
    assert_array_equal(np.argmax(np.dstack(predict_proba), axis=1), predictions)

    # 对每一列数据分别使用随机森林进行训练，并验证预测结果是否一致
    for i in range(3):
        # 克隆一个具有相同状态的随机森林分类器
        forest_ = clone(forest)
        # 使用单独的列 y[:, i] 训练克隆的随机森林分类器
        forest_.fit(X, y[:, i])
        # 验证预测结果是否一致
        assert list(forest_.predict(X)) == list(predictions[:, i])
        # 验证预测概率数组是否一致
        assert_array_equal(list(forest_.predict_proba(X)), list(predict_proba[i]))


# 测试多类别多输出估计器的功能
def test_multiclass_multioutput_estimator():
    # 创建一个线性支持向量机分类器，随机状态为0
    svc = LinearSVC(random_state=0)
    # 创建一个一对多分类器，使用上面创建的线性支持向量机分类器作为基础估计器
    multi_class_svc = OneVsRestClassifier(svc)
    # 创建一个多输出分类器，使用上面创建的一对多分类器作为基础估计器
    multi_target_svc = MultiOutputClassifier(multi_class_svc)

    # 使用训练数据 X, y 来训练多输出分类器
    multi_target_svc.fit(X, y)

    # 使用训练好的多输出分类器进行预测
    predictions = multi_target_svc.predict(X)
    # 验证预测结果的形状是否符合预期
    assert (n_samples, n_outputs) == predictions.shape

    # 对每一列数据分别使用克隆的一对多分类器进行训练，并验证预测结果是否一致
    for i in range(3):
        # 克隆一个一对多分类器
        multi_class_svc_ = clone(multi_class_svc)
        # 使用单独的列 y[:, i] 训练克隆的一对多分类器
        multi_class_svc_.fit(X, y[:, i])
        # 验证预测结果是否一致
        assert list(multi_class_svc_.predict(X)) == list(predictions[:, i])


# 测试多类别多输出估计器的预测概率功能
def test_multiclass_multioutput_estimator_predict_proba():
    seed = 542

    # 设置随机数种子，使测试结果具有确定性
    rng = np.random.RandomState(seed)

    # 随机生成特征数据
    X = rng.normal(size=(5, 5))

    # 随机生成标签数据
    y1 = np.array(["b", "a", "a", "b", "a"]).reshape(5, 1)  # 2个类别
    y2 = np.array(["d", "e", "f", "e", "d"]).reshape(5, 1)  # 3个类别

    Y = np.concatenate([y1, y2], axis=1)

    # 创建一个多输出分类器，使用逻辑回归作为基础估计器
    clf = MultiOutputClassifier(
        LogisticRegression(solver="liblinear", random_state=seed)
    )

    # 使用训练数据 X, Y 来训练多输出分类器
    clf.fit(X, Y)

    # 使用训练好的多输出分类器进行预测概率计算
    y_result = clf.predict_proba(X)
    # 实际值列表，包含两个 NumPy 数组
    y_actual = [
        np.array(
            [
                [0.23481764, 0.76518236],    # 第一个数组的第一行数据
                [0.67196072, 0.32803928],    # 第一个数组的第二行数据
                [0.54681448, 0.45318552],    # 第一个数组的第三行数据
                [0.34883923, 0.65116077],    # 第一个数组的第四行数据
                [0.73687069, 0.26312931],    # 第一个数组的第五行数据
            ]
        ),
        np.array(
            [
                [0.5171785, 0.23878628, 0.24403522],  # 第二个数组的第一行数据
                [0.22141451, 0.64102704, 0.13755846],  # 第二个数组的第二行数据
                [0.16751315, 0.18256843, 0.64991843],  # 第二个数组的第三行数据
                [0.27357372, 0.55201592, 0.17441036],  # 第二个数组的第四行数据
                [0.65745193, 0.26062899, 0.08191907],  # 第二个数组的第五行数据
            ]
        ),
    ]

    # 遍历 y_actual 列表的索引
    for i in range(len(y_actual)):
        # 断言：验证 y_result[i] 和 y_actual[i] 的近似相等性
        assert_almost_equal(y_result[i], y_actual[i])
def test_multi_output_classification_sample_weights():
    # weighted classifier
    Xw = [[1, 2, 3], [4, 5, 6]]  # 特征向量 Xw
    yw = [[3, 2], [2, 3]]  # 目标标签 yw
    w = np.asarray([2.0, 1.0])  # 样本权重 w
    forest = RandomForestClassifier(n_estimators=10, random_state=1)  # 创建随机森林分类器
    clf_w = MultiOutputClassifier(forest)  # 创建多输出分类器对象 clf_w
    clf_w.fit(Xw, yw, w)  # 使用加权样本进行拟合

    # unweighted, but with repeated samples
    X = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]  # 特征向量 X
    y = [[3, 2], [3, 2], [2, 3]]  # 目标标签 y
    forest = RandomForestClassifier(n_estimators=10, random_state=1)  # 创建随机森林分类器
    clf = MultiOutputClassifier(forest)  # 创建多输出分类器对象 clf
    clf.fit(X, y)  # 使用普通样本进行拟合

    X_test = [[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]
    assert_almost_equal(clf.predict(X_test), clf_w.predict(X_test))


def test_multi_output_classification_partial_fit_sample_weights():
    # weighted classifier
    Xw = [[1, 2, 3], [4, 5, 6], [1.5, 2.5, 3.5]]  # 特征向量 Xw
    yw = [[3, 2], [2, 3], [3, 2]]  # 目标标签 yw
    w = np.asarray([2.0, 1.0, 1.0])  # 样本权重 w
    sgd_linear_clf = SGDClassifier(random_state=1, max_iter=20)  # 创建随机梯度下降分类器
    clf_w = MultiOutputClassifier(sgd_linear_clf)  # 创建多输出分类器对象 clf_w
    clf_w.fit(Xw, yw, w)  # 使用加权样本进行拟合

    # unweighted, but with repeated samples
    X = [[1, 2, 3], [1, 2, 3], [4, 5, 6], [1.5, 2.5, 3.5]]  # 特征向量 X
    y = [[3, 2], [3, 2], [2, 3], [3, 2]]  # 目标标签 y
    sgd_linear_clf = SGDClassifier(random_state=1, max_iter=20)  # 创建随机梯度下降分类器
    clf = MultiOutputClassifier(sgd_linear_clf)  # 创建多输出分类器对象 clf
    clf.fit(X, y)  # 使用普通样本进行拟合
    X_test = [[1.5, 2.5, 3.5]]
    assert_array_almost_equal(clf.predict(X_test), clf_w.predict(X_test))


def test_multi_output_exceptions():
    # NotFittedError when fit is not done but score, predict and
    # and predict_proba are called
    moc = MultiOutputClassifier(LinearSVC(random_state=0))  # 创建多输出分类器对象 moc
    with pytest.raises(NotFittedError):
        moc.score(X, y)  # 检查未拟合时调用 score 是否会引发 NotFittedError

    # ValueError when number of outputs is different
    # for fit and score
    y_new = np.column_stack((y1, y2))  # 将 y1 和 y2 合并成 y_new
    moc.fit(X, y)  # 使用数据 X 和标签 y 进行拟合
    with pytest.raises(ValueError):
        moc.score(X, y_new)  # 检查当拟合和评分的输出数量不一致时是否会引发 ValueError

    # ValueError when y is continuous
    msg = "Unknown label type"
    with pytest.raises(ValueError, match=msg):
        moc.fit(X, X[:, 1])  # 检查当标签 y 是连续值时是否会引发 ValueError


@pytest.mark.parametrize("response_method", ["predict_proba", "predict"])
def test_multi_output_not_fitted_error(response_method):
    """Check that we raise the proper error when the estimator is not fitted"""
    moc = MultiOutputClassifier(LogisticRegression())  # 创建多输出分类器对象 moc
    with pytest.raises(NotFittedError):
        getattr(moc, response_method)(X)  # 检查当模型未拟合时调用 predict_proba 或 predict 是否会引发 NotFittedError


def test_multi_output_delegate_predict_proba():
    """Check the behavior for the delegation of predict_proba to the underlying
    estimator"""

    # A base estimator with `predict_proba`should expose the method even before fit
    moc = MultiOutputClassifier(LogisticRegression())  # 创建多输出分类器对象 moc
    assert hasattr(moc, "predict_proba")  # 检查基础估计器是否具有 predict_proba 方法
    moc.fit(X, y)  # 使用数据 X 和标签 y 进行拟合
    assert hasattr(moc, "predict_proba")  # 检查在拟合之后是否仍然具有 predict_proba 方法

    # A base estimator without `predict_proba` should raise an AttributeError
    moc = MultiOutputClassifier(LinearSVC())  # 创建多输出分类器对象 moc
    assert not hasattr(moc, "predict_proba")  # 检查基础估计器是否不具有 predict_proba 方法

    outer_msg = "'MultiOutputClassifier' has no attribute 'predict_proba'"
    # 定义一个内部错误消息字符串，用于断言异常中的具体信息
    inner_msg = "'LinearSVC' object has no attribute 'predict_proba'"
    # 使用 pytest 模块的 raises 方法检测是否引发指定异常，并匹配外部错误消息
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        # 调用模型对象 moc 的 predict_proba 方法，期望引发 AttributeError 异常
        moc.predict_proba(X)
    # 断言异常信息的原因为 AttributeError
    assert isinstance(exec_info.value.__cause__, AttributeError)
    # 断言内部错误消息与异常原因的字符串表示一致
    assert inner_msg == str(exec_info.value.__cause__)
    
    # 使用训练后的模型对象 moc 对数据集 X 进行拟合
    moc.fit(X, y)
    # 断言模型对象 moc 不具有 predict_proba 属性
    assert not hasattr(moc, "predict_proba")
    # 再次使用 pytest 模块的 raises 方法检测是否引发指定异常，并匹配外部错误消息
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        # 再次调用模型对象 moc 的 predict_proba 方法，期望引发 AttributeError 异常
        moc.predict_proba(X)
    # 断言异常信息的原因为 AttributeError
    assert isinstance(exec_info.value.__cause__, AttributeError)
    # 断言内部错误消息与异常原因的字符串表示一致
    assert inner_msg == str(exec_info.value.__cause__)
# 生成具有相关性的多标签数据集，从多类数据集中使用二进制编码来表示原始类的整数号码。
def generate_multilabel_dataset_with_correlations():
    # 使用make_classification函数生成具有相关性的多标签数据集
    X, y = make_classification(
        n_samples=1000, n_features=100, n_classes=16, n_informative=10, random_state=0
    )

    # 将原始类的整数号码转换为二进制编码的多标签表示
    Y_multi = np.array([[int(yyy) for yyy in format(yy, "#06b")[2:]] for yy in y])
    return X, Y_multi


# 使用LinearSVC拟合分类器链，并验证使用指定方法（预测或决策函数）的性能
@pytest.mark.parametrize("chain_method", ["predict", "decision_function"])
def test_classifier_chain_fit_and_predict_with_linear_svc(chain_method):
    # 生成具有相关性的多标签数据集
    X, Y = generate_multilabel_dataset_with_correlations()

    # 创建分类器链对象，使用LinearSVC作为基础分类器，并进行拟合
    classifier_chain = ClassifierChain(
        LinearSVC(),
        chain_method=chain_method,
    ).fit(X, Y)

    # 预测并断言预测结果形状与真实标签形状相同
    Y_pred = classifier_chain.predict(X)
    assert Y_pred.shape == Y.shape

    # 计算决策函数的值
    Y_decision = classifier_chain.decision_function(X)

    # 将决策函数值转换为二进制预测
    Y_binary = Y_decision >= 0
    assert_array_equal(Y_binary, Y_pred)
    assert not hasattr(classifier_chain, "predict_proba")


# 使用稀疏数据进行分类器链的拟合和预测验证
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_classifier_chain_fit_and_predict_with_sparse_data(csr_container):
    # 生成具有相关性的多标签数据集
    X, Y = generate_multilabel_dataset_with_correlations()

    # 将数据转换为稀疏表示
    X_sparse = csr_container(X)

    # 使用LogisticRegression作为基础分类器拟合分类器链（稀疏数据版本）
    classifier_chain = ClassifierChain(LogisticRegression()).fit(X_sparse, Y)
    Y_pred_sparse = classifier_chain.predict(X_sparse)

    # 使用LogisticRegression作为基础分类器拟合分类器链（密集数据版本）
    classifier_chain = ClassifierChain(LogisticRegression()).fit(X, Y)
    Y_pred_dense = classifier_chain.predict(X)

    # 断言稀疏和密集版本的预测结果相等
    assert_array_equal(Y_pred_sparse, Y_pred_dense)


# 验证分类器链（每个链的长度为N）的集合可以比N个独立模型获得更高的Jaccard相似度分数
def test_classifier_chain_vs_independent_models():
    # 生成具有相关性的多标签数据集
    X, Y = generate_multilabel_dataset_with_correlations()

    # 划分训练集和测试集
    X_train = X[:600, :]
    X_test = X[600:, :]
    Y_train = Y[:600, :]
    Y_test = Y[600:, :]

    # 训练并预测使用OneVsRestClassifier的结果
    ovr = OneVsRestClassifier(LogisticRegression())
    ovr.fit(X_train, Y_train)
    Y_pred_ovr = ovr.predict(X_test)

    # 训练并预测使用ClassifierChain的结果
    chain = ClassifierChain(LogisticRegression())
    chain.fit(X_train, Y_train)
    Y_pred_chain = chain.predict(X_test)

    # 断言使用ClassifierChain获得的Jaccard相似度分数比使用OneVsRestClassifier更高
    assert jaccard_score(Y_test, Y_pred_chain, average="samples") > jaccard_score(
        Y_test, Y_pred_ovr, average="samples"
    )


# 使用不同的链方法（预测、预测概率、预测对数概率、决策函数）和不同的响应方法（预测概率、预测对数概率）验证分类器链的拟合和预测性能
@pytest.mark.parametrize(
    "chain_method",
    ["predict", "predict_proba", "predict_log_proba", "decision_function"],
)
@pytest.mark.parametrize("response_method", ["predict_proba", "predict_log_proba"])
def test_classifier_chain_fit_and_predict(chain_method, response_method):
    # 生成具有相关性的多标签数据集
    X, Y = generate_multilabel_dataset_with_correlations()

    # 创建分类器链对象，使用LogisticRegression作为基础分类器，并进行拟合
    chain = ClassifierChain(LogisticRegression(), chain_method=chain_method)
    chain.fit(X, Y)

    # 进行预测并断言预测结果形状与真实标签形状相同
    Y_pred = chain.predict(X)
    assert Y_pred.shape == Y.shape
    # 使用列表推导式生成一个包含每个子分类器特征数的列表，断言其与预期列表相等
    assert [c.coef_.size for c in chain.estimators_] == list(
        range(X.shape[1], X.shape[1] + Y.shape[1])
    )

    # 调用链中的指定方法（响应方法）来预测响应变量 Y_prob
    Y_prob = getattr(chain, response_method)(X)

    # 如果响应方法是 "predict_log_proba"，则对 Y_prob 应用指数函数，转换为概率
    if response_method == "predict_log_proba":
        Y_prob = np.exp(Y_prob)

    # 基于概率阈值（0.5），生成二进制预测 Y_binary
    Y_binary = Y_prob >= 0.5

    # 断言 Y_binary 与预测 Y_pred 相等
    assert_array_equal(Y_binary, Y_pred)

    # 断言 chain 对象是 ClassifierMixin 的实例
    assert isinstance(chain, ClassifierMixin)
def test_regressor_chain_fit_and_predict():
    # Fit regressor chain and verify Y and estimator coefficients shape

    # 生成具有相关性的多标签数据集
    X, Y = generate_multilabel_dataset_with_correlations()

    # 创建一个使用 Ridge 回归器的回归链对象
    chain = RegressorChain(Ridge())

    # 使用数据集 X, Y 进行拟合
    chain.fit(X, Y)

    # 预测输出 Y_pred
    Y_pred = chain.predict(X)

    # 断言预测结果的形状与 Y 相同
    assert Y_pred.shape == Y.shape

    # 断言每个估计器的系数形状正确
    assert [c.coef_.size for c in chain.estimators_] == list(
        range(X.shape[1], X.shape[1] + Y.shape[1])
    )


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_base_chain_fit_and_predict_with_sparse_data_and_cv(csr_container):
    # Fit base chain with sparse data cross_val_predict

    # 生成具有相关性的多标签数据集
    X, Y = generate_multilabel_dataset_with_correlations()

    # 将数据集 X 转换为稀疏格式
    X_sparse = csr_container(X)

    # 创建包含分类链和回归链的基本链列表
    base_chains = [
        ClassifierChain(LogisticRegression(), cv=3),
        RegressorChain(Ridge(), cv=3),
    ]

    # 对于每个链进行拟合和预测
    for chain in base_chains:
        chain.fit(X_sparse, Y)
        Y_pred = chain.predict(X_sparse)

        # 断言预测结果的形状与 Y 相同
        assert Y_pred.shape == Y.shape


def test_base_chain_random_order():
    # Fit base chain with random order

    # 生成具有相关性的多标签数据集
    X, Y = generate_multilabel_dataset_with_correlations()

    # 对于分类链和回归链，使用随机顺序进行拟合
    for chain in [ClassifierChain(LogisticRegression()), RegressorChain(Ridge())]:
        chain_random = clone(chain).set_params(order="random", random_state=42)
        chain_random.fit(X, Y)
        
        # 使用相同的顺序创建固定顺序的链
        chain_fixed = clone(chain).set_params(order=chain_random.order_)
        chain_fixed.fit(X, Y)
        
        # 断言随机顺序与固定顺序的顺序数组相等
        assert_array_equal(chain_fixed.order_, chain_random.order_)
        
        # 断言随机顺序不等于默认顺序
        assert list(chain_random.order) != list(range(4))
        
        # 断言随机顺序数组长度为 4
        assert len(chain_random.order_) == 4
        
        # 断言随机顺序数组中唯一值的数量为 4
        assert len(set(chain_random.order_)) == 4
        
        # 随机顺序的链应该与具有相同顺序的固定顺序链表现相同
        for est1, est2 in zip(chain_random.estimators_, chain_fixed.estimators_):
            assert_array_almost_equal(est1.coef_, est2.coef_)


@pytest.mark.parametrize(
    "chain_type, chain_method",
    [
        ("classifier", "predict"),
        ("classifier", "predict_proba"),
        ("classifier", "predict_log_proba"),
        ("classifier", "decision_function"),
        ("regressor", ""),
    ],
)
def test_base_chain_crossval_fit_and_predict(chain_type, chain_method):
    # Fit chain with cross_val_predict and verify predict
    # performance

    # 生成具有相关性的多标签数据集
    X, Y = generate_multilabel_dataset_with_correlations()

    # 根据 chain_type 创建分类链或回归链
    if chain_type == "classifier":
        chain = ClassifierChain(LogisticRegression(), chain_method=chain_method)
    else:
        chain = RegressorChain(Ridge())

    # 使用数据集 X, Y 进行拟合
    chain.fit(X, Y)

    # 创建具有 3 折交叉验证的 chain_cv
    chain_cv = clone(chain).set_params(cv=3)
    chain_cv.fit(X, Y)

    # 使用交叉验证数据集 X 进行预测
    Y_pred_cv = chain_cv.predict(X)
    Y_pred = chain.predict(X)

    # 断言交叉验证预测结果的形状与直接预测结果相同
    assert Y_pred_cv.shape == Y_pred.shape
    
    # 断言直接预测结果不全等于交叉验证预测结果
    assert not np.all(Y_pred == Y_pred_cv)

    # 根据链类型不同，使用不同的评估指标进行断言
    if isinstance(chain, ClassifierChain):
        assert jaccard_score(Y, Y_pred_cv, average="samples") > 0.4
    else:
        assert mean_squared_error(Y, Y_pred_cv) < 0.25
    [
        # 创建一个随机森林分类器对象，使用2棵决策树作为基分类器
        RandomForestClassifier(n_estimators=2),
        # 创建一个多输出分类器对象，内部使用随机森林分类器，每个输出都有2棵决策树
        MultiOutputClassifier(RandomForestClassifier(n_estimators=2)),
        # 创建一个分类器链对象，每个链节点使用随机森林分类器，每个分类器有2棵决策树
        ClassifierChain(RandomForestClassifier(n_estimators=2)),
    ],
def test_multi_output_classes_(estimator):
    # 测试多输出分类器的 classes_ 属性
    # RandomForestClassifier 支持直接多输出
    estimator.fit(X, y)
    # 断言 classes_ 是一个列表
    assert isinstance(estimator.classes_, list)
    # 断言 classes_ 的长度等于输出的数量
    assert len(estimator.classes_) == n_outputs
    # 检查每个分类器的 classes_ 是否与预期的一致
    for estimator_classes, expected_classes in zip(classes, estimator.classes_):
        assert_array_equal(estimator_classes, expected_classes)


class DummyRegressorWithFitParams(DummyRegressor):
    def fit(self, X, y, sample_weight=None, **fit_params):
        # 将 fit_params 存储在 _fit_params 中
        self._fit_params = fit_params
        return super().fit(X, y, sample_weight)


class DummyClassifierWithFitParams(DummyClassifier):
    def fit(self, X, y, sample_weight=None, **fit_params):
        # 将 fit_params 存储在 _fit_params 中
        self._fit_params = fit_params
        return super().fit(X, y, sample_weight)


@pytest.mark.filterwarnings("ignore:`n_features_in_` is deprecated")
@pytest.mark.parametrize(
    "estimator, dataset",
    [
        (
            MultiOutputClassifier(DummyClassifierWithFitParams(strategy="prior")),
            datasets.make_multilabel_classification(),
        ),
        (
            MultiOutputRegressor(DummyRegressorWithFitParams()),
            datasets.make_regression(n_targets=3, random_state=0),
        ),
    ],
)
def test_multioutput_estimator_with_fit_params(estimator, dataset):
    X, y = dataset
    some_param = np.zeros_like(X)
    # 使用 some_param 参数进行拟合
    estimator.fit(X, y, some_param=some_param)
    # 检查每个估计器的 _fit_params 是否包含 "some_param"
    for dummy_estimator in estimator.estimators_:
        assert "some_param" in dummy_estimator._fit_params


def test_regressor_chain_w_fit_params():
    # 确保 fit_params 正确传递给子估计器
    rng = np.random.RandomState(0)
    X, y = datasets.make_regression(n_targets=3, random_state=0)
    weight = rng.rand(y.shape[0])

    class MySGD(SGDRegressor):
        def fit(self, X, y, **fit_params):
            # 存储 sample_weight_ 参数
            self.sample_weight_ = fit_params["sample_weight"]
            super().fit(X, y, **fit_params)

    model = RegressorChain(MySGD())

    # 使用 fit_param 进行拟合
    fit_param = {"sample_weight": weight}
    model.fit(X, y, **fit_param)

    # 检查每个估计器的 sample_weight_ 是否为预期的 weight
    for est in model.estimators_:
        assert est.sample_weight_ is weight


@pytest.mark.parametrize(
    "MultiOutputEstimator, Estimator",
    [(MultiOutputClassifier, LogisticRegression), (MultiOutputRegressor, Ridge)],
)
# FIXME: we should move this test in `estimator_checks` once we are able
# to construct meta-estimator instances
def test_support_missing_values(MultiOutputEstimator, Estimator):
    # 烟雾测试以检查管道 MultioutputEstimators 是否将缺失值验证委托给底层管道、回归器或分类器
    rng = np.random.RandomState(42)
    X, y = rng.randn(50, 2), rng.binomial(1, 0.5, (50, 3))
    mask = rng.choice([1, 0], X.shape, p=[0.01, 0.99]).astype(bool)
    X[mask] = np.nan

    # 创建管道，包含 SimpleImputer 和 Estimator
    pipe = make_pipeline(SimpleImputer(), Estimator())
    # 使用 MultiOutputEstimator 对象来拟合数据集 X 和标签 y，并计算拟合后的模型在训练集上的得分
    MultiOutputEstimator(pipe).fit(X, y).score(X, y)
@pytest.mark.parametrize("order_type", [list, np.array, tuple])
# 使用 pytest 的参数化装饰器，为 test_classifier_chain_tuple_order 函数提供不同的 order_type 参数进行多次测试
def test_classifier_chain_tuple_order(order_type):
    X = [[1, 2, 3], [4, 5, 6], [1.5, 2.5, 3.5]]
    y = [[3, 2], [2, 3], [3, 2]]
    order = order_type([1, 0])

    chain = ClassifierChain(
        RandomForestClassifier(n_estimators=2, random_state=0), order=order
    )
    # 创建 ClassifierChain 对象，使用随机森林分类器，指定顺序为 order

    chain.fit(X, y)
    # 训练 ClassifierChain 对象，拟合输入的 X 和 y 数据

    X_test = [[1.5, 2.5, 3.5]]
    y_test = [[3, 2]]
    assert_array_almost_equal(chain.predict(X_test), y_test)
    # 断言预测结果与期望的 y_test 相近


def test_classifier_chain_tuple_invalid_order():
    X = [[1, 2, 3], [4, 5, 6], [1.5, 2.5, 3.5]]
    y = [[3, 2], [2, 3], [3, 2]]
    order = tuple([1, 2])

    chain = ClassifierChain(RandomForestClassifier(), order=order)
    # 创建 ClassifierChain 对象，使用随机森林分类器，指定顺序为 order

    with pytest.raises(ValueError, match="invalid order"):
        chain.fit(X, y)
    # 使用 pytest 的断言，期望抛出 ValueError 异常并且异常信息包含 "invalid order"


def test_classifier_chain_verbose(capsys):
    X, y = make_multilabel_classification(
        n_samples=100, n_features=5, n_classes=3, n_labels=3, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pattern = (
        r"\[Chain\].*\(1 of 3\) Processing order 0, total=.*\n"
        r"\[Chain\].*\(2 of 3\) Processing order 1, total=.*\n"
        r"\[Chain\].*\(3 of 3\) Processing order 2, total=.*\n$"
    )

    classifier = ClassifierChain(
        DecisionTreeClassifier(),
        order=[0, 1, 2],
        random_state=0,
        verbose=True,
    )
    # 创建 ClassifierChain 对象，使用决策树分类器，指定顺序为 [0, 1, 2]，启用详细输出

    classifier.fit(X_train, y_train)
    # 训练 ClassifierChain 对象，拟合输入的训练数据 X_train 和 y_train

    assert re.match(pattern, capsys.readouterr()[0])
    # 使用正则表达式匹配输出，验证是否符合预期的输出格式


def test_regressor_chain_verbose(capsys):
    X, y = make_regression(n_samples=125, n_targets=3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pattern = (
        r"\[Chain\].*\(1 of 3\) Processing order 1, total=.*\n"
        r"\[Chain\].*\(2 of 3\) Processing order 0, total=.*\n"
        r"\[Chain\].*\(3 of 3\) Processing order 2, total=.*\n$"
    )
    regressor = RegressorChain(
        LinearRegression(),
        order=[1, 0, 2],
        random_state=0,
        verbose=True,
    )
    # 创建 RegressorChain 对象，使用线性回归器，指定顺序为 [1, 0, 2]，启用详细输出

    regressor.fit(X_train, y_train)
    # 训练 RegressorChain 对象，拟合输入的训练数据 X_train 和 y_train

    assert re.match(pattern, capsys.readouterr()[0])
    # 使用正则表达式匹配输出，验证是否符合预期的输出格式


def test_multioutputregressor_ducktypes_fitted_estimator():
    """Test that MultiOutputRegressor checks the fitted estimator for
    predict. Non-regression test for #16549."""
    X, y = load_linnerud(return_X_y=True)
    stacker = StackingRegressor(
        estimators=[("sgd", SGDRegressor(random_state=1))],
        final_estimator=Ridge(),
        cv=2,
    )

    reg = MultiOutputRegressor(estimator=stacker).fit(X, y)
    # 创建 MultiOutputRegressor 对象，使用堆叠回归器作为估计器，拟合输入的 X 和 y 数据

    # Does not raise
    reg.predict(X)
    # 验证预测是否成功，不应该引发异常


@pytest.mark.parametrize(
    "Cls, method", [(ClassifierChain, "fit"), (MultiOutputClassifier, "partial_fit")]
)
# 使用 pytest 的参数化装饰器，为 test_fit_params_no_routing 函数提供不同的 Cls 和 method 参数进行多次测试
def test_fit_params_no_routing(Cls, method):
    """Check that we raise an error when passing metadata not requested by the
    underlying classifier.
    """
    X, y = make_classification(n_samples=50)
    clf = Cls(PassiveAggressiveClassifier())
    # 创建指定类型的分类器对象，例如 ClassifierChain 或 MultiOutputClassifier
    # 使用 pytest 的断言来测试代码中是否会抛出 ValueError 异常，并验证异常消息是否包含特定字符串
    with pytest.raises(ValueError, match="is only supported if"):
        # 调用 getattr 函数获取对象 clf 中的方法 method，并调用该方法
        # 传入参数 X, y，并传入额外的参数 test=1
        getattr(clf, method)(X, y, test=1)
# 定义测试函数，用于验证未拟合的 MultiOutputRegressor 是否正确处理 partial_fit 的可用性
def test_multioutput_regressor_has_partial_fit():
    # 创建 MultiOutputRegressor 对象，使用 LinearRegression 作为基础估计器
    est = MultiOutputRegressor(LinearRegression())
    # 预期的错误消息，用于断言 'partial_fit' 是否是 MultiOutputRegressor 的属性
    msg = "This 'MultiOutputRegressor' has no attribute 'partial_fit'"
    # 使用 pytest 检查是否会引发 AttributeError，并匹配预期的错误消息
    with pytest.raises(AttributeError, match=msg):
        # 获取 est 对象的 'partial_fit' 属性，预期会引发 AttributeError
        getattr(est, "partial_fit")
```