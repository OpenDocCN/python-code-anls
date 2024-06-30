# `D:\src\scipysrc\scikit-learn\sklearn\semi_supervised\tests\test_label_propagation.py`

```
"""test the label propagation module"""

import warnings  # 导入警告模块

import numpy as np  # 导入 numpy 库，并使用别名 np
import pytest  # 导入 pytest 库
from scipy.sparse import issparse  # 从 scipy.sparse 模块中导入 issparse 函数

from sklearn.datasets import make_classification  # 从 sklearn.datasets 模块中导入 make_classification 函数
from sklearn.exceptions import ConvergenceWarning  # 从 sklearn.exceptions 模块中导入 ConvergenceWarning
from sklearn.metrics.pairwise import rbf_kernel  # 从 sklearn.metrics.pairwise 模块中导入 rbf_kernel 函数
from sklearn.model_selection import train_test_split  # 从 sklearn.model_selection 模块中导入 train_test_split 函数
from sklearn.neighbors import NearestNeighbors  # 从 sklearn.neighbors 模块中导入 NearestNeighbors 类
from sklearn.semi_supervised import _label_propagation as label_propagation  # 从 sklearn.semi_supervised 模块中导入 _label_propagation
from sklearn.utils._testing import (  # 从 sklearn.utils._testing 模块中导入 _convert_container、assert_allclose、assert_array_equal 函数
    _convert_container,
    assert_allclose,
    assert_array_equal,
)

CONSTRUCTOR_TYPES = ("array", "sparse_csr", "sparse_csc")  # 设定元组 CONSTRUCTOR_TYPES

ESTIMATORS = [  # 创建列表 ESTIMATORS
    (label_propagation.LabelPropagation, {"kernel": "rbf"}),  # 将 LabelPropagation 类和参数字典（kernel 为 rbf）添加到 ESTIMATORS 中
    (label_propagation.LabelPropagation, {"kernel": "knn", "n_neighbors": 2}),  # 将 LabelPropagation 类和参数字典（kernel 为 knn，n_neighbors 为 2）添加到 ESTIMATORS 中
    (
        label_propagation.LabelPropagation,
        {"kernel": lambda x, y: rbf_kernel(x, y, gamma=20)},
    ),  # 将 LabelPropagation 类和参数字典（kernel 为自定义函数）添加到 ESTIMATORS 中
    (label_propagation.LabelSpreading, {"kernel": "rbf"}),  # 将 LabelSpreading 类和参数字典（kernel 为 rbf）添加到 ESTIMATORS 中
    (label_propagation.LabelSpreading, {"kernel": "knn", "n_neighbors": 2}),  # 将 LabelSpreading 类和参数字典（kernel 为 knn，n_neighbors 为 2）添加到 ESTIMATORS 中
    (
        label_propagation.LabelSpreading,
        {"kernel": lambda x, y: rbf_kernel(x, y, gamma=20)},
    ),  # 将 LabelSpreading 类和参数字典（kernel 为自定义函数）添加到 ESTIMATORS 中
]


@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)  # 使用 pytest.mark.parametrize 装饰器设置参数化测试
def test_fit_transduction(global_dtype, Estimator, parameters):  # 定义测试函数 test_fit_transduction，包含参数 global_dtype, Estimator, parameters
    samples = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 3.0]], dtype=global_dtype)  # 创建样本数组 samples
    labels = [0, 1, -1]  # 创建标签列表 labels
    clf = Estimator(**parameters).fit(samples, labels)  # 使用给定参数构建 Estimator 对象 clf，并对样本数据和标签进行拟合训练
    assert clf.transduction_[2] == 1  # 断言拟合结果中下标为 2 的 transduction_ 值为 1


@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)  # 使用 pytest.mark.parametrize 装饰器设置参数化测试
def test_distribution(global_dtype, Estimator, parameters):  # 定义测试函数 test_distribution，包含参数 global_dtype, Estimator, parameters
    if parameters["kernel"] == "knn":  # 如果参数字典中的 kernel 为 knn
        pytest.skip(  # 进行跳过测试的操作
            "Unstable test for this configuration: changes in k-NN ordering break it."
        )
    samples = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=global_dtype)  # 创建样本数组 samples
    labels = [0, 1, -1]  # 创建标签列表 labels
    clf = Estimator(**parameters).fit(samples, labels)  # 使用给定参数构建 Estimator 对象 clf，并对样本数据和标签进行拟合训练
    assert_allclose(clf.label_distributions_[2], [0.5, 0.5], atol=1e-2)  # 使用 assert_allclose 函数进行断言，判断 label_distributions_ 中下标为 2 的值与给定数组的接近程度


@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)  # 使用 pytest.mark.parametrize 装饰器设置参数化测试
def test_predict(global_dtype, Estimator, parameters):  # 定义测试函数 test_predict，包含参数 global_dtype, Estimator, parameters
    samples = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 3.0]], dtype=global_dtype)  # 创建样本数组 samples
    labels = [0, 1, -1]  # 创建标签列表 labels
    clf = Estimator(**parameters).fit(samples, labels)  # 使用给定参数构建 Estimator 对象 clf，并对样本数据和标签进行拟合训练
    assert_array_equal(clf.predict([[0.5, 2.5]]), np.array([1]))  # 使用 assert_array_equal 函数进行断言，判断预测结果与给定数组的相等性


@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)  # 使用 pytest.mark.parametrize 装饰器设置参数化测试
def test_predict_proba(global_dtype, Estimator, parameters):  # 定义测试函数 test_predict_proba，包含参数 global_dtype, Estimator, parameters
    samples = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 2.5]], dtype=global_dtype)  # 创建样本数组 samples
    labels = [0, 1, -1]  # 创建标签列表 labels
    clf = Estimator(**parameters).fit(samples, labels)  # 使用给定参数构建 Estimator 对象 clf，并对样本数据和标签进行拟合训练
    assert_allclose(clf.predict_proba([[1.0, 1.0]]), np.array([[0.5, 0.5]]))  # 使用 assert_allclose 函数进行断言，判断预测概率结果与给定数组的接近程度


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])  # 使用 pytest.mark.parametrize 装饰器设置参数化测试
@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)  # 使用 pytest.mark.parametrize 装饰器设置参数化测试
def test_label_spreading_closed_form(global_dtype, Estimator, parameters, alpha):  # 定义测试函数 test_label_spreading_closed_form，包含参数 global_dtype, Estimator, parameters, alpha
    n_classes = 2  # 定义变量 n_classes，表示类别数为 2
    # 使用 make_classification 函数生成一个具有多类别的合成数据集，样本数为 200，随机种子为 0
    X, y = make_classification(n_classes=n_classes, n_samples=200, random_state=0)
    # 将 X 数据类型转换为全局变量 global_dtype 所指定的类型，不进行复制
    X = X.astype(global_dtype, copy=False)
    # 将 y 数组中每隔三个元素的值设置为 -1，用于模拟标签中的噪声

    # 设置 gamma 值为 0.1，用于标签传播算法中的参数
    gamma = 0.1
    # 使用标签传播算法中的 LabelSpreading 类，并使用 gamma 参数进行初始化
    clf = label_propagation.LabelSpreading(gamma=gamma).fit(X, y)
    # 采用 Zhou et al. (2004) 的符号约定：
    # 构建图结构并返回表示图的邻接矩阵 S
    S = clf._build_graph()
    # 创建一个大小为 (len(y), n_classes + 1) 的全零数组 Y，数据类型与 X 相同
    Y = np.zeros((len(y), n_classes + 1), dtype=X.dtype)
    # 将 Y 中每行对应 y 中的值的列设为 1，表示每个样本的真实类别
    Y[np.arange(len(y)), y] = 1
    # 从 Y 中去掉最后一列，因为这一列只是用来初始化的，不需要在后续的计算中使用

    # 计算期望的标签分布，采用标签传播算法中的数学公式进行计算
    expected = np.dot(np.linalg.inv(np.eye(len(S), dtype=S.dtype) - alpha * S), Y)
    # 对每一行的元素除以该行元素之和，以确保每个样本的标签分布概率之和为 1
    expected /= expected.sum(axis=1)[:, np.newaxis]

    # 使用标签传播算法中的 LabelSpreading 类，设置最大迭代次数为 100，alpha 参数为先前定义的 alpha，容差为 1e-10，gamma 为先前定义的 gamma
    clf = label_propagation.LabelSpreading(
        max_iter=100, alpha=alpha, tol=1e-10, gamma=gamma
    )
    # 使用新的参数重新拟合模型
    clf.fit(X, y)

    # 断言预期的标签分布与模型计算得到的标签分布相近
    assert_allclose(expected, clf.label_distributions_)
def test_label_propagation_closed_form(global_dtype):
    n_classes = 2
    # 生成一个二分类的数据集，包含200个样本
    X, y = make_classification(n_classes=n_classes, n_samples=200, random_state=0)
    # 将数据类型转换为全局指定的数据类型
    X = X.astype(global_dtype, copy=False)
    # 将每隔三个样本的标签设为-1，表示未标记样本
    y[::3] = -1
    # 创建一个Y矩阵，将y转换为one-hot编码
    Y = np.zeros((len(y), n_classes + 1))
    Y[np.arange(len(y)), y] = 1
    # 找出未标记样本的索引
    unlabelled_idx = Y[:, (-1,)].nonzero()[0]
    # 找出已标记样本的索引
    labelled_idx = (Y[:, (-1,)] == 0).nonzero()[0]

    # 使用标签传播算法进行训练
    clf = label_propagation.LabelPropagation(max_iter=100, tol=1e-10, gamma=0.1)
    clf.fit(X, y)
    # 构建图的传播矩阵，采用了Zhu等人（2002）的符号表示
    T_bar = clf._build_graph()
    # 计算未标记样本之间的传播矩阵
    Tuu = T_bar[tuple(np.meshgrid(unlabelled_idx, unlabelled_idx, indexing="ij"))]
    # 计算未标记样本与已标记样本之间的传播矩阵
    Tul = T_bar[tuple(np.meshgrid(unlabelled_idx, labelled_idx, indexing="ij"))]
    # 移除Y矩阵的最后一列，得到标签数据Y_l
    Y = Y[:, :-1]
    Y_l = Y[labelled_idx, :]
    # 计算未标记样本的标签预测值Y_u
    Y_u = np.dot(np.dot(np.linalg.inv(np.eye(Tuu.shape[0]) - Tuu), Tul), Y_l)

    # 创建预期结果矩阵，并将未标记样本的标签预测值插入其中
    expected = Y.copy()
    expected[unlabelled_idx, :] = Y_u
    expected /= expected.sum(axis=1)[:, np.newaxis]

    # 断言预测结果与模型输出的标签分布非常接近
    assert_allclose(expected, clf.label_distributions_, atol=1e-4)


@pytest.mark.parametrize("accepted_sparse_type", ["sparse_csr", "sparse_csc"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)
def test_sparse_input_types(
    accepted_sparse_type, index_dtype, dtype, Estimator, parameters
):
    # 这是对＃17085的非回归测试
    # 将输入稀疏化，确保适当类型和索引类型
    X = _convert_container([[1.0, 0.0], [0.0, 2.0], [1.0, 3.0]], accepted_sparse_type)
    X.data = X.data.astype(dtype, copy=False)
    X.indices = X.indices.astype(index_dtype, copy=False)
    X.indptr = X.indptr.astype(index_dtype, copy=False)
    labels = [0, 1, -1]
    # 使用给定的估算器和参数进行拟合
    clf = Estimator(**parameters).fit(X, labels)
    # 断言预测结果与期望结果相等
    assert_array_equal(clf.predict([[0.5, 2.5]]), np.array([1]))


@pytest.mark.parametrize("constructor_type", CONSTRUCTOR_TYPES)
def test_convergence_speed(constructor_type):
    # 这是对＃5774的非回归测试
    # 将输入数据转换为特定构造类型
    X = _convert_container([[1.0, 0.0], [0.0, 1.0], [1.0, 2.5]], constructor_type)
    y = np.array([0, 1, -1])
    # 使用标签扩展模型，设置RBF核和最大迭代次数
    mdl = label_propagation.LabelSpreading(kernel="rbf", max_iter=5000)
    mdl.fit(X, y)

    # 断言模型迭代次数少于10次
    assert mdl.n_iter_ < 10
    # 断言预测结果与标签一致
    assert_array_equal(mdl.predict(X), [0, 1, 1])


def test_convergence_warning():
    # 这是对＃5774的非回归测试
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.5]])
    y = np.array([0, 1, -1])
    # 设置最大迭代次数为1，确保产生警告信息
    mdl = label_propagation.LabelSpreading(kernel="rbf", max_iter=1)
    warn_msg = "max_iter=1 was reached without convergence."
    with pytest.warns(ConvergenceWarning, match=warn_msg):
        mdl.fit(X, y)
    # 断言模型达到最大迭代次数
    assert mdl.n_iter_ == mdl.max_iter

    mdl = label_propagation.LabelPropagation(kernel="rbf", max_iter=1)
    with pytest.warns(ConvergenceWarning, match=warn_msg):
        mdl.fit(X, y)
    assert mdl.n_iter_ == mdl.max_iter

    mdl = label_propagation.LabelSpreading(kernel="rbf", max_iter=500)
    # 使用 `warnings` 模块捕获警告信息
    with warnings.catch_warnings():
        # 设置警告过滤器，捕获特定类型的警告（这里是 ConvergenceWarning），将其视为异常处理
        warnings.simplefilter("error", ConvergenceWarning)
        # 使用模型对象 `mdl` 对数据集 `X` 和标签 `y` 进行拟合训练
        mdl.fit(X, y)
    
    # 创建一个标签传播模型对象 `mdl`，使用径向基函数核并设定最大迭代次数为 500
    mdl = label_propagation.LabelPropagation(kernel="rbf", max_iter=500)
    # 使用 `warnings` 模块再次捕获警告信息
    with warnings.catch_warnings():
        # 设置警告过滤器，捕获特定类型的警告（这里是 ConvergenceWarning），将其视为异常处理
        warnings.simplefilter("error", ConvergenceWarning)
        # 使用模型对象 `mdl` 对数据集 `X` 和标签 `y` 进行拟合训练
        mdl.fit(X, y)
@pytest.mark.parametrize(
    "LabelPropagationCls",
    [label_propagation.LabelSpreading, label_propagation.LabelPropagation],
)
def test_label_propagation_non_zero_normalizer(LabelPropagationCls):
    # 检查在空的标准化器情况下不会除以零
    # 针对以下问题的非回归测试：
    # https://github.com/scikit-learn/scikit-learn/pull/15946
    # https://github.com/scikit-learn/scikit-learn/issues/9292
    X = np.array([[100.0, 100.0], [100.0, 100.0], [0.0, 0.0], [0.0, 0.0]])
    y = np.array([0, 1, -1, -1])
    # 使用指定的标签传播类别，创建标签传播模型实例
    mdl = LabelPropagationCls(kernel="knn", max_iter=100, n_neighbors=1)
    # 使用警告捕获机制捕获运行时警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # 对模型进行拟合
        mdl.fit(X, y)


def test_predict_sparse_callable_kernel(global_dtype):
    # 这是对问题 #15866 的非回归测试

    # 自定义稀疏核函数（前K个最近邻RBF）
    def topk_rbf(X, Y=None, n_neighbors=10, gamma=1e-5):
        nn = NearestNeighbors(n_neighbors=10, metric="euclidean", n_jobs=2)
        nn.fit(X)
        # 计算最近邻图的负权重RBF核矩阵
        W = -1 * nn.kneighbors_graph(Y, mode="distance").power(2) * gamma
        np.exp(W.data, out=W.data)
        assert issparse(W)
        return W.T

    n_classes = 4
    n_samples = 500
    n_test = 10
    # 生成分类数据集
    X, y = make_classification(
        n_classes=n_classes,
        n_samples=n_samples,
        n_features=20,
        n_informative=20,
        n_redundant=0,
        n_repeated=0,
        random_state=0,
    )
    X = X.astype(global_dtype)

    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=0
    )

    # 使用自定义稀疏核函数创建标签传播模型，并对训练数据进行拟合
    model = label_propagation.LabelSpreading(kernel=topk_rbf)
    model.fit(X_train, y_train)
    # 断言模型在测试集上的得分大于等于0.9
    assert model.score(X_test, y_test) >= 0.9

    # 使用自定义稀疏核函数创建标签传播模型，并对训练数据进行拟合
    model = label_propagation.LabelPropagation(kernel=topk_rbf)
    model.fit(X_train, y_train)
    # 断言模型在测试集上的得分大于等于0.9
    assert model.score(X_test, y_test) >= 0.9
```