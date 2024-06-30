# `D:\src\scipysrc\scikit-learn\sklearn\svm\tests\test_sparse.py`

```
# 导入必要的库
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试
from scipy import sparse  # 导入scipy库的稀疏模块

from sklearn import base, datasets, linear_model, svm  # 导入sklearn的基础模块、数据集、线性模型和支持向量机模型
from sklearn.datasets import load_digits, make_blobs, make_classification  # 导入加载数字数据集、生成聚类数据、生成分类数据的函数
from sklearn.exceptions import ConvergenceWarning  # 导入收敛警告异常类
from sklearn.svm.tests import test_svm  # 导入支持向量机的测试模块
from sklearn.utils._testing import (  # 导入用于测试的实用工具函数
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
    skip_if_32bit,
)
from sklearn.utils.extmath import safe_sparse_dot  # 导入用于稀疏矩阵乘法的函数
from sklearn.utils.fixes import (  # 导入用于修复的实用工具
    CSR_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
)

# test sample 1
# 定义测试样本1的数据
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y = [1, 1, 1, 2, 2, 2]
T = np.array([[-1, -1], [2, 2], [3, 2]])
true_result = [1, 2, 2]

# test sample 2
# 定义测试样本2的数据
X2 = np.array(
    [
        [0, 0, 0],
        [1, 1, 1],
        [2, 0, 0],
        [0, 0, 2],
        [3, 3, 3],
    ]
)
Y2 = [1, 2, 2, 2, 3]
T2 = np.array([[-1, -1, -1], [1, 1, 1], [2, 2, 2]])
true_result2 = [1, 2, 3]

# 加载鸢尾花数据集并随机打乱顺序
iris = datasets.load_iris()
rng = np.random.RandomState(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# 生成用于测试的聚类数据
X_blobs, y_blobs = make_blobs(n_samples=100, centers=10, random_state=0)


def check_svm_model_equal(dense_svm, X_train, y_train, X_test):
    # 克隆一个与原始svm模型相同的稀疏svm模型
    sparse_svm = base.clone(dense_svm)

    # 使用稠密数据拟合原始svm模型，并使用相同数据类型的稀疏数据拟合克隆的svm模型
    dense_svm.fit(X_train.toarray(), y_train)
    if sparse.issparse(X_test):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
    sparse_svm.fit(X_train, y_train)

    # 断言稀疏模型的支持向量和对偶系数是稀疏矩阵
    assert sparse.issparse(sparse_svm.support_vectors_)
    assert sparse.issparse(sparse_svm.dual_coef_)

    # 断言稠密模型和稀疏模型的支持向量和对偶系数相近
    assert_allclose(dense_svm.support_vectors_, sparse_svm.support_vectors_.toarray())
    assert_allclose(dense_svm.dual_coef_, sparse_svm.dual_coef_.toarray())

    # 如果是线性核函数，断言稠密模型和稀疏模型的系数相近
    if dense_svm.kernel == "linear":
        assert sparse.issparse(sparse_svm.coef_)
        assert_array_almost_equal(dense_svm.coef_, sparse_svm.coef_.toarray())

    # 断言稠密模型和稀疏模型的支持向量索引相近
    assert_allclose(dense_svm.support_, sparse_svm.support_)

    # 断言稠密模型和稀疏模型在测试数据上的预测结果相近
    assert_allclose(dense_svm.predict(X_test_dense), sparse_svm.predict(X_test))

    # 断言稠密模型和稀疏模型在测试数据上的决策函数值相近
    assert_array_almost_equal(
        dense_svm.decision_function(X_test_dense), sparse_svm.decision_function(X_test)
    )
    assert_array_almost_equal(
        dense_svm.decision_function(X_test_dense),
        sparse_svm.decision_function(X_test_dense),
    )

    # 如果是OneClassSVM模型，断言不能在稀疏输入上进行预测
    if isinstance(dense_svm, svm.OneClassSVM):
        msg = "cannot use sparse input in 'OneClassSVM' trained on dense data"
    else:
        # 否则，断言不能在稀疏输入上进行预测
        assert_array_almost_equal(
            dense_svm.predict_proba(X_test_dense),
            sparse_svm.predict_proba(X_test),
            decimal=4,
        )
        msg = "cannot use sparse input in 'SVC' trained on dense data"

    # 如果测试数据是稀疏的，断言会引发相应的错误信息
    if sparse.issparse(X_test):
        with pytest.raises(ValueError, match=msg):
            dense_svm.predict(X_test)


@skip_if_32bit  # 如果是32位系统，跳过该测试
@pytest.mark.parametrize(
    "X_train, y_train, X_test",
    [
        [X, Y, T],  # 参数化测试数据集：训练数据 X_train, 标签 y_train, 测试数据 X_test
        [X2, Y2, T2],  # 参数化测试数据集：训练数据 X2, 标签 Y2, 测试数据 T2
        [X_blobs[:80], y_blobs[:80], X_blobs[80:]],  # 参数化测试数据集：从 X_blobs 中划分训练和测试数据
        [iris.data, iris.target, iris.data],  # 参数化测试数据集：使用 iris 数据集的特征和标签作为训练和测试数据
    ],
)
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])  # 参数化测试：不同的核函数类型
@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + LIL_CONTAINERS)
def test_svc(X_train, y_train, X_test, kernel, sparse_container):
    """Check that sparse SVC gives the same result as SVC."""
    X_train = sparse_container(X_train)  # 使用稀疏容器对训练数据进行转换

    clf = svm.SVC(
        gamma=1,
        kernel=kernel,
        probability=True,
        random_state=0,
        decision_function_shape="ovo",
    )
    check_svm_model_equal(clf, X_train, y_train, X_test)  # 检查稀疏 SVC 和普通 SVC 的结果是否相同


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_unsorted_indices(csr_container):
    # test that the result with sorted and unsorted indices in csr is the same
    # we use a subset of digits as iris, blobs or make_classification didn't
    # show the problem
    X, y = load_digits(return_X_y=True)  # 载入手写数字数据集
    X_test = csr_container(X[50:100])  # 使用稀疏容器对测试数据进行转换
    X, y = X[:50], y[:50]  # 只使用前50个样本作为训练数据

    X_sparse = csr_container(X)  # 使用稀疏容器对训练数据进行转换
    coef_dense = (
        svm.SVC(kernel="linear", probability=True, random_state=0).fit(X, y).coef_
    )  # 训练普通 SVM 并获取其系数
    sparse_svc = svm.SVC(kernel="linear", probability=True, random_state=0).fit(
        X_sparse, y
    )  # 训练稀疏 SVM
    coef_sorted = sparse_svc.coef_  # 获取稀疏 SVM 的系数
    # make sure dense and sparse SVM give the same result
    assert_allclose(coef_dense, coef_sorted.toarray())  # 断言稠密和稀疏 SVM 的系数是否相同

    # reverse each row's indices
    def scramble_indices(X):
        new_data = []
        new_indices = []
        for i in range(1, len(X.indptr)):
            row_slice = slice(*X.indptr[i - 1 : i + 1])
            new_data.extend(X.data[row_slice][::-1])
            new_indices.extend(X.indices[row_slice][::-1])
        return csr_container((new_data, new_indices, X.indptr), shape=X.shape)

    X_sparse_unsorted = scramble_indices(X_sparse)  # 对稀疏数据进行索引混乱操作
    X_test_unsorted = scramble_indices(X_test)  # 对测试数据进行索引混乱操作

    assert not X_sparse_unsorted.has_sorted_indices  # 断言索引是否未排序
    assert not X_test_unsorted.has_sorted_indices  # 断言测试数据索引是否未排序

    unsorted_svc = svm.SVC(kernel="linear", probability=True, random_state=0).fit(
        X_sparse_unsorted, y
    )  # 训练使用混乱索引的稀疏 SVM
    coef_unsorted = unsorted_svc.coef_  # 获取混乱索引的稀疏 SVM 的系数
    # make sure unsorted indices give same result
    assert_allclose(coef_unsorted.toarray(), coef_sorted.toarray())  # 断言混乱索引和有序索引的系数是否相同
    assert_allclose(
        sparse_svc.predict_proba(X_test_unsorted), sparse_svc.predict_proba(X_test)
    )  # 断言混乱索引的预测概率与有序索引的预测概率是否相同


@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_svc_with_custom_kernel(lil_container):
    def kfunc(x, y):
        return safe_sparse_dot(x, y.T)  # 自定义核函数，计算稀疏数据的点积

    X_sp = lil_container(X)  # 使用 LIL 容器对数据进行转换
    clf_lin = svm.SVC(kernel="linear").fit(X_sp, Y)  # 训练普通 SVM
    clf_mylin = svm.SVC(kernel=kfunc).fit(X_sp, Y)  # 训练使用自定义核函数的 SVM
    assert_array_equal(clf_lin.predict(X_sp), clf_mylin.predict(X_sp))  # 断言两种 SVM 的预测结果相同


@skip_if_32bit
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf"])
def test_svc_iris(csr_container, kernel):
    # 使用不同的核函数进行测试，包括线性核、多项式核和高斯核
    # 准备稀疏表示的 iris 数据集
    iris_data_sp = csr_container(iris.data)

    # 使用指定的核函数创建稠密 SVC 模型，并拟合数据
    sp_clf = svm.SVC(kernel=kernel).fit(iris_data_sp, iris.target)
    clf = svm.SVC(kernel=kernel).fit(iris.data, iris.target)

    # 检查支持向量是否接近
    assert_allclose(clf.support_vectors_, sp_clf.support_vectors_.toarray())
    # 检查对偶系数是否接近
    assert_allclose(clf.dual_coef_, sp_clf.dual_coef_.toarray())
    # 检查预测结果是否一致
    assert_allclose(clf.predict(iris.data), sp_clf.predict(iris_data_sp))
    
    # 对于线性核，还需检查系数是否接近
    if kernel == "linear":
        assert_allclose(clf.coef_, sp_clf.coef_.toarray())


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_decision_function(csr_container):
    # 测试 decision_function 方法的实现是否正确

    # 测试多分类情况
    iris_data_sp = csr_container(iris.data)
    svc = svm.SVC(kernel="linear", C=0.1, decision_function_shape="ovo")
    clf = svc.fit(iris_data_sp, iris.target)

    # 计算决策函数的输出值
    dec = safe_sparse_dot(iris_data_sp, clf.coef_.T) + clf.intercept_

    # 检查计算的决策函数值是否接近预期值
    assert_allclose(dec, clf.decision_function(iris_data_sp))

    # 测试二分类情况
    clf.fit(X, Y)
    dec = np.dot(X, clf.coef_.T) + clf.intercept_
    prediction = clf.predict(X)

    # 检查计算的决策函数值和预测是否一致
    assert_allclose(dec.ravel(), clf.decision_function(X))
    assert_allclose(
        prediction, clf.classes_[(clf.decision_function(X) > 0).astype(int).ravel()]
    )

    # 预期的输出值
    expected = np.array([-1.0, -0.66, -1.0, 0.66, 1.0, 1.0])
    # 检查决策函数值是否接近预期输出值
    assert_array_almost_equal(clf.decision_function(X), expected, decimal=2)


@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_error(lil_container):
    # 测试对于不合格的输入是否能够正确地引发异常
    clf = svm.SVC()
    X_sp = lil_container(X)

    # 生成错误的标签维度
    Y2 = Y[:-1]  # wrong dimensions for labels
    # 应当引发 ValueError 异常
    with pytest.raises(ValueError):
        clf.fit(X_sp, Y2)

    # 正确情况下，应当能够拟合并预测结果
    clf.fit(X_sp, Y)
    assert_array_equal(clf.predict(T), true_result)


@pytest.mark.parametrize(
    "lil_container, dok_container", zip(LIL_CONTAINERS, DOK_CONTAINERS)
)
def test_linearsvc(lil_container, dok_container):
    # 类似于 test_SVC 的测试，但针对 LinearSVC

    # 准备稀疏表示的数据集
    X_sp = lil_container(X)
    X2_sp = dok_container(X2)

    # 创建并拟合 LinearSVC 模型
    clf = svm.LinearSVC(random_state=0).fit(X, Y)
    sp_clf = svm.LinearSVC(random_state=0).fit(X_sp, Y)

    # 检查是否启用了拟合截距
    assert sp_clf.fit_intercept

    # 检查系数是否接近
    assert_array_almost_equal(clf.coef_, sp_clf.coef_, decimal=4)
    # 检查截距是否接近
    assert_array_almost_equal(clf.intercept_, sp_clf.intercept_, decimal=4)

    # 检查预测结果是否一致
    assert_allclose(clf.predict(X), sp_clf.predict(X_sp))

    # 再次拟合并检查结果
    clf.fit(X2, Y2)
    sp_clf.fit(X2_sp, Y2)

    # 再次检查系数是否接近
    assert_array_almost_equal(clf.coef_, sp_clf.coef_, decimal=4)
    # 再次检查截距是否接近
    assert_array_almost_equal(clf.intercept_, sp_clf.intercept_, decimal=4)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_linearsvc_iris(csr_container):
    # 测试稀疏表示的 iris 数据集上的 LinearSVC
    iris_data_sp = csr_container(iris.data)
    # 使用线性支持向量机（LinearSVC）在稠密数据上训练分类器，使用iris_data_sp和iris.target
    sp_clf = svm.LinearSVC(random_state=0).fit(iris_data_sp, iris.target)
    # 使用线性支持向量机（LinearSVC）在完整数据上训练分类器，使用iris.data和iris.target
    clf = svm.LinearSVC(random_state=0).fit(iris.data, iris.target)

    # 断言两个分类器是否具有相同的fit_intercept属性
    assert clf.fit_intercept == sp_clf.fit_intercept

    # 断言两个分类器的coef_属性（系数）在小数点后一位精度上是否几乎相等
    assert_array_almost_equal(clf.coef_, sp_clf.coef_, decimal=1)
    # 断言两个分类器的intercept_属性（截距）在小数点后一位精度上是否几乎相等
    assert_array_almost_equal(clf.intercept_, sp_clf.intercept_, decimal=1)
    # 断言两个分类器在完整数据上进行预测是否产生几乎相同的结果
    assert_allclose(clf.predict(iris.data), sp_clf.predict(iris_data_sp))

    # 检查decision_function方法的输出
    pred = np.argmax(sp_clf.decision_function(iris_data_sp), axis=1)
    # 断言使用decision_function计算的预测结果与使用完整数据预测的结果是否几乎相等
    assert_allclose(pred, clf.predict(iris.data))

    # 在两个模型上稀疏化系数，并检查它们是否仍然产生相同的预测结果
    clf.sparsify()
    # 断言稀疏化后的clf模型在稀疏数据iris_data_sp上预测结果与之前计算的pred是否相等
    assert_array_equal(pred, clf.predict(iris_data_sp))
    sp_clf.sparsify()
    # 断言稀疏化后的sp_clf模型在稀疏数据iris_data_sp上预测结果与之前计算的pred是否相等
    assert_array_equal(pred, sp_clf.predict(iris_data_sp))
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest 的 parametrize 装饰器，对 csr_container 参数化测试
def test_weight(csr_container):
    # 测试类别权重
    X_, y_ = make_classification(
        n_samples=200, n_features=100, weights=[0.833, 0.167], random_state=0
    )
    # 创建一个样本特征为 100，样本数为 200 的分类数据集 X_ 和对应的标签 y_

    X_ = csr_container(X_)
    # 使用 csr_container 转换 X_

    for clf in (
        linear_model.LogisticRegression(),
        svm.LinearSVC(random_state=0),
        svm.SVC(),
    ):
        # 遍历三种分类器：LogisticRegression，LinearSVC 和 SVC

        clf.set_params(class_weight={0: 5})
        # 设置分类器的类别权重，将类别 0 的权重设为 5

        clf.fit(X_[:180], y_[:180])
        # 使用前 180 个样本进行训练

        y_pred = clf.predict(X_[180:])
        # 对剩余的样本进行预测

        assert np.sum(y_pred == y_[180:]) >= 11
        # 断言预测正确的样本数至少为 11


@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
# 使用 pytest 的 parametrize 装饰器，对 lil_container 参数化测试
def test_sample_weights(lil_container):
    # 测试个别样本的权重
    X_sp = lil_container(X)
    # 使用 lil_container 转换 X 成为稀疏矩阵

    clf = svm.SVC()
    # 创建一个 SVM 分类器

    clf.fit(X_sp, Y)
    # 使用转换后的稀疏矩阵 X_sp 和标签 Y 进行训练

    assert_array_equal(clf.predict([X[2]]), [1.0])
    # 断言对第三个样本进行预测的结果为 1.0

    sample_weight = [0.1] * 3 + [10] * 3
    # 设置样本权重，前三个样本权重为 0.1，后三个样本权重为 10

    clf.fit(X_sp, Y, sample_weight=sample_weight)
    # 使用样本权重重新训练分类器

    assert_array_equal(clf.predict([X[2]]), [2.0])
    # 断言对第三个样本进行预测的结果为 2.0


def test_sparse_liblinear_intercept_handling():
    # 测试稀疏 liblinear 是否遵循 intercept_scaling 参数
    test_svm.test_dense_liblinear_intercept_handling(svm.LinearSVC)


@pytest.mark.parametrize(
    "X_train, y_train, X_test",
    [
        [X, None, T],
        [X2, None, T2],
        [X_blobs[:80], None, X_blobs[80:]],
        [iris.data, None, iris.data],
    ],
)
# 使用 pytest 的 parametrize 装饰器，对多个参数进行参数化测试
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
# 对 kernel 参数化测试，值为 "linear", "poly", "rbf", "sigmoid"
@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + LIL_CONTAINERS)
# 对 sparse_container 参数化测试，值为 CSR_CONTAINERS 和 LIL_CONTAINERS 的组合
@skip_if_32bit
# 跳过 32 位系统的测试
def test_sparse_oneclasssvm(X_train, y_train, X_test, kernel, sparse_container):
    # 检查稀疏 OneClassSVM 是否与密集 OneClassSVM 得出相同结果
    X_train = sparse_container(X_train)
    # 使用 sparse_container 转换 X_train 成为稀疏格式

    clf = svm.OneClassSVM(gamma=1, kernel=kernel)
    # 创建一个 OneClassSVM 分类器

    check_svm_model_equal(clf, X_train, y_train, X_test)
    # 使用 check_svm_model_equal 函数检查分类器的结果是否与预期一致


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest 的 parametrize 装饰器，对 csr_container 参数化测试
def test_sparse_realdata(csr_container):
    # 在 20newsgroups 数据集的子集上进行测试
    # 这可以捕获输入未正确转换为稀疏格式或权重未正确初始化的一些错误
    data = np.array([0.03771744, 0.1003567, 0.01174647, 0.027069])
    # 创建一个 NumPy 数组作为输入数据

    # SVC 不支持大规模稀疏输入，所以我们指定 int32 类型的索引
    # 在这种情况下，`csr_matrix` 自动使用 int32，而不管 `indices` 和 `indptr` 的数据类型是什么
    # 但 `csr_array` 可能使用与 `indices` 和 `indptr` 不同的数据类型，如果未指定则为 int64
    indices = np.array([6, 5, 35, 31], dtype=np.int32)
    indptr = np.array([0] * 8 + [1] * 32 + [2] * 38 + [4] * 3, dtype=np.int32)
    # 创建 indices 和 indptr 数组，指定其数据类型为 int32

    X = csr_container((data, indices, indptr))
    # 使用 csr_container 将 data, indices, indptr 组合成稀疏矩阵
    # 创建一个包含浮点数的 NumPy 数组 y，用于存储分类标签
    y = np.array(
        [
            1.0,
            0.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            0.0,
            1.0,
            2.0,
            2.0,
            0.0,
            2.0,
            0.0,
            3.0,
            0.0,
            3.0,
            0.0,
            1.0,
            1.0,
            3.0,
            2.0,
            3.0,
            2.0,
            0.0,
            3.0,
            1.0,
            0.0,
            2.0,
            1.0,
            2.0,
            0.0,
            1.0,
            0.0,
            2.0,
            3.0,
            1.0,
            3.0,
            0.0,
            1.0,
            0.0,
            0.0,
            2.0,
            0.0,
            1.0,
            2.0,
            2.0,
            2.0,
            3.0,
            2.0,
            0.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            2.0,
            2.0,
            0.0,
            1.0,
            0.0,
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            2.0,
            2.0,
            1.0,
            3.0,
            1.0,
            1.0,
            0.0,
            1.0,
            2.0,
            1.0,
            1.0,
            3.0,
        ]
    )

    # 使用线性核函数创建支持向量机分类器 clf，并用稀疏表示的 X 训练数据拟合模型
    clf = svm.SVC(kernel="linear").fit(X.toarray(), y)
    
    # 使用线性核函数创建支持向量机分类器 sp_clf，并用 COO 格式的 X 训练数据拟合模型
    sp_clf = svm.SVC(kernel="linear").fit(X.tocoo(), y)

    # 断言两个分类器的支持向量相等，即支持向量和对偶系数的稀疏表示应当一致
    assert_array_equal(clf.support_vectors_, sp_clf.support_vectors_.toarray())
    assert_array_equal(clf.dual_coef_, sp_clf.dual_coef_.toarray())
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
# 使用 pytest 的 parametrize 装饰器，为每个 lil_container 参数化测试
def test_sparse_svc_clone_with_callable_kernel(lil_container):
    # 测试即使使用稀疏输入，也会调用 "dense_fit"，即一切正常运行。
    a = svm.SVC(C=1, kernel=lambda x, y: x @ y.T, probability=True, random_state=0)
    # 创建 SVC 对象 a，使用可调用的矩阵乘法核函数
    b = base.clone(a)
    # 克隆 SVC 对象 a，得到 b

    X_sp = lil_container(X)
    # 使用 lil_container 处理输入 X，得到稀疏矩阵 X_sp
    b.fit(X_sp, Y)
    # 对 b 应用 fit 方法，使用稀疏矩阵 X_sp 和目标值 Y 进行拟合
    pred = b.predict(X_sp)
    # 预测 b 在 X_sp 上的输出
    b.predict_proba(X_sp)
    # 预测 b 在 X_sp 上的概率估计

    dense_svm = svm.SVC(
        C=1, kernel=lambda x, y: np.dot(x, y.T), probability=True, random_state=0
    )
    # 创建另一个 SVC 对象 dense_svm，使用矩阵乘法核函数
    pred_dense = dense_svm.fit(X, Y).predict(X)
    # 对 dense_svm 应用 fit 方法，使用密集输入 X 和目标值 Y 进行拟合，并预测输出
    assert_array_equal(pred_dense, pred)
    # 断言预测结果相等
    # b.decision_function(X_sp)  # XXX : should be supported


@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
# 使用 pytest 的 parametrize 装饰器，为每个 lil_container 参数化测试
def test_timeout(lil_container):
    sp = svm.SVC(
        C=1, kernel=lambda x, y: x @ y.T, probability=True, random_state=0, max_iter=1
    )
    # 创建 SVC 对象 sp，使用矩阵乘法核函数和最大迭代次数为 1
    warning_msg = (
        r"Solver terminated early \(max_iter=1\).  Consider pre-processing "
        r"your data with StandardScaler or MinMaxScaler."
    )
    # 设置警告信息字符串
    with pytest.warns(ConvergenceWarning, match=warning_msg):
        # 捕获 ConvergenceWarning 警告，并匹配 warning_msg
        sp.fit(lil_container(X), Y)
        # 对 sp 应用 fit 方法，使用 lil_container 处理输入 X 和目标值 Y 进行拟合


def test_consistent_proba():
    a = svm.SVC(probability=True, max_iter=1, random_state=0)
    # 创建 SVC 对象 a，设置 probability 为 True，最大迭代次数为 1，随机状态为 0
    with ignore_warnings(category=ConvergenceWarning):
        # 忽略 ConvergenceWarning 类别的警告
        proba_1 = a.fit(X, Y).predict_proba(X)
        # 对 a 应用 fit 方法，使用输入 X 和目标值 Y 进行拟合，并预测输出概率
    a = svm.SVC(probability=True, max_iter=1, random_state=0)
    # 重新创建 SVC 对象 a，设置相同的参数
    with ignore_warnings(category=ConvergenceWarning):
        # 再次忽略 ConvergenceWarning 类别的警告
        proba_2 = a.fit(X, Y).predict_proba(X)
        # 对 a 应用 fit 方法，使用输入 X 和目标值 Y 进行拟合，并预测输出概率
    assert_allclose(proba_1, proba_2)
    # 断言两次预测的概率输出相等
```