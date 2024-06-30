# `D:\src\scipysrc\scikit-learn\sklearn\svm\tests\test_svm.py`

```
"""
Testing for Support Vector Machine module (sklearn.svm)

TODO: remove hard coded numerical results when possible
"""

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from sklearn import base, datasets, linear_model, metrics, svm
from sklearn.datasets import make_blobs, make_classification
from sklearn.exceptions import (
    ConvergenceWarning,
    NotFittedError,
    UndefinedMetricWarning,
)
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

# mypy error: Module 'sklearn.svm' has no attribute '_libsvm'
from sklearn.svm import (  # type: ignore
    SVR,
    LinearSVC,
    LinearSVR,
    NuSVR,
    OneClassSVM,
    _libsvm,
)
from sklearn.svm._classes import _validate_dual_parameter
from sklearn.utils import check_random_state, shuffle
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import _num_samples

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
Y = [1, 1, 1, 2, 2, 2]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [1, 2, 2]

# also load the iris dataset
iris = datasets.load_iris()
rng = check_random_state(42)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


def test_libsvm_parameters():
    # Test parameters on classes that make use of libsvm.
    # 使用线性核创建 SVM 分类器，并用样本 X, Y 进行训练
    clf = svm.SVC(kernel="linear").fit(X, Y)
    # 断言双系数的值，用于支持向量的线性组合
    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    # 断言支持向量的索引
    assert_array_equal(clf.support_, [1, 3])
    # 断言支持向量的值
    assert_array_equal(clf.support_vectors_, (X[1], X[3]))
    # 断言截距的值
    assert_array_equal(clf.intercept_, [0.0])
    # 断言预测结果与真实标签的一致性
    assert_array_equal(clf.predict(X), Y)


def test_libsvm_iris():
    # Check consistency on dataset iris.

    # shuffle the dataset so that labels are not ordered
    for k in ("linear", "rbf"):
        # 使用指定核函数创建 SVM 分类器，并用 iris 数据集进行训练
        clf = svm.SVC(kernel=k).fit(iris.data, iris.target)
        # 断言分类器在 iris 数据集上的预测准确率高于 0.9
        assert np.mean(clf.predict(iris.data) == iris.target) > 0.9
        # 当且仅当核函数为线性时，断言分类器具有 coef_ 属性
        assert hasattr(clf, "coef_") == (k == "linear")

    # 断言分类器的类别与排序后的类别数组一致
    assert_array_equal(clf.classes_, np.sort(clf.classes_))

    # check also the low-level API
    # We unpack the values to create a dictionary with some of the return values
    # from Libsvm's fit.
    (
        libsvm_support,
        libsvm_support_vectors,
        libsvm_n_class_SV,
        libsvm_sv_coef,
        libsvm_intercept,
        libsvm_probA,
        libsvm_probB,
        # libsvm_fit_status and libsvm_n_iter won't be used below.
        libsvm_fit_status,
        libsvm_n_iter,
    ) = _libsvm.fit(iris.data, iris.target.astype(np.float64))
    # 定义一个包含 Libsvm 模型参数的字典
    model_params = {
        "support": libsvm_support,           # 支持向量
        "SV": libsvm_support_vectors,       # 支持向量数据
        "nSV": libsvm_n_class_SV,           # 每个类的支持向量数目
        "sv_coef": libsvm_sv_coef,          # 支持向量的系数
        "intercept": libsvm_intercept,      # 模型的截距
        "probA": libsvm_probA,              # 模型的概率参数 A
        "probB": libsvm_probB,              # 模型的概率参数 B
    }
    # 使用 Libsvm 的预测函数进行预测
    pred = _libsvm.predict(iris.data, **model_params)
    # 断言预测精度大于 95%
    assert np.mean(pred == iris.target) > 0.95

    # 调用 Libsvm 的拟合函数，并解包返回值来创建一个字典
    (
        libsvm_support,                     # 支持向量
        libsvm_support_vectors,             # 支持向量数据
        libsvm_n_class_SV,                  # 每个类的支持向量数目
        libsvm_sv_coef,                     # 支持向量的系数
        libsvm_intercept,                   # 模型的截距
        libsvm_probA,                       # 模型的概率参数 A
        libsvm_probB,                       # 模型的概率参数 B
        # libsvm_fit_status and libsvm_n_iter 不会在后续使用。
        libsvm_fit_status,                  # 拟合状态（不使用）
        libsvm_n_iter,                      # 迭代次数（不使用）
    ) = _libsvm.fit(iris.data, iris.target.astype(np.float64), kernel="linear")

    # 更新模型参数字典
    model_params = {
        "support": libsvm_support,           # 支持向量
        "SV": libsvm_support_vectors,       # 支持向量数据
        "nSV": libsvm_n_class_SV,           # 每个类的支持向量数目
        "sv_coef": libsvm_sv_coef,          # 支持向量的系数
        "intercept": libsvm_intercept,      # 模型的截距
        "probA": libsvm_probA,              # 模型的概率参数 A
        "probB": libsvm_probB,              # 模型的概率参数 B
    }
    # 使用线性核函数进行预测
    pred = _libsvm.predict(iris.data, **model_params, kernel="linear")
    # 断言预测精度大于 95%
    assert np.mean(pred == iris.target) > 0.95

    # 使用线性核函数进行 5 折交叉验证预测
    pred = _libsvm.cross_validation(
        iris.data, iris.target.astype(np.float64), 5, kernel="linear", random_seed=0
    )
    # 断言交叉验证的平均预测精度大于 95%
    assert np.mean(pred == iris.target) > 0.95

    # 如果 random_seed >= 0，则设置 libsvm 的随机数种子，以确保得到确定性结果
    # （假设没有其它线程在调用这个函数时并发调用 `srand`）
    pred2 = _libsvm.cross_validation(
        iris.data, iris.target.astype(np.float64), 5, kernel="linear", random_seed=0
    )
    # 断言两次交叉验证的预测结果数组相等
    assert_array_equal(pred, pred2)
# 测试使用预计算核的支持向量机 (SVC)。
def test_precomputed():
    # 使用预计算核创建 SVC 对象
    clf = svm.SVC(kernel="precomputed")
    
    # 计算训练数据的 Gram 矩阵（方阵）
    K = np.dot(X, np.array(X).T)
    clf.fit(K, Y)
    
    # 计算测试数据的 Gram 矩阵（矩形矩阵）
    KT = np.dot(T, np.array(X).T)
    pred = clf.predict(KT)
    
    # 使用 pytest 检查预测时的值错误
    with pytest.raises(ValueError):
        clf.predict(KT.T)
    
    # 断言支持向量机的双重系数
    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    # 断言支持向量的索引
    assert_array_equal(clf.support_, [1, 3])
    # 断言截距
    assert_array_equal(clf.intercept_, [0])
    # 精确断言支持向量的索引
    assert_array_almost_equal(clf.support_, [1, 3])
    # 断言预测结果与真实结果的数组相等
    assert_array_equal(pred, true_result)

    # 重新初始化 KT 为全零矩阵，只计算支持向量的 KT[i,j]
    KT = np.zeros_like(KT)
    for i in range(len(T)):
        for j in clf.support_:
            KT[i, j] = np.dot(T[i], X[j])

    pred = clf.predict(KT)
    # 断言预测结果与真实结果的数组相等
    assert_array_equal(pred, true_result)

    # 使用一个可调用函数代替核矩阵来计算，这里使用线性核函数
    def kfunc(x, y):
        return np.dot(x, y.T)

    clf = svm.SVC(kernel=kfunc)
    clf.fit(np.array(X), Y)
    pred = clf.predict(T)

    # 断言支持向量机的双重系数
    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    # 断言截距
    assert_array_equal(clf.intercept_, [0])
    # 精确断言支持向量的索引
    assert_array_almost_equal(clf.support_, [1, 3])
    # 断言预测结果与真实结果的数组相等
    assert_array_equal(pred, true_result)

    # 使用预计算核在 iris 数据集上进行测试
    clf = svm.SVC(kernel="precomputed")
    clf2 = svm.SVC(kernel="linear")
    K = np.dot(iris.data, iris.data.T)
    clf.fit(K, iris.target)
    clf2.fit(iris.data, iris.target)
    pred = clf.predict(K)
    
    # 精确断言支持向量的索引
    assert_array_almost_equal(clf.support_, clf2.support_)
    # 精确断言支持向量的双重系数
    assert_array_almost_equal(clf.dual_coef_, clf2.dual_coef_)
    # 精确断言截距
    assert_array_almost_equal(clf.intercept_, clf2.intercept_)
    # 精确断言预测准确率达到 99%
    assert_almost_equal(np.mean(pred == iris.target), 0.99, decimal=2)

    # 重新初始化 K 为全零矩阵，只计算支持向量的 K[i,j]
    K = np.zeros_like(K)
    for i in range(len(iris.data)):
        for j in clf.support_:
            K[i, j] = np.dot(iris.data[i], iris.data[j])

    pred = clf.predict(K)
    # 精确断言预测准确率达到 99%
    assert_almost_equal(np.mean(pred == iris.target), 0.99, decimal=2)

    clf = svm.SVC(kernel=kfunc)
    clf.fit(iris.data, iris.target)
    # 精确断言预测准确率达到 99%
    assert_almost_equal(np.mean(pred == iris.target), 0.99, decimal=2)


# 测试支持向量回归 (SVR)
def test_svr():
    # 测试支持向量回归 (SVR)
    diabetes = datasets.load_diabetes()
    for clf in (
        svm.NuSVR(kernel="linear", nu=0.4, C=1.0),
        svm.NuSVR(kernel="linear", nu=0.4, C=10.0),
        svm.SVR(kernel="linear", C=10.0),
        svm.LinearSVR(C=10.0),
        svm.LinearSVR(C=10.0),
    ):
        clf.fit(diabetes.data, diabetes.target)
        # 断言回归得分大于 0.02
        assert clf.score(diabetes.data, diabetes.target) > 0.02

    # 非回归测试；以前 BaseLibSVM 会检查这个
    # 对于只有一个类别的情况（len(np.unique(y)) < 2），此操作仅适用于SVC（支持向量分类器）。
    svm.SVR().fit(diabetes.data, np.ones(len(diabetes.data)))
    # 对于只有一个类别的情况（len(np.unique(y)) < 2），此操作仅适用于LinearSVR（线性支持向量回归器）。
    svm.LinearSVR().fit(diabetes.data, np.ones(len(diabetes.data)))
def test_linearsvr():
    # 检查 SVR(kernel='linear') 和 LinearSVC() 是否给出了可比较的结果
    diabetes = datasets.load_diabetes()  # 载入糖尿病数据集
    lsvr = svm.LinearSVR(C=1e3).fit(diabetes.data, diabetes.target)  # 使用线性SVR拟合数据
    score1 = lsvr.score(diabetes.data, diabetes.target)  # 计算SVR模型的得分

    svr = svm.SVR(kernel="linear", C=1e3).fit(diabetes.data, diabetes.target)  # 使用线性核SVR拟合数据
    score2 = svr.score(diabetes.data, diabetes.target)  # 计算SVR模型的得分

    assert_allclose(np.linalg.norm(lsvr.coef_), np.linalg.norm(svr.coef_), 1, 0.0001)  # 检查两个模型的系数是否接近
    assert_almost_equal(score1, score2, 2)  # 检查两个模型的得分是否接近


def test_linearsvr_fit_sampleweight():
    # 检查当样本权重为1时是否得到正确结果
    # 检查 SVR(kernel='linear') 和 LinearSVC() 是否给出了可比较的结果
    diabetes = datasets.load_diabetes()  # 载入糖尿病数据集
    n_samples = len(diabetes.target)  # 获取样本数量
    unit_weight = np.ones(n_samples)  # 创建长度为样本数量的全1数组作为样本权重

    # 使用带有样本权重的线性SVR拟合数据
    lsvr = svm.LinearSVR(C=1e3, tol=1e-12, max_iter=10000).fit(
        diabetes.data, diabetes.target, sample_weight=unit_weight
    )
    score1 = lsvr.score(diabetes.data, diabetes.target)  # 计算带有样本权重的SVR模型得分

    # 使用不带样本权重的线性SVR拟合数据
    lsvr_no_weight = svm.LinearSVR(C=1e3, tol=1e-12, max_iter=10000).fit(
        diabetes.data, diabetes.target
    )
    score2 = lsvr_no_weight.score(diabetes.data, diabetes.target)  # 计算不带样本权重的SVR模型得分

    # 检查带权重和不带权重的模型系数是否接近
    assert_allclose(
        np.linalg.norm(lsvr.coef_), np.linalg.norm(lsvr_no_weight.coef_), 1, 0.0001
    )
    # 检查带权重和不带权重的模型得分是否接近
    assert_almost_equal(score1, score2, 2)

    # 检查 fit(X) = fit([X1, X2, X3], sample_weight=[n1, n2, n3]) 的情况
    # 其中 X = X1 重复 n1 次，X2 重复 n2 次，依此类推
    random_state = check_random_state(0)
    random_weight = random_state.randint(0, 10, n_samples)

    # 使用带有随机样本权重的线性SVR拟合数据
    lsvr_unflat = svm.LinearSVR(C=1e3, tol=1e-12, max_iter=10000).fit(
        diabetes.data, diabetes.target, sample_weight=random_weight
    )
    score3 = lsvr_unflat.score(
        diabetes.data, diabetes.target, sample_weight=random_weight
    )

    # 将数据和目标展开为带有随机样本权重的形式
    X_flat = np.repeat(diabetes.data, random_weight, axis=0)
    y_flat = np.repeat(diabetes.target, random_weight, axis=0)

    # 使用展开后的数据拟合线性SVR模型
    lsvr_flat = svm.LinearSVR(C=1e3, tol=1e-12, max_iter=10000).fit(X_flat, y_flat)
    score4 = lsvr_flat.score(X_flat, y_flat)

    # 检查带有随机样本权重和展开后的模型得分是否接近
    assert_almost_equal(score3, score4, 2)


def test_svr_errors():
    X = [[0.0], [1.0]]
    y = [0.0, 0.5]

    # 错误的核函数
    clf = svm.SVR(kernel=lambda x, y: np.array([[1.0]]))
    clf.fit(X, y)
    with pytest.raises(ValueError):
        clf.predict(X)


def test_oneclass():
    # 测试 OneClassSVM
    clf = svm.OneClassSVM()
    clf.fit(X)
    pred = clf.predict(T)

    # 断言预测结果与预期相等
    assert_array_equal(pred, [1, -1, -1])
    # 断言预测结果的数据类型是否为 intp
    assert pred.dtype == np.dtype("intp")
    # 断言截距的值与预期相等（精确到小数点后三位）
    assert_array_almost_equal(clf.intercept_, [-1.218], decimal=3)
    # 断言对偶系数的值与预期相等（精确到小数点后三位）
    assert_array_almost_equal(clf.dual_coef_, [[0.750, 0.750, 0.750, 0.750]], decimal=3)
    # 断言试图访问 coef_ 属性时会引发 AttributeError 异常
    with pytest.raises(AttributeError):
        (lambda: clf.coef_)()


def test_oneclass_decision_function():
    # 测试 OneClassSVM 的决策函数
    clf = svm.OneClassSVM()
    rnd = check_random_state(2)

    # 生成训练数据
    X = 0.3 * rnd.randn(100, 2)
    # 将数组 X 中的每个元素加上 2，然后与原始 X 数组合并
    X_train = np.r_[X + 2, X - 2]

    # 使用正态分布生成一个形状为 (20, 2) 的随机数据矩阵 X
    X = 0.3 * rnd.randn(20, 2)
    # 将数组 X 中的每个元素加上 2，然后与原始 X 数组合并，形成测试数据 X_test
    X_test = np.r_[X + 2, X - 2]

    # 使用均匀分布生成形状为 (20, 2) 的异常数据矩阵 X_outliers
    X_outliers = rnd.uniform(low=-4, high=4, size=(20, 2))

    # 使用 OneClassSVM 模型，设置参数 nu=0.1, kernel="rbf", gamma=0.1 进行拟合
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)

    # 对测试数据 X_test 进行预测
    y_pred_test = clf.predict(X_test)
    # 断言预测结果中标签为正常的占比大于 90%
    assert np.mean(y_pred_test == 1) > 0.9
    # 对异常数据 X_outliers 进行预测
    y_pred_outliers = clf.predict(X_outliers)
    # 断言预测结果中标签为异常的占比大于 90%
    assert np.mean(y_pred_outliers == -1) > 0.9

    # 获取测试数据 X_test 的决策函数输出值
    dec_func_test = clf.decision_function(X_test)
    # 断言决策函数输出大于 0 的结果与预测标签为正常的一致
    assert_array_equal((dec_func_test > 0).ravel(), y_pred_test == 1)

    # 获取异常数据 X_outliers 的决策函数输出值
    dec_func_outliers = clf.decision_function(X_outliers)
    # 断言决策函数输出大于 0 的结果与预测标签为异常的一致
    assert_array_equal((dec_func_outliers > 0).ravel(), y_pred_outliers == 1)
def test_oneclass_score_samples():
    X_train = [[1, 1], [1, 2], [2, 1]]
    clf = svm.OneClassSVM(gamma=1).fit(X_train)
    assert_array_equal(
        clf.score_samples([[2.0, 2.0]]),
        clf.decision_function([[2.0, 2.0]]) + clf.offset_,
    )


# 测试单类SVM的score_samples方法
# 设置训练数据集X_train
X_train = [[1, 1], [1, 2], [2, 1]]
# 使用gamma=1训练一个OneClassSVM模型clf
clf = svm.OneClassSVM(gamma=1).fit(X_train)
# 断言score_samples方法的输出等于decision_function方法与offset_的和
assert_array_equal(
    clf.score_samples([[2.0, 2.0]]),
    clf.decision_function([[2.0, 2.0]]) + clf.offset_,
)


def test_tweak_params():
    # Make sure some tweaking of parameters works.
    # We change clf.dual_coef_ at run time and expect .predict() to change
    # accordingly. Notice that this is not trivial since it involves a lot
    # of C/Python copying in the libsvm bindings.
    # The success of this test ensures that the mapping between libsvm and
    # the python classifier is complete.
    clf = svm.SVC(kernel="linear", C=1.0)
    clf.fit(X, Y)
    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    assert_array_equal(clf.predict([[-0.1, -0.1]]), [1])
    clf._dual_coef_ = np.array([[0.0, 1.0]])
    assert_array_equal(clf.predict([[-0.1, -0.1]]), [2])


# 测试调整参数后的SVC模型表现
# 创建一个线性核的SVC模型clf，设置参数C=1.0
clf = svm.SVC(kernel="linear", C=1.0)
# 使用数据集X和标签Y进行训练
clf.fit(X, Y)
# 断言模型的dual_coef_属性与[[-0.25, 0.25]]相等
assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
# 断言模型对[[-0.1, -0.1]]的预测结果为[1]
assert_array_equal(clf.predict([[-0.1, -0.1]]), [1])
# 将模型的_dual_coef_属性设置为[[0.0, 1.0]]
clf._dual_coef_ = np.array([[0.0, 1.0]])
# 断言模型对[[-0.1, -0.1]]的预测结果为[2]
assert_array_equal(clf.predict([[-0.1, -0.1]]), [2])


def test_probability():
    # Predict probabilities using SVC
    # This uses cross validation, so we use a slightly bigger testing set.

    for clf in (
        svm.SVC(probability=True, random_state=0, C=1.0),
        svm.NuSVC(probability=True, random_state=0),
    ):
        clf.fit(iris.data, iris.target)

        prob_predict = clf.predict_proba(iris.data)
        assert_array_almost_equal(np.sum(prob_predict, 1), np.ones(iris.data.shape[0]))
        assert np.mean(np.argmax(prob_predict, 1) == clf.predict(iris.data)) > 0.9

        assert_almost_equal(
            clf.predict_proba(iris.data), np.exp(clf.predict_log_proba(iris.data)), 8
        )


# 测试使用SVC预测概率
# 这里使用交叉验证，因此使用稍大的测试集合
for clf in (
    svm.SVC(probability=True, random_state=0, C=1.0),
    svm.NuSVC(probability=True, random_state=0),
):
    # 使用鸢尾花数据集iris的数据和标签进行模型训练
    clf.fit(iris.data, iris.target)

    # 预测每个样本的概率值
    prob_predict = clf.predict_proba(iris.data)
    # 断言每行概率值之和接近1
    assert_array_almost_equal(np.sum(prob_predict, 1), np.ones(iris.data.shape[0]))
    # 断言大部分样本的最大概率类别与预测类别一致的比例大于0.9
    assert np.mean(np.argmax(prob_predict, 1) == clf.predict(iris.data)) > 0.9

    # 断言使用predict_proba预测的概率与使用predict_log_proba的指数函数值接近
    assert_almost_equal(
        clf.predict_proba(iris.data), np.exp(clf.predict_log_proba(iris.data)), 8
    )


def test_decision_function():
    # Test decision_function
    # Sanity check, test that decision_function implemented in python
    # returns the same as the one in libsvm
    # multi class:
    clf = svm.SVC(kernel="linear", C=0.1, decision_function_shape="ovo").fit(
        iris.data, iris.target
    )

    dec = np.dot(iris.data, clf.coef_.T) + clf.intercept_

    assert_array_almost_equal(dec, clf.decision_function(iris.data))

    # binary:
    clf.fit(X, Y)
    dec = np.dot(X, clf.coef_.T) + clf.intercept_
    prediction = clf.predict(X)
    assert_array_almost_equal(dec.ravel(), clf.decision_function(X))
    assert_array_almost_equal(
        prediction, clf.classes_[(clf.decision_function(X) > 0).astype(int)]
    )
    expected = np.array([-1.0, -0.66, -1.0, 0.66, 1.0, 1.0])
    assert_array_almost_equal(clf.decision_function(X), expected, 2)

    # kernel binary:
    clf = svm.SVC(kernel="rbf", gamma=1, decision_function_shape="ovo")
    clf.fit(X, Y)

    rbfs = rbf_kernel(X, clf.support_vectors_, gamma=clf.gamma)
    dec = np.dot(rbfs, clf.dual_coef_.T) + clf.intercept_
    assert_array_almost_equal(dec.ravel(), clf.decision_function(X))


# 测试decision_function方法
# 检查实现在Python中的decision_function方法是否与libsvm中的一致
# 多分类情况:
clf = svm.SVC(kernel="linear", C=0.1, decision_function_shape="ovo").fit(
    iris.data, iris.target
)

dec = np.dot(iris.data, clf.coef_.T) + clf.intercept_

# 断言计算得到的决策函数值与模型的decision_function方法输出接近
assert_array_almost_equal(dec, clf.decision_function(iris.data))

# 二分类情况:
clf.fit(X, Y)
dec = np.dot(X, clf.coef_.T) + clf.intercept_
prediction = clf.predict(X)
# 断言计算得到的决策函数值与模型的decision_function方法输出接近
assert_array_almost_equal(dec.ravel(), clf.decision_function(X))
# 断言预测结果与决策函数大于0的类别一致
assert_array_almost_equal(
    prediction, clf.classes_[(clf.decision_function(X) > 0).astype(int)]
)
expected = np.array([-1.0, -0.66, -1.0, 0.66, 1.0, 1.0])
# 断言计算得到的决策函数值与期望值接近
assert_array_almost_equal(clf.decision_function(X), expected, 2)

# 核函数的二分类情况:
clf = svm.SVC(kernel="rbf", gamma=1, decision_function_shape="ovo")
clf.fit(X, Y)

rbfs = rbf_kernel(X, clf.support_vectors_, gamma=clf.gamma)
dec = np.dot(rbfs, clf.dual_coef_.T) + clf.intercept_
# 断言计算得到的决策函数值与模型的decision_function方法输出接近
assert_array_almost_equal(dec.ravel(), clf.decision_function(X))


@pytest.mark.parametrize("SVM", (svm.SVC, svm.NuSVC))
def test_decision_function_shape(SVM):
    # check that decision_function_shape='ovr' or 'ovo' gives
    # correct shape and is consistent with predict


# 测试decision_function_shape参数
# 检查当decision_function_shape设置为'ovr'或'ovo'时，决策函数的形状和与predict方法的一致性
    # 使用线性核和一对多策略创建 SVM 分类器，使用 iris 数据集进行训练
    clf = SVM(kernel="linear", decision_function_shape="ovr").fit(
        iris.data, iris.target
    )
    # 计算决策函数值
    dec = clf.decision_function(iris.data)
    # 断言决策函数值的形状符合预期，即每个样本有3个类别的决策函数值
    assert dec.shape == (len(iris.data), 3)
    # 断言分类器预测结果与决策函数值中最大值对应的类别索引一致
    assert_array_equal(clf.predict(iris.data), np.argmax(dec, axis=1))

    # 使用五个类别创建数据集
    X, y = make_blobs(n_samples=80, centers=5, random_state=0)
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 使用线性核和一对多策略创建 SVM 分类器，使用训练集进行训练
    clf = SVM(kernel="linear", decision_function_shape="ovr").fit(X_train, y_train)
    # 计算测试集的决策函数值
    dec = clf.decision_function(X_test)
    # 断言决策函数值的形状符合预期，即每个测试样本有5个类别的决策函数值
    assert dec.shape == (len(X_test), 5)
    # 断言分类器在测试集上的预测结果与决策函数值中最大值对应的类别索引一致
    assert_array_equal(clf.predict(X_test), np.argmax(dec, axis=1))

    # 检查在使用一对一策略时决策函数的形状
    clf = SVM(kernel="linear", decision_function_shape="ovo").fit(X_train, y_train)
    # 计算训练集上的决策函数值
    dec = clf.decision_function(X_train)
    # 断言决策函数值的形状符合预期，即每个训练样本有10个类别的决策函数值（一对一策略）
    assert dec.shape == (len(X_train), 10)
def test_svr_predict():
    # Test SVR's decision_function
    # 对 SVR 的 decision_function 进行测试
    # Sanity check, test that predict implemented in python
    # 进行健全性检查，测试 Python 实现的 predict 是否与 libsvm 中的相同

    X = iris.data
    y = iris.target

    # linear kernel
    # 使用线性核函数进行支持向量回归
    reg = svm.SVR(kernel="linear", C=0.1).fit(X, y)

    dec = np.dot(X, reg.coef_.T) + reg.intercept_
    assert_array_almost_equal(dec.ravel(), reg.predict(X).ravel())

    # rbf kernel
    # 使用径向基核函数进行支持向量回归
    reg = svm.SVR(kernel="rbf", gamma=1).fit(X, y)

    rbfs = rbf_kernel(X, reg.support_vectors_, gamma=reg.gamma)
    dec = np.dot(rbfs, reg.dual_coef_.T) + reg.intercept_
    assert_array_almost_equal(dec.ravel(), reg.predict(X).ravel())


def test_weight():
    # Test class weights
    # 测试类别权重功能
    clf = svm.SVC(class_weight={1: 0.1})
    # we give a small weights to class 1
    # 给类别 1 分配一个较小的权重
    clf.fit(X, Y)
    # so all predicted values belong to class 2
    # 所有预测值都属于类别 2
    assert_array_almost_equal(clf.predict(X), [2] * 6)

    X_, y_ = make_classification(
        n_samples=200, n_features=10, weights=[0.833, 0.167], random_state=2
    )

    for clf in (
        linear_model.LogisticRegression(),
        svm.LinearSVC(random_state=0),
        svm.SVC(),
    ):
        clf.set_params(class_weight={0: 0.1, 1: 10})
        clf.fit(X_[:100], y_[:100])
        y_pred = clf.predict(X_[100:])
        assert f1_score(y_[100:], y_pred) > 0.3


@pytest.mark.parametrize("estimator", [svm.SVC(C=1e-2), svm.NuSVC()])
def test_svm_classifier_sided_sample_weight(estimator):
    # fit a linear SVM and check that giving more weight to opposed samples
    # in the space will flip the decision toward these samples.
    # 拟合一个线性 SVM，并检查给予对立样本更高的权重是否会使决策向这些样本倾斜。
    X = [[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 0]]
    estimator.set_params(kernel="linear")

    # check that with unit weights, a sample is supposed to be predicted on
    # the boundary
    # 检查在单位权重的情况下，样本是否应该被预测在边界上
    sample_weight = [1] * 6
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.decision_function([[-1.0, 1.0]])
    assert y_pred == pytest.approx(0)

    # give more weights to opposed samples
    # 给对立样本更高的权重
    sample_weight = [10.0, 0.1, 0.1, 0.1, 0.1, 10]
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.decision_function([[-1.0, 1.0]])
    assert y_pred < 0

    sample_weight = [1.0, 0.1, 10.0, 10.0, 0.1, 0.1]
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.decision_function([[-1.0, 1.0]])
    assert y_pred > 0


@pytest.mark.parametrize("estimator", [svm.SVR(C=1e-2), svm.NuSVR(C=1e-2)])
def test_svm_regressor_sided_sample_weight(estimator):
    # similar test to test_svm_classifier_sided_sample_weight but for
    # SVM regressors
    # 类似于 test_svm_classifier_sided_sample_weight 的测试，但用于 SVM 回归器
    X = [[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 0]]
    estimator.set_params(kernel="linear")

    # check that with unit weights, a sample is supposed to be predicted on
    # the boundary
    # 检查在单位权重的情况下，样本是否应该被预测在边界上
    sample_weight = [1] * 6
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.predict([[-1.0, 1.0]])
    assert y_pred == pytest.approx(1.5)
    # 给反对样本分配更高的权重
    sample_weight = [10.0, 0.1, 0.1, 0.1, 0.1, 10]
    # 使用给定的权重训练估算器（estimator）
    estimator.fit(X, Y, sample_weight=sample_weight)
    # 对输入数据[[-1.0, 1.0]]进行预测
    y_pred = estimator.predict([[-1.0, 1.0]])
    # 断言预测结果应小于1.5
    assert y_pred < 1.5
    
    # 更新样本权重，重新分配
    sample_weight = [1.0, 0.1, 10.0, 10.0, 0.1, 0.1]
    # 使用更新后的权重重新训练估算器
    estimator.fit(X, Y, sample_weight=sample_weight)
    # 对输入数据[[-1.0, 1.0]]再次进行预测
    y_pred = estimator.predict([[-1.0, 1.0]])
    # 断言预测结果应大于1.5
    assert y_pred > 1.5
def test_svm_equivalence_sample_weight_C():
    # 测试：重新缩放所有样本等同于改变 C 参数
    clf = svm.SVC()
    # 使用无样本权重训练分类器
    clf.fit(X, Y)
    # 保存无样本权重的双重系数
    dual_coef_no_weight = clf.dual_coef_
    # 设置 C 参数为 100
    clf.set_params(C=100)
    # 使用重复值为 0.01 的样本权重重新训练分类器
    clf.fit(X, Y, sample_weight=np.repeat(0.01, len(X)))
    # 断言：验证无样本权重的双重系数与有样本权重的双重系数的近似相等
    assert_allclose(dual_coef_no_weight, clf.dual_coef_)


@pytest.mark.parametrize(
    "Estimator, err_msg",
    [
        (svm.SVC, "Invalid input - all samples have zero or negative weights."),
        (svm.NuSVC, "(negative dimensions are not allowed|nu is infeasible)"),
        (svm.SVR, "Invalid input - all samples have zero or negative weights."),
        (svm.NuSVR, "Invalid input - all samples have zero or negative weights."),
        (svm.OneClassSVM, "Invalid input - all samples have zero or negative weights."),
    ],
    ids=["SVC", "NuSVC", "SVR", "NuSVR", "OneClassSVM"],
)
@pytest.mark.parametrize(
    "sample_weight",
    [[0] * len(Y), [-0.3] * len(Y)],
    ids=["weights-are-zero", "weights-are-negative"],
)
def test_negative_sample_weights_mask_all_samples(Estimator, err_msg, sample_weight):
    # 创建指定估算器类型的实例
    est = Estimator(kernel="linear")
    # 使用 pytest 断言检查是否会引发 ValueError 异常，并匹配指定错误消息
    with pytest.raises(ValueError, match=err_msg):
        # 使用给定的样本权重训练估算器实例
        est.fit(X, Y, sample_weight=sample_weight)


@pytest.mark.parametrize(
    "Classifier, err_msg",
    [
        (
            svm.SVC,
            (
                "Invalid input - all samples with positive weights belong to the same"
                " class"
            ),
        ),
        (svm.NuSVC, "specified nu is infeasible"),
    ],
    ids=["SVC", "NuSVC"],
)
@pytest.mark.parametrize(
    "sample_weight",
    [[0, -0.5, 0, 1, 1, 1], [1, 1, 1, 0, -0.1, -0.3]],
    ids=["mask-label-1", "mask-label-2"],
)
def test_negative_weights_svc_leave_just_one_label(Classifier, err_msg, sample_weight):
    # 创建指定分类器类型的实例
    clf = Classifier(kernel="linear")
    # 使用 pytest 断言检查是否会引发 ValueError 异常，并匹配指定错误消息
    with pytest.raises(ValueError, match=err_msg):
        # 使用给定的样本权重训练分类器实例
        clf.fit(X, Y, sample_weight=sample_weight)


@pytest.mark.parametrize(
    "Classifier, model",
    [
        (svm.SVC, {"when-left": [0.3998, 0.4], "when-right": [0.4, 0.3999]}),
        (svm.NuSVC, {"when-left": [0.3333, 0.3333], "when-right": [0.3333, 0.3333]}),
    ],
    ids=["SVC", "NuSVC"],
)
@pytest.mark.parametrize(
    "sample_weight, mask_side",
    [([1, -0.5, 1, 1, 1, 1], "when-left"), ([1, 1, 1, 0, 1, 1], "when-right")],
    ids=["partial-mask-label-1", "partial-mask-label-2"],
)
def test_negative_weights_svc_leave_two_labels(
    Classifier, model, sample_weight, mask_side
):
    # 创建指定分类器类型的实例
    clf = Classifier(kernel="linear")
    # 使用给定的样本权重训练分类器实例
    clf.fit(X, Y, sample_weight=sample_weight)
    # 断言：验证分类器的系数与预期模型匹配
    assert_allclose(clf.coef_, [model[mask_side]], rtol=1e-3)


@pytest.mark.parametrize(
    "Estimator", [svm.SVC, svm.NuSVC, svm.NuSVR], ids=["SVC", "NuSVC", "NuSVR"]
)
@pytest.mark.parametrize(
    "sample_weight",
    [[1, -0.5, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1]],
    ids=["partial-mask-label-1", "partial-mask-label-2"],
)
def test_negative_weight_equal_coeffs(Estimator, sample_weight):
    # 创建指定估算器类型的实例
    clf = Estimator(kernel="linear")
    # 使用给定的样本权重训练估算器实例
    clf.fit(X, Y, sample_weight=sample_weight)
    # 创建一个线性核函数的估计器对象
    est = Estimator(kernel="linear")
    # 使用给定的数据 X 和 Y 进行拟合，同时考虑样本权重 sample_weight
    est.fit(X, Y, sample_weight=sample_weight)
    # 获取拟合后的系数，并将其绝对值展平成一维数组
    coef = np.abs(est.coef_).ravel()
    # 断言：第一个系数应该与第二个系数的近似相等，相对误差为 1e-3
    assert coef[0] == pytest.approx(coef[1], rel=1e-3)
# 忽略未定义度量警告类别，通常用于测试
@ignore_warnings(category=UndefinedMetricWarning)
# 测试自动权重设置函数
def test_auto_weight():
    # 从 sklearn 库中导入 LogisticRegression 类
    from sklearn.linear_model import LogisticRegression

    # 从 sklearn.utils 中导入 compute_class_weight 函数，用于计算类别权重
    from sklearn.utils import compute_class_weight

    # 从 iris 数据集中获取前两列特征和标签，将标签加一以确保不是回归测试
    X, y = iris.data[:, :2], iris.target + 1

    # 创建一个非平衡数据集 unbalanced，移除类别为 1 的一半预测器
    unbalanced = np.delete(np.arange(y.size), np.where(y > 2)[0][::2])

    # 获取非平衡数据集 unbalanced 中唯一的类别
    classes = np.unique(y[unbalanced])

    # 计算平衡类别权重
    class_weights = compute_class_weight("balanced", classes=classes, y=y[unbalanced])

    # 断言最大权重对应的类别索引为 2
    assert np.argmax(class_weights) == 2

    # 对于以下分类器，验证设置 class_weight="balanced" 时分数更好
    for clf in (
        svm.SVC(kernel="linear"),
        svm.LinearSVC(random_state=0),
        LogisticRegression(),
    ):
        # 使用非平衡数据集训练 clf 模型并预测
        y_pred = clf.fit(X[unbalanced], y[unbalanced]).predict(X)

        # 设置 clf 模型参数 class_weight="balanced" 并重新训练
        clf.set_params(class_weight="balanced")
        y_pred_balanced = clf.fit(
            X[unbalanced],
            y[unbalanced],
        ).predict(X)

        # 断言使用 balanced class_weight 时的 F1 分数大于等于未使用时的 F1 分数
        assert metrics.f1_score(y, y_pred, average="macro") <= metrics.f1_score(
            y, y_pred_balanced, average="macro"
        )


# 使用参数化测试框架对 LIL_CONTAINERS 中的每个容器运行测试
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_bad_input(lil_container):
    # 检查标签维度是否正确，创建 Y2 作为错误维度的标签副本
    Y2 = Y[:-1]  # 错误维度的标签
    with pytest.raises(ValueError):
        # 断言 svm.SVC() 对 X, Y2 抛出 ValueError
        svm.SVC().fit(X, Y2)

    # 对于数组非连续的情况进行测试
    for clf in (svm.SVC(), svm.LinearSVC(random_state=0)):
        # 将 X 转换为 Fortran 风格数组 Xf
        Xf = np.asfortranarray(X)
        assert not Xf.flags["C_CONTIGUOUS"]

        # 将 Y 复制为 Fortran 风格数组 yf
        yf = np.ascontiguousarray(np.tile(Y, (2, 1)).T)
        yf = yf[:, -1]
        assert not yf.flags["F_CONTIGUOUS"]
        assert not yf.flags["C_CONTIGUOUS"]

        # 使用 clf 模型训练 Xf, yf 并断言预测结果与真实结果相等
        clf.fit(Xf, yf)
        assert_array_equal(clf.predict(T), true_result)

    # 对于预先计算核函数的错误情况进行测试
    clf = svm.SVC(kernel="precomputed")
    with pytest.raises(ValueError):
        # 断言 svm.SVC() 对 X, Y 抛出 ValueError
        clf.fit(X, Y)

    # 当使用稠密训练数据，预测稀疏输入时抛出错误的测试
    clf = svm.SVC().fit(X, Y)
    with pytest.raises(ValueError):
        # 断言 clf.predict(lil_container(X)) 抛出 ValueError
        clf.predict(lil_container(X))

    # 使用 Xt 训练 SVC 模型，预测 X 时抛出错误的测试
    Xt = np.array(X).T
    clf.fit(np.dot(X, Xt), Y)
    with pytest.raises(ValueError):
        # 断言 clf.predict(X) 抛出 ValueError
        clf.predict(X)

    # 训练 SVC 模型，预测 Xt 时抛出错误的测试
    clf = svm.SVC()
    clf.fit(X, Y)
    with pytest.raises(ValueError):
        # 断言 clf.predict(Xt) 抛出 ValueError
        clf.predict(Xt)


# 检查 SVC 在处理非有限参数值时是否抛出 ValueError
def test_svc_nonfinite_params():
    # 设置随机种子和样本数
    rng = np.random.RandomState(0)
    n_samples = 10

    # 创建非有限参数值的数组 X
    fmax = np.finfo(np.float64).max
    X = fmax * rng.uniform(size=(n_samples, 2))
    y = rng.randint(0, 2, size=n_samples)

    # 初始化 SVC 模型
    clf = svm.SVC()

    # 断言当 dual coefficients 或 intercepts 非有限时，fit 函数抛出 ValueError
    msg = "The dual coefficients or intercepts are not finite"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)
def test_unicode_kernel():
    # 测试一个Unicode内核名称不会导致TypeError
    # 创建一个SVC分类器，使用线性内核，并启用概率估计
    clf = svm.SVC(kernel="linear", probability=True)
    # 使用训练数据X和标签Y拟合分类器
    clf.fit(X, Y)
    # 对测试数据T进行预测并返回概率估计
    clf.predict_proba(T)
    # 使用_libsvm.cross_validation进行交叉验证，使用线性内核，随机种子为0
    _libsvm.cross_validation(
        iris.data, iris.target.astype(np.float64), 5, kernel="linear", random_seed=0
    )


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_precomputed(csr_container):
    # 测试稀疏预计算内核的情况
    # 创建一个SVC分类器，使用预计算内核
    clf = svm.SVC(kernel="precomputed")
    # 生成稀疏的Gram矩阵
    sparse_gram = csr_container([[1, 0], [0, 1]])
    # 断言拟合过程中会抛出TypeError，并且错误信息中包含"Sparse precomputed"
    with pytest.raises(TypeError, match="Sparse precomputed"):
        clf.fit(sparse_gram, [0, 1])


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_fit_support_vectors_empty(csr_container):
    # 对空支持向量的拟合的回归测试，用于问题 #14893
    # 创建训练数据的稀疏表示
    X_train = csr_container([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    y_train = np.array([0.04, 0.04, 0.10, 0.16])
    # 创建一个SVR回归模型，使用线性内核
    model = svm.SVR(kernel="linear")
    # 使用训练数据X_train和标签y_train拟合模型
    model.fit(X_train, y_train)
    # 断言支持向量数据为空
    assert not model.support_vectors_.data.size
    # 断言对偶系数数据为空
    assert not model.dual_coef_.data.size


@pytest.mark.parametrize("loss", ["hinge", "squared_hinge"])
@pytest.mark.parametrize("penalty", ["l1", "l2"])
@pytest.mark.parametrize("dual", [True, False])
def test_linearsvc_parameters(loss, penalty, dual):
    # 测试LinearSVC中可能的参数组合
    # 生成包含可能参数组合的列表
    X, y = make_classification(n_samples=5, n_features=5, random_state=0)

    # 创建一个LinearSVC分类器，使用指定的penalty、loss、dual和随机种子
    clf = svm.LinearSVC(penalty=penalty, loss=loss, dual=dual, random_state=0)
    # 如果参数组合满足以下条件，预期会抛出ValueError，并且错误信息中包含具体参数的组合
    if (
        (loss, penalty) == ("hinge", "l1")
        or (loss, penalty, dual) == ("hinge", "l2", False)
        or (penalty, dual) == ("l1", True)
    ):
        with pytest.raises(
            ValueError,
            match="Unsupported set of arguments.*penalty='%s.*loss='%s.*dual=%s"
            % (penalty, loss, dual),
        ):
            clf.fit(X, y)
    else:
        # 使用训练数据X和标签y拟合分类器
        clf.fit(X, y)


def test_linearsvc():
    # 测试使用LinearSVC的基本例程
    # 创建一个LinearSVC分类器，使用默认参数和随机种子
    clf = svm.LinearSVC(random_state=0).fit(X, Y)

    # 默认情况下应该有截距
    assert clf.fit_intercept

    # 断言预测结果与真实结果数组相等
    assert_array_equal(clf.predict(T), true_result)
    # 断言截距数组接近0
    assert_array_almost_equal(clf.intercept_, [0], decimal=3)

    # 使用l1惩罚和squared_hinge损失的情况
    clf = svm.LinearSVC(
        penalty="l1", loss="squared_hinge", dual=False, random_state=0
    ).fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)

    # 使用l2惩罚和对偶形式的情况
    clf = svm.LinearSVC(penalty="l2", dual=True, random_state=0).fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)

    # 使用l2惩罚和hinge损失的情况
    clf = svm.LinearSVC(penalty="l2", loss="hinge", dual=True, random_state=0)
    clf.fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)

    # 还测试决策函数
    dec = clf.decision_function(T)
    res = (dec > 0).astype(int) + 1
    assert_array_equal(res, true_result)


def test_linearsvc_crammer_singer():
    # Test LinearSVC with crammer_singer multi-class svm
    
    # 使用 LinearSVC 模型进行单类别对多类别的 SVM 分类（One-vs-Rest）
    # 使用 iris 数据集训练 ovr_clf 模型
    ovr_clf = svm.LinearSVC(random_state=0).fit(iris.data, iris.target)
    
    # 使用 LinearSVC 模型进行 crammer_singer 多类别的 SVM 分类
    # 使用 iris 数据集训练 cs_clf 模型
    cs_clf = svm.LinearSVC(multi_class="crammer_singer", random_state=0)
    cs_clf.fit(iris.data, iris.target)
    
    # 对 ovr 和 crammer-singer 进行类似的预测:
    # 断言 ovr_clf 和 cs_clf 的预测结果相等的比例大于 0.9
    assert (ovr_clf.predict(iris.data) == cs_clf.predict(iris.data)).mean() > 0.9
    
    # 确保 ovr_clf 和 cs_clf 的分类器不同
    assert (ovr_clf.coef_ != cs_clf.coef_).all()
    
    # 测试决策函数
    # 断言 cs_clf 的预测结果与通过决策函数计算出的类别最大值索引一致
    assert_array_equal(
        cs_clf.predict(iris.data),
        np.argmax(cs_clf.decision_function(iris.data), axis=1),
    )
    
    # 计算决策函数值并进行近似相等的断言
    dec_func = np.dot(iris.data, cs_clf.coef_.T) + cs_clf.intercept_
    assert_array_almost_equal(dec_func, cs_clf.decision_function(iris.data))
def test_linearsvc_fit_sampleweight():
    # check correct result when sample_weight is 1
    # 获取样本数量
    n_samples = len(X)
    # 创建具有单位权重的数组
    unit_weight = np.ones(n_samples)
    # 使用默认参数训练 LinearSVC 模型
    clf = svm.LinearSVC(random_state=0).fit(X, Y)
    # 使用单位权重训练 LinearSVC 模型
    clf_unitweight = svm.LinearSVC(random_state=0, tol=1e-12, max_iter=1000).fit(
        X, Y, sample_weight=unit_weight
    )

    # 检查预测结果是否与 sample_weight=None 时相同
    assert_array_equal(clf_unitweight.predict(T), clf.predict(T))
    # 检查权重为 1 时的系数是否非常接近于默认条件下的系数
    assert_allclose(clf.coef_, clf_unitweight.coef_, 1, 0.0001)

    # 检查 fit(X) = fit([X1, X2, X3], sample_weight=[n1, n2, n3]) 的情况
    # 其中 X = X1 重复 n1 次, X2 重复 n2 次，依此类推
    random_state = check_random_state(0)
    # 创建随机权重数组
    random_weight = random_state.randint(0, 10, n_samples)
    # 使用随机权重训练 LinearSVC 模型
    lsvc_unflat = svm.LinearSVC(random_state=0, tol=1e-12, max_iter=1000).fit(
        X, Y, sample_weight=random_weight
    )

    # 对 T 进行预测
    pred1 = lsvc_unflat.predict(T)

    # 将 X 按照随机权重展开
    X_flat = np.repeat(X, random_weight, axis=0)
    y_flat = np.repeat(Y, random_weight, axis=0)
    # 使用展开后的数据训练 LinearSVC 模型
    lsvc_flat = svm.LinearSVC(random_state=0, tol=1e-12, max_iter=1000).fit(
        X_flat, y_flat
    )
    # 对 T 进行预测
    pred2 = lsvc_flat.predict(T)

    # 检查两种方法预测结果是否一致
    assert_array_equal(pred1, pred2)
    # 检查两种方法的系数是否非常接近
    assert_allclose(lsvc_unflat.coef_, lsvc_flat.coef_, 1, 0.0001)


def test_crammer_singer_binary():
    # Test Crammer-Singer formulation in the binary case
    # 创建二分类数据集
    X, y = make_classification(n_classes=2, random_state=0)

    # 遍历拟合截距参数为 True 和 False 的情况
    for fit_intercept in (True, False):
        # 创建 LinearSVC 模型，使用 Crammer-Singer 多分类策略
        # 计算分类准确率
        acc = (
            svm.LinearSVC(
                fit_intercept=fit_intercept,
                multi_class="crammer_singer",
                random_state=0,
            )
            .fit(X, y)
            .score(X, y)
        )
        # 断言准确率大于 0.9
        assert acc > 0.9


def test_linearsvc_iris():
    # Test that LinearSVC gives plausible predictions on the iris dataset
    # Also, test symbolic class names (classes_).
    # 使用 LinearSVC 对鸢尾花数据集进行训练
    target = iris.target_names[iris.target]
    clf = svm.LinearSVC(random_state=0).fit(iris.data, target)
    # 断言分类器的类别与数据集中的类别一致
    assert set(clf.classes_) == set(iris.target_names)
    # 断言预测准确率大于 0.8
    assert np.mean(clf.predict(iris.data) == target) > 0.8

    # 计算决策函数的值
    dec = clf.decision_function(iris.data)
    # 根据决策函数预测类别
    pred = iris.target_names[np.argmax(dec, 1)]
    # 断言预测结果与分类器预测结果一致
    assert_array_equal(pred, clf.predict(iris.data))


def test_dense_liblinear_intercept_handling(classifier=svm.LinearSVC):
    # Test that dense liblinear honours intercept_scaling param
    # 创建示例数据 X 和 y
    X = [[2, 1], [3, 1], [1, 3], [2, 3]]
    y = [0, 0, 1, 1]
    # 创建 LinearSVC 分类器对象
    clf = classifier(
        fit_intercept=True,
        penalty="l1",
        loss="squared_hinge",
        dual=False,
        C=4,
        tol=1e-7,
        random_state=0,
    )
    # 断言分类器的截距缩放参数为 1
    assert clf.intercept_scaling == 1, clf.intercept_scaling
    # 断言分类器使用了截距
    assert clf.fit_intercept

    # 当截距缩放参数较小时，截距值受正则化的影响较大
    clf.intercept_scaling = 1
    clf.fit(X, y)
    # 断言截距值接近于 0
    assert_almost_equal(clf.intercept_, 0, decimal=5)

    # 当截距缩放参数足够大时，截距值
    # 设置分类器的截距缩放系数为100，用于不受正则化影响的情况
    clf.intercept_scaling = 100
    # 使用训练数据 X 和标签 y 对分类器进行训练
    clf.fit(X, y)
    # 获取训练后的分类器截距值
    intercept1 = clf.intercept_
    # 断言截距值应小于-1
    assert intercept1 < -1

    # 当截距缩放系数足够高时，截距值不再依赖于截距缩放系数的具体值
    clf.intercept_scaling = 1000
    # 重新使用训练数据 X 和标签 y 对分类器进行训练
    clf.fit(X, y)
    # 获取新的截距值
    intercept2 = clf.intercept_
    # 使用数值数组近似判断两个截距值是否相等，精度为小数点后两位
    assert_array_almost_equal(intercept1, intercept2, decimal=2)
def test_liblinear_set_coef():
    # multi-class case
    # 使用 LinearSVC 模型对鸢尾花数据进行训练
    clf = svm.LinearSVC().fit(iris.data, iris.target)
    # 获取决策函数的值
    values = clf.decision_function(iris.data)
    # 复制 coef_ 和 intercept_ 属性以确保修改不会影响原始对象
    clf.coef_ = clf.coef_.copy()
    clf.intercept_ = clf.intercept_.copy()
    # 再次计算决策函数的值
    values2 = clf.decision_function(iris.data)
    # 断言两次计算结果的近似相等性
    assert_array_almost_equal(values, values2)

    # binary-class case
    X = [[2, 1], [3, 1], [1, 3], [2, 3]]
    y = [0, 0, 1, 1]

    # 使用 LinearSVC 模型对二元数据进行训练
    clf = svm.LinearSVC().fit(X, y)
    # 获取决策函数的值
    values = clf.decision_function(X)
    # 复制 coef_ 和 intercept_ 属性以确保修改不会影响原始对象
    clf.coef_ = clf.coef_.copy()
    clf.intercept_ = clf.intercept_.copy()
    # 再次计算决策函数的值
    values2 = clf.decision_function(X)
    # 断言两次计算结果的完全相等性
    assert_array_equal(values, values2)


def test_immutable_coef_property():
    # 检查原始 coef_ 属性不能被修改，否则会引发 AttributeError
    svms = [
        svm.SVC(kernel="linear").fit(iris.data, iris.target),
        svm.NuSVC(kernel="linear").fit(iris.data, iris.target),
        svm.SVR(kernel="linear").fit(iris.data, iris.target),
        svm.NuSVR(kernel="linear").fit(iris.data, iris.target),
        svm.OneClassSVM(kernel="linear").fit(iris.data),
    ]
    for clf in svms:
        with pytest.raises(AttributeError):
            # 尝试设置 coef_ 属性会引发 AttributeError
            clf.__setattr__("coef_", np.arange(3))
        with pytest.raises((RuntimeError, ValueError)):
            # 尝试通过 __setitem__ 修改 coef_ 的值会引发 RuntimeError 或 ValueError
            clf.coef_.__setitem__((0, 0), 0)


def test_linearsvc_verbose():
    # 将标准输出重定向到临时管道，用于捕获输出
    import os

    stdout = os.dup(1)  # 保存原始的标准输出
    os.dup2(os.pipe()[1], 1)  # 替换标准输出为新的管道

    # 实际调用 LinearSVC 模型，并设置 verbose=1
    clf = svm.LinearSVC(verbose=1)
    clf.fit(X, Y)

    # 恢复原始的标准输出
    os.dup2(stdout, 1)


def test_svc_clone_with_callable_kernel():
    # 创建一个使用可调用线性核的 SVM 模型，并检查结果与内置线性核相同
    svm_callable = svm.SVC(
        kernel=lambda x, y: np.dot(x, y.T),
        probability=True,
        random_state=0,
        decision_function_shape="ovr",
    )
    # 克隆模型以检查 lambda 函数的克隆性
    svm_cloned = base.clone(svm_callable)
    svm_cloned.fit(iris.data, iris.target)

    # 创建一个使用内置线性核的 SVM 模型
    svm_builtin = svm.SVC(
        kernel="linear", probability=True, random_state=0, decision_function_shape="ovr"
    )
    svm_builtin.fit(iris.data, iris.target)

    # 断言克隆的模型的 dual_coef_ 和 intercept_ 与内置模型的相等
    assert_array_almost_equal(svm_cloned.dual_coef_, svm_builtin.dual_coef_)
    assert_array_almost_equal(svm_cloned.intercept_, svm_builtin.intercept_)
    # 断言预测结果相等
    assert_array_equal(svm_cloned.predict(iris.data), svm_builtin.predict(iris.data))

    # 断言预测概率相等
    assert_array_almost_equal(
        svm_cloned.predict_proba(iris.data),
        svm_builtin.predict_proba(iris.data),
        decimal=4,
    )
    # 断言决策函数结果相等
    assert_array_almost_equal(
        svm_cloned.decision_function(iris.data),
        svm_builtin.decision_function(iris.data),
    )


def test_svc_bad_kernel():
    # 创建一个使用错误的核函数的 SVM 模型，并验证是否会引发 ValueError
    svc = svm.SVC(kernel=lambda x, y: x)
    with pytest.raises(ValueError):
        svc.fit(X, Y)


def test_libsvm_convergence_warnings():
    # 这个测试函数未完成，没有提供任何代码。
    pass
    # 创建一个支持向量机分类器对象（SVC），使用自定义的线性核函数 np.dot(x, y.T)
    # 设置模型能够输出类别概率
    # 设置随机种子为 0
    # 设置最大迭代次数为 2
    a = svm.SVC(
        kernel=lambda x, y: np.dot(x, y.T), probability=True, random_state=0, max_iter=2
    )
    
    # 设置警告信息，提示模型提前终止（max_iter=2）。建议在数据预处理时考虑使用 StandardScaler 或 MinMaxScaler
    warning_msg = (
        r"Solver terminated early \(max_iter=2\).  Consider pre-processing "
        r"your data with StandardScaler or MinMaxScaler."
    )
    
    # 使用 pytest 的 warn 工具，检查是否产生 ConvergenceWarning 警告，并且匹配特定的警告信息
    with pytest.warns(ConvergenceWarning, match=warning_msg):
        # 使用训练数据 X 和标签 Y 来训练 SVC 模型 a
        a.fit(np.array(X), Y)
    
    # 断言检查模型的迭代次数是否都为 2
    assert np.all(a.n_iter_ == 2)
def test_unfitted():
    X = "foo!"  # 定义输入字符串X，此处不需要验证输入，因为SVM尚未拟合

    clf = svm.SVC()  # 创建一个SVC分类器对象clf
    # 使用pytest断言检查是否引发了异常，确保SVC未拟合时调用predict会引发异常
    with pytest.raises(Exception, match=r".*\bSVC\b.*\bnot\b.*\bfitted\b"):
        clf.predict(X)

    clf = svm.NuSVR()  # 创建一个NuSVR回归器对象clf
    # 使用pytest断言检查是否引发了异常，确保NuSVR未拟合时调用predict会引发异常
    with pytest.raises(Exception, match=r".*\bNuSVR\b.*\bnot\b.*\bfitted\b"):
        clf.predict(X)


@ignore_warnings  # 使用装饰器ignore_warnings来忽略收敛警告
def test_consistent_proba():
    a = svm.SVC(probability=True, max_iter=1, random_state=0)  # 创建一个SVC分类器对象a，设置概率为True和最大迭代次数为1
    proba_1 = a.fit(X, Y).predict_proba(X)  # 对a进行拟合并预测概率值proba_1
    a = svm.SVC(probability=True, max_iter=1, random_state=0)  # 重新创建一个SVC分类器对象a
    proba_2 = a.fit(X, Y).predict_proba(X)  # 对a进行拟合并预测概率值proba_2
    # 使用断言确保两次预测的概率值几乎相等
    assert_array_almost_equal(proba_1, proba_2)


def test_linear_svm_convergence_warnings():
    # 测试当模型不收敛时是否会引发警告

    lsvc = svm.LinearSVC(random_state=0, max_iter=2)  # 创建一个LinearSVC分类器对象lsvc，设置随机种子和最大迭代次数
    warning_msg = "Liblinear failed to converge, increase the number of iterations."
    # 使用pytest断言检查是否引发了收敛警告，匹配警告消息warning_msg
    with pytest.warns(ConvergenceWarning, match=warning_msg):
        lsvc.fit(X, Y)
    # 检查n_iter_属性是否为整数类型，以匹配文档说明
    assert isinstance(lsvc.n_iter_, int)
    assert lsvc.n_iter_ == 2

    lsvr = svm.LinearSVR(random_state=0, max_iter=2)  # 创建一个LinearSVR回归器对象lsvr，设置随机种子和最大迭代次数
    # 使用pytest断言检查是否引发了收敛警告，匹配警告消息warning_msg
    with pytest.warns(ConvergenceWarning, match=warning_msg):
        lsvr.fit(iris.data, iris.target)
    assert isinstance(lsvr.n_iter_, int)
    assert lsvr.n_iter_ == 2


def test_svr_coef_sign():
    # 测试SVR(kernel="linear")的coef_是否具有正确的符号
    # 用于非回归#2933的非退化测试

    X = np.random.RandomState(21).randn(10, 3)  # 创建一个随机数据矩阵X
    y = np.random.RandomState(12).randn(10)  # 创建一个随机目标向量y

    for svr in [
        svm.SVR(kernel="linear"),  # 创建一个SVR核为线性的回归器对象svr
        svm.NuSVR(kernel="linear"),  # 创建一个NuSVR核为线性的回归器对象svr
        svm.LinearSVR(),  # 创建一个LinearSVR回归器对象svr
    ]:
        svr.fit(X, y)  # 对svr进行拟合
        # 使用断言确保预测值几乎等于np.dot(X, svr.coef_.ravel()) + svr.intercept_
        assert_array_almost_equal(
            svr.predict(X), np.dot(X, svr.coef_.ravel()) + svr.intercept_
        )


def test_lsvc_intercept_scaling_zero():
    # 测试当fit_intercept为False时是否忽略intercept_scaling

    lsvc = svm.LinearSVC(fit_intercept=False)  # 创建一个fit_intercept为False的LinearSVC分类器对象lsvc
    lsvc.fit(X, Y)  # 对lsvc进行拟合
    assert lsvc.intercept_ == 0.0  # 使用断言确保intercept_为0.0


def test_hasattr_predict_proba():
    # 在拟合前或拟合后，通过probability参数切换来测试方法的可用性

    G = svm.SVC(probability=True)  # 创建一个probability为True的SVC分类器对象G
    assert hasattr(G, "predict_proba")  # 使用断言确保G具有"predict_proba"方法
    G.fit(iris.data, iris.target)  # 对G进行拟合
    assert hasattr(G, "predict_proba")  # 使用断言确保G具有"predict_proba"方法

    G = svm.SVC(probability=False)  # 创建一个probability为False的SVC分类器对象G
    assert not hasattr(G, "predict_proba")  # 使用断言确保G没有"predict_proba"方法
    G.fit(iris.data, iris.target)  # 对G进行拟合
    assert not hasattr(G, "predict_proba")  # 使用断言确保G没有"predict_proba"方法

    # 在拟合后切换为probability=True应该使predict_proba可用，但调用时不应工作：
    G.probability = True  # 将probability设置为True
    assert hasattr(G, "predict_proba")  # 使用断言确保G具有"predict_proba"方法
    msg = "predict_proba is not available when fitted with probability=False"

    with pytest.raises(NotFittedError, match=msg):
        G.predict_proba(iris.data)  # 使用pytest断言确保在probability=False时调用predict_proba会引发NotFittedError异常


def test_decision_function_shape_two_class():
    # 在双类问题中测试decision_function的形状
    # 对于每个类别数量，分别为2和3
    for n_classes in [2, 3]:
        # 使用make_blobs生成具有指定中心点数量的数据集，随机种子为0
        X, y = make_blobs(centers=n_classes, random_state=0)
        # 对于每个分类器，包括svm.SVC和svm.NuSVC
        for estimator in [svm.SVC, svm.NuSVC]:
            # 使用OneVsRestClassifier进行多分类处理，使用ovr模式
            clf = OneVsRestClassifier(estimator(decision_function_shape="ovr")).fit(
                X, y
            )
            # 断言预测结果的长度与实际标签y的长度相等
            assert len(clf.predict(X)) == len(y)
def test_ovr_decision_function():
    # One point from each quadrant represents one class
    # 定义训练数据 X_train，每个象限一个点代表一个类别
    X_train = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    # 定义训练数据标签 y_train
    y_train = [0, 1, 2, 3]

    # First point is closer to the decision boundaries than the second point
    # 定义基础测试点，第一个点比第二个点更靠近决策边界
    base_points = np.array([[5, 5], [10, 10]])

    # For all the quadrants (classes)
    # 构建测试数据 X_test，涵盖所有象限（类别）
    X_test = np.vstack(
        (
            base_points * [1, 1],     # Q1
            base_points * [-1, 1],    # Q2
            base_points * [-1, -1],   # Q3
            base_points * [1, -1],    # Q4
        )
    )

    # 构建测试数据标签 y_test
    y_test = [0] * 2 + [1] * 2 + [2] * 2 + [3] * 2

    # 使用线性核和一对余（OVR）的支持向量分类器
    clf = svm.SVC(kernel="linear", decision_function_shape="ovr")
    # 在训练数据上拟合分类器
    clf.fit(X_train, y_train)

    # 使用测试数据进行预测
    y_pred = clf.predict(X_test)

    # 测试预测结果是否与真实标签 y_test 相同
    assert_array_equal(y_pred, y_test)

    # 获取测试点的决策函数值
    deci_val = clf.decision_function(X_test)

    # 断言预测类别的决策函数值中最大值对应的索引与预测类别相同
    assert_array_equal(np.argmax(deci_val, axis=1), y_pred)

    # 获取预测类别的决策函数值
    pred_class_deci_val = deci_val[range(8), y_pred].reshape((4, 2))

    # 断言 pred_class_deci_val 中所有值大于 0
    assert np.min(pred_class_deci_val) > 0.0

    # 测试第一个点在每个象限的决策函数值是否比第二个点低
    assert np.all(pred_class_deci_val[:, 0] < pred_class_deci_val[:, 1])


@pytest.mark.parametrize("SVCClass", [svm.SVC, svm.NuSVC])
def test_svc_invalid_break_ties_param(SVCClass):
    X, y = make_blobs(random_state=42)

    # 使用给定的 SVCClass 初始化支持向量分类器对象
    svm = SVCClass(
        kernel="linear", decision_function_shape="ovo", break_ties=True, random_state=42
    ).fit(X, y)

    # 断言在 break_ties 参数为 True 时会引发 ValueError 异常
    with pytest.raises(ValueError, match="break_ties must be False"):
        svm.predict(y)


@pytest.mark.parametrize("SVCClass", [svm.SVC, svm.NuSVC])
def test_svc_ovr_tie_breaking(SVCClass):
    """Test if predict breaks ties in OVR mode.
    Related issue: https://github.com/scikit-learn/scikit-learn/issues/8277
    """
    X, y = make_blobs(random_state=0, n_samples=20, n_features=2)

    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    ys = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx, yy = np.meshgrid(xs, ys)

    common_params = dict(
        kernel="rbf", gamma=1e6, random_state=42, decision_function_shape="ovr"
    )
    # 初始化支持向量分类器对象，测试在 OVR 模式下是否会处理并打破平局
    svm = SVCClass(
        break_ties=False,
        **common_params,
    ).fit(X, y)
    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    dv = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    assert not np.all(pred == np.argmax(dv, axis=1))

    # 初始化支持向量分类器对象，测试在 OVR 模式下是否会处理并打破平局
    svm = SVCClass(
        break_ties=True,
        **common_params,
    ).fit(X, y)
    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    dv = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    assert np.all(pred == np.argmax(dv, axis=1))


def test_gamma_scale():
    X, y = [[0.0], [1.0]], [0, 1]

    # 初始化支持向量分类器对象
    clf = svm.SVC()
    clf.fit(X, y)
    # 断言计算的 gamma 值约等于 4
    assert_almost_equal(clf._gamma, 4)


@pytest.mark.parametrize(
    "SVM, params",
    [  # 创建包含 SVM 参数的列表
        (LinearSVC, {"penalty": "l1", "loss": "squared_hinge", "dual": False}),  # 元组1: LinearSVC分类器，参数为L1惩罚、平方hinge损失，不使用对偶形式
        (LinearSVC, {"penalty": "l2", "loss": "squared_hinge", "dual": True}),   # 元组2: LinearSVC分类器，参数为L2惩罚、平方hinge损失，使用对偶形式
        (LinearSVC, {"penalty": "l2", "loss": "squared_hinge", "dual": False}),  # 元组3: LinearSVC分类器，参数为L2惩罚、平方hinge损失，不使用对偶形式
        (LinearSVC, {"penalty": "l2", "loss": "hinge", "dual": True}),           # 元组4: LinearSVC分类器，参数为L2惩罚、hinge损失，使用对偶形式
        (LinearSVR, {"loss": "epsilon_insensitive", "dual": True}),               # 元组5: LinearSVR回归器，损失函数为epsilon-insensitive，使用对偶形式
        (LinearSVR, {"loss": "squared_epsilon_insensitive", "dual": True}),       # 元组6: LinearSVR回归器，损失函数为squared epsilon-insensitive，使用对偶形式
        (LinearSVR, {"loss": "squared_epsilon_insensitive", "dual": True}),       # 元组7: LinearSVR回归器，损失函数为squared epsilon-insensitive，使用对偶形式
    ],
)
# 定义测试函数 test_linearsvm_liblinear_sample_weight，接受 SVM 模型和参数作为输入
def test_linearsvm_liblinear_sample_weight(SVM, params):
    # 定义输入特征矩阵 X，包含16个样本，每个样本2个特征
    X = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.dtype("float"),
    )
    # 定义标签向量 y，包含16个样本的分类标签
    y = np.array(
        [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.dtype("int")
    )

    # 创建扩展后的特征矩阵 X2 和对应的标签向量 y2
    X2 = np.vstack([X, X])
    y2 = np.hstack([y, 3 - y])
    
    # 创建样本权重向量 sample_weight，初始化为长度为32的全1向量
    sample_weight = np.ones(shape=len(y) * 2)
    # 将后16个样本的权重设为0
    sample_weight[len(y):] = 0
    
    # 对 X2, y2, sample_weight 进行洗牌（随机打乱顺序）
    X2, y2, sample_weight = shuffle(X2, y2, sample_weight, random_state=0)

    # 创建基础估计器对象 base_estimator，使用给定的 SVM 模型和参数初始化
    base_estimator = SVM(random_state=42)
    base_estimator.set_params(**params)
    base_estimator.set_params(tol=1e-12, max_iter=1000)
    
    # 使用 base_estimator 克隆对象，并使用 X, y 进行拟合，得到 est_no_weight 模型
    est_no_weight = base.clone(base_estimator).fit(X, y)
    
    # 使用 base_estimator 克隆对象，并使用 X2, y2 和 sample_weight 进行拟合，得到 est_with_weight 模型
    est_with_weight = base.clone(base_estimator).fit(
        X2, y2, sample_weight=sample_weight
    )

    # 遍历预测方法 predict 和 decision_function
    for method in ("predict", "decision_function"):
        # 如果 base_estimator 对象具有 method 方法
        if hasattr(base_estimator, method):
            # 分别对 est_no_weight 和 est_with_weight 模型使用 method 方法进行预测
            X_est_no_weight = getattr(est_no_weight, method)(X)
            X_est_with_weight = getattr(est_with_weight, method)(X)
            # 断言两个预测结果在精度上相近
            assert_allclose(X_est_no_weight, X_est_with_weight)


@pytest.mark.parametrize("Klass", (OneClassSVM, SVR, NuSVR))
# 定义测试函数 test_n_support，参数 Klass 是 OneClassSVM, SVR, NuSVR 中的一种
def test_n_support(Klass):
    # 创建输入特征矩阵 X，包含5个样本，每个样本1个特征
    X = np.array([[0], [0.44], [0.45], [0.46], [1]])
    # 创建对应的标签向量 y
    y = np.arange(X.shape[0])
    # 创建 Klass 模型对象 est
    est = Klass()
    # 断言 est 模型对象没有属性 "n_support_"
    assert not hasattr(est, "n_support_")
    # 使用 X, y 对 est 模型进行拟合
    est.fit(X, y)
    # 断言 est.n_support_ 的第一个元素等于 est.support_vectors_.shape[0]
    assert est.n_support_[0] == est.support_vectors_.shape[0]
    # 断言 est.n_support_ 的大小为1
    assert est.n_support_.size == 1


@pytest.mark.parametrize("Estimator", [svm.SVC, svm.SVR])
# 定义测试函数 test_custom_kernel_not_array_input，参数 Estimator 是 svm.SVC 或 svm.SVR
def test_custom_kernel_not_array_input(Estimator):
    """Test using a custom kernel that is not fed with array-like for floats"""
    # 创建数据列表 data 和对应的特征矩阵 X
    data = ["A A", "A", "B", "B B", "A B"]
    X = np.array([[2, 0], [1, 0], [0, 1], [0, 2], [1, 1]])  # count encoding
    # 创建标签向量 y
    y = np.array([1, 1, 2, 2, 1])

    # 定义自定义核函数 string_kernel，接受 X1, X2 作为输入
    def string_kernel(X1, X2):
        assert isinstance(X1[0], str)
        n_samples1 = _num_samples(X1)
        n_samples2 = _num_samples(X2)
        # 初始化核矩阵 K
        K = np.zeros((n_samples1, n_samples2))
        # 计算核矩阵 K 的每个元素
        for ii in range(n_samples1):
            for jj in range(ii, n_samples2):
                K[ii, jj] = X1[ii].count("A") * X2[jj].count("A")
                K[ii, jj] += X1[ii].count("B") * X2[jj].count("B")
                K[jj, ii] = K[ii, jj]
        return K

    # 计算数据 data 的核矩阵 K，并与预期的线性核矩阵 X * X.T 进行比较
    K = string_kernel(data, data)
    assert_array_equal(np.dot(X, X.T), K)

    # 使用 string_kernel 作为核函数创建 Estimator 模型对象 svc1，并使用 data, y 进行拟合
    svc1 = Estimator(kernel=string_kernel).fit(data, y)
    # 创建 Estimator 模型对象 svc2，使用线性核函数 "linear" 和 X, y 进行拟合
    svc2 = Estimator(kernel="linear").fit(X, y)
    # 创建 Estimator 模型对象 svc3，使用预先计算的核矩阵 "precomputed" 和 K, y 进行拟合
    svc3 = Estimator(kernel="precomputed").fit(K, y)

    # 断言 svc1 和 svc3 的预测精度相等
    assert svc1.score(data, y) == svc3.score(K, y)
    # 断言两个支持向量机分类器的得分相等
    assert svc1.score(data, y) == svc2.score(X, y)
    
    # 如果支持向量机分类器具有 decision_function 方法（即为分类器），执行以下断言
    if hasattr(svc1, "decision_function"):
        # 断言两个支持向量机分类器的 decision_function 结果近似相等
        assert_allclose(svc1.decision_function(data), svc2.decision_function(X))
        # 断言第一个支持向量机分类器的 decision_function 结果与第三个分类器的结果近似相等
        assert_allclose(svc1.decision_function(data), svc3.decision_function(K))
        # 断言两个支持向量机分类器的预测结果相等
        assert_array_equal(svc1.predict(data), svc2.predict(X))
        # 断言第一个支持向量机分类器的预测结果与第三个分类器的预测结果相等
        assert_array_equal(svc1.predict(data), svc3.predict(K))
    else:
        # 如果支持向量机分类器是回归器，执行以下断言
        # 断言两个支持向量机回归器的预测结果近似相等
        assert_allclose(svc1.predict(data), svc2.predict(X))
        # 断言第一个支持向量机回归器的预测结果与第三个回归器的预测结果近似相等
        assert_allclose(svc1.predict(data), svc3.predict(K))
# 检查当内部表示被修改时，SVC 是否会引发错误。
# 这是针对问题 #18891 和 CVE-2020-28975 的非回归测试。
def test_svc_raises_error_internal_representation():
    clf = svm.SVC(kernel="linear").fit(X, Y)
    # 修改 SVC 的内部表示
    clf._n_support[0] = 1000000

    msg = "The internal representation of SVC was altered"
    # 使用 pytest 检查是否引发 ValueError，并匹配指定的错误消息
    with pytest.raises(ValueError, match=msg):
        clf.predict(X)


@pytest.mark.parametrize(
    "estimator, expected_n_iter_type",
    [
        (svm.SVC, np.ndarray),
        (svm.NuSVC, np.ndarray),
        (svm.SVR, int),
        (svm.NuSVR, int),
        (svm.OneClassSVM, int),
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        make_classification(n_classes=2, n_informative=2, random_state=0),
        make_classification(n_classes=3, n_informative=3, random_state=0),
        make_classification(n_classes=4, n_informative=4, random_state=0),
    ],
)
def test_n_iter_libsvm(estimator, expected_n_iter_type, dataset):
    # 检查继承自 BaseSVC 的类的 n_iter_ 类型是否正确。
    # 对于 SVC 和 NuSVC，n_iter_ 应该是 ndarray；对于 SVR、NuSVR 和 OneClassSVM，应该是 int。
    X, y = dataset
    # 获取拟合后的模型的 n_iter_
    n_iter = estimator(kernel="linear").fit(X, y).n_iter_
    assert type(n_iter) == expected_n_iter_type
    if estimator in [svm.SVC, svm.NuSVC]:
        n_classes = len(np.unique(y))
        assert n_iter.shape == (n_classes * (n_classes - 1) // 2,)


@pytest.mark.parametrize("loss", ["squared_hinge", "squared_epsilon_insensitive"])
def test_dual_auto(loss):
    # 测试自动确定双重优化参数时的不同情况。
    # 这里展示了几个具体的情况，如 OvR、L2、N > M 和 N < M 的情况。
    # 使用 _validate_dual_parameter 函数来检查双重优化参数是否符合预期。
    dual = _validate_dual_parameter("auto", loss, "l2", "ovr", np.asarray(X))
    assert dual is False
    dual = _validate_dual_parameter("auto", loss, "l2", "ovr", np.asarray(X).T)
    assert dual is True


def test_dual_auto_edge_cases():
    # 测试双重优化参数在边缘情况下的行为。
    # 这里展示了 Hinge、OvR、L2、N > M 和 SqHinge、OvR、L1、N < M 的情况。
    # 使用 _validate_dual_parameter 函数来检查双重优化参数是否符合预期。
    dual = _validate_dual_parameter("auto", "hinge", "l2", "ovr", np.asarray(X))
    assert dual is True  # 只支持 True
    dual = _validate_dual_parameter("auto", "epsilon_insensitive", "l2", "ovr", np.asarray(X))
    assert dual is True  # 只支持 True
    dual = _validate_dual_parameter("auto", "squared_hinge", "l1", "ovr", np.asarray(X).T)
    assert dual is False  # 只支持 False
```