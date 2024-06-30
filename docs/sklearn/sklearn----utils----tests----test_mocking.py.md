# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_mocking.py`

```
# 导入所需的库和模块
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

# 导入 sklearn 中的相关模块和函数
from sklearn.datasets import load_iris
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._mocking import (
    CheckingClassifier,
    _MockEstimatorOnOffPrediction,
)
from sklearn.utils._testing import _convert_container
from sklearn.utils.fixes import CSR_CONTAINERS

# 定义 pytest 的 fixture，加载鸢尾花数据集
@pytest.fixture
def iris():
    return load_iris(return_X_y=True)

# 成功的辅助函数
def _success(x):
    return True

# 失败的辅助函数
def _fail(x):
    return False

# 参数化测试，检查训练成功的情况
@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # 默认情况
        {"check_X": _success},  # 成功检查输入特征
        {"check_y": _success},  # 成功检查目标变量
        {"check_X": _success, "check_y": _success},  # 成功检查输入特征和目标变量
    ],
)
def test_check_on_fit_success(iris, kwargs):
    X, y = iris
    CheckingClassifier(**kwargs).fit(X, y)

# 参数化测试，检查训练失败的情况
@pytest.mark.parametrize(
    "kwargs",
    [
        {"check_X": _fail},  # 失败检查输入特征
        {"check_y": _fail},  # 失败检查目标变量
        {"check_X": _success, "check_y": _fail},  # 输入特征成功，目标变量失败
        {"check_X": _fail, "check_y": _success},  # 输入特征失败，目标变量成功
        {"check_X": _fail, "check_y": _fail},  # 输入特征和目标变量均失败
    ],
)
def test_check_on_fit_fail(iris, kwargs):
    X, y = iris
    clf = CheckingClassifier(**kwargs)
    with pytest.raises(AssertionError):
        clf.fit(X, y)

# 参数化测试，检查预测成功的情况
@pytest.mark.parametrize(
    "pred_func", ["predict", "predict_proba", "decision_function", "score"]
)
def test_check_X_on_predict_success(iris, pred_func):
    X, y = iris
    clf = CheckingClassifier(check_X=_success).fit(X, y)
    getattr(clf, pred_func)(X)

# 参数化测试，检查预测失败的情况
@pytest.mark.parametrize(
    "pred_func", ["predict", "predict_proba", "decision_function", "score"]
)
def test_check_X_on_predict_fail(iris, pred_func):
    X, y = iris
    clf = CheckingClassifier(check_X=_success).fit(X, y)
    clf.set_params(check_X=_fail)
    with pytest.raises(AssertionError):
        getattr(clf, pred_func)(X)

# 参数化测试，检查不同输入类型下的分类器检查
@pytest.mark.parametrize("input_type", ["list", "array", "sparse", "dataframe"])
def test_checking_classifier(iris, input_type):
    # 检查 CheckingClassifier 输出是否符合预期
    X, y = iris
    X = _convert_container(X, input_type)
    clf = CheckingClassifier()
    clf.fit(X, y)

    # 断言分类器的类别是否正确
    assert_array_equal(clf.classes_, np.unique(y))
    assert len(clf.classes_) == 3
    assert clf.n_features_in_ == 4

    # 预测和评分的断言
    y_pred = clf.predict(X)
    assert all(pred in clf.classes_ for pred in y_pred)

    assert clf.score(X) == pytest.approx(0)
    clf.set_params(foo_param=10)
    assert clf.fit(X, y).score(X) == pytest.approx(1)

    # 预测概率的断言
    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (150, 3)
    assert np.logical_and(y_proba >= 0, y_proba <= 1).all()

    # 决策函数的断言
    y_decision = clf.decision_function(X)
    assert y_decision.shape == (150, 3)

    # 对于二分类的形状检查
    first_2_classes = np.logical_or(y == 0, y == 1)
    X = _safe_indexing(X, first_2_classes)
    y = _safe_indexing(y, first_2_classes)
    clf.fit(X, y)

    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (100, 2)
    # 断言：确保所有的预测概率 y_proba 都在 [0, 1] 的范围内
    assert np.logical_and(y_proba >= 0, y_proba <= 1).all()
    
    # 获取分类器 clf 对输入数据集 X 的决策函数值
    y_decision = clf.decision_function(X)
    # 断言：确保决策函数的输出形状为 (100,)，即包含 100 个元素的一维数组
    assert y_decision.shape == (100,)
# 使用 pytest 的参数化装饰器，对 test_checking_classifier_with_params 函数进行多次参数化测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_checking_classifier_with_params(iris, csr_container):
    # 从 iris 数据中获取特征 X 和标签 y
    X, y = iris
    # 使用给定的 csr_container 将 X 转换为稀疏矩阵 X_sparse
    X_sparse = csr_container(X)

    # 初始化一个 CheckingClassifier 对象，指定 check_X 参数为 sparse.issparse 函数
    clf = CheckingClassifier(check_X=sparse.issparse)
    # 断言调用 fit 方法时会引发 AssertionError 异常
    with pytest.raises(AssertionError):
        clf.fit(X, y)
    # 使用稀疏矩阵 X_sparse 调用 fit 方法
    clf.fit(X_sparse, y)

    # 初始化另一个 CheckingClassifier 对象，指定 check_X 参数为 check_array 函数，
    # 并且指定 check_X_params 参数为 {"accept_sparse": False}
    clf = CheckingClassifier(
        check_X=check_array, check_X_params={"accept_sparse": False}
    )
    # 使用原始的特征 X 调用 fit 方法
    clf.fit(X, y)
    # 断言调用 fit 方法时会引发 TypeError 异常，并且异常信息中包含 "Sparse data was passed"
    with pytest.raises(TypeError, match="Sparse data was passed"):
        clf.fit(X_sparse, y)


# 对 test_checking_classifier_fit_params 函数进行单元测试
def test_checking_classifier_fit_params(iris):
    # 从 iris 数据中获取特征 X 和标签 y
    X, y = iris
    # 初始化一个 CheckingClassifier 对象，指定 expected_sample_weight 参数为 True
    clf = CheckingClassifier(expected_sample_weight=True)
    # 创建一个长度为 len(X) // 2 的样本权重数组
    sample_weight = np.ones(len(X) // 2)

    # 准备一个期望的错误信息
    msg = f"sample_weight.shape == ({len(X) // 2},), expected ({len(X)},)!"
    # 断言调用 fit 方法时会引发 ValueError 异常，并且异常信息与预期的 msg 相同
    with pytest.raises(ValueError) as exc:
        clf.fit(X, y, sample_weight=sample_weight)
    assert exc.value.args[0] == msg


# 对 test_checking_classifier_missing_fit_params 函数进行单元测试
def test_checking_classifier_missing_fit_params(iris):
    # 从 iris 数据中获取特征 X 和标签 y
    X, y = iris
    # 初始化一个 CheckingClassifier 对象，指定 expected_sample_weight 参数为 True
    clf = CheckingClassifier(expected_sample_weight=True)
    # 准备一个错误信息字符串
    err_msg = "Expected sample_weight to be passed"
    # 断言调用 fit 方法时会引发 AssertionError 异常，并且异常信息中包含 err_msg
    with pytest.raises(AssertionError, match=err_msg):
        clf.fit(X, y)


# 使用 pytest 的参数化装饰器，对 test_checking_classifier_methods_to_check 函数进行多次参数化测试
@pytest.mark.parametrize(
    "methods_to_check",
    [["predict"], ["predict", "predict_proba"]],
)
@pytest.mark.parametrize(
    "predict_method", ["predict", "predict_proba", "decision_function", "score"]
)
def test_checking_classifier_methods_to_check(iris, methods_to_check, predict_method):
    # 从 iris 数据中获取特征 X 和标签 y
    X, y = iris

    # 初始化一个 CheckingClassifier 对象，指定 check_X 参数为 sparse.issparse，
    # 并且指定 methods_to_check 参数为 methods_to_check 中指定的方法列表
    clf = CheckingClassifier(
        check_X=sparse.issparse,
        methods_to_check=methods_to_check,
    )

    # 调用 fit 方法
    clf.fit(X, y)
    # 如果 predict_method 存在于 methods_to_check 中，则断言调用 getattr(clf, predict_method)(X) 时会引发 AssertionError 异常
    if predict_method in methods_to_check:
        with pytest.raises(AssertionError):
            getattr(clf, predict_method)(X)
    else:
        # 否则，调用 getattr(clf, predict_method)(X)
        getattr(clf, predict_method)(X)


# 使用 pytest 的参数化装饰器，对 test_mock_estimator_on_off_prediction 函数进行多次参数化测试
@pytest.mark.parametrize(
    "response_methods",
    [
        ["predict"],
        ["predict", "predict_proba"],
        ["predict", "decision_function"],
        ["predict", "predict_proba", "decision_function"],
    ],
)
def test_mock_estimator_on_off_prediction(iris, response_methods):
    # 从 iris 数据中获取特征 X 和标签 y
    X, y = iris
    # 初始化一个 _MockEstimatorOnOffPrediction 对象，指定 response_methods 参数为 response_methods 中指定的方法列表
    estimator = _MockEstimatorOnOffPrediction(response_methods=response_methods)

    # 调用 fit 方法
    estimator.fit(X, y)
    # 断言 estimator 对象具有 classes_ 属性，并且其值与 y 中唯一值的数组相同
    assert hasattr(estimator, "classes_")
    assert_array_equal(estimator.classes_, np.unique(y))

    # 准备一个可能的响应方法列表
    possible_responses = ["predict", "predict_proba", "decision_function"]
    # 遍历可能的响应方法
    for response in possible_responses:
        # 如果 response 存在于 response_methods 中，则断言 estimator 对象具有 response 方法，
        # 并且调用 getattr(estimator, response)(X) 返回的结果等于 response
        if response in response_methods:
            assert hasattr(estimator, response)
            assert getattr(estimator, response)(X) == response
        else:
            # 否则，断言 estimator 对象不具有 response 方法
            assert not hasattr(estimator, response)
```