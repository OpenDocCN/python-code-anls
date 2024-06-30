# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_response.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from sklearn.datasets import (  # 导入 Scikit-Learn 库中的数据集模块
    load_iris,  # 导入鸢尾花数据集加载函数
    make_classification,  # 导入创建分类数据集的函数
    make_multilabel_classification,  # 导入创建多标签分类数据集的函数
    make_regression,  # 导入创建回归数据集的函数
)
from sklearn.ensemble import IsolationForest  # 导入隔离森林异常检测算法
from sklearn.linear_model import (  # 导入线性模型
    LinearRegression,  # 导入线性回归模型
    LogisticRegression,  # 导入逻辑回归模型
)
from sklearn.multioutput import ClassifierChain  # 导入多输出分类链模型
from sklearn.preprocessing import scale  # 导入数据预处理函数 scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入决策树模型
from sklearn.utils._mocking import _MockEstimatorOnOffPrediction  # 导入用于模拟估计器的模块
from sklearn.utils._response import _get_response_values, _get_response_values_binary  # 导入用于获取响应值的模块
from sklearn.utils._testing import assert_allclose, assert_array_equal  # 导入用于测试的模块

X, y = load_iris(return_X_y=True)  # 加载鸢尾花数据集，并返回特征 X 和标签 y
# 将数据进行标准化处理，避免 LogisticRegression 的收敛警告
X = scale(X, copy=False)
X_binary, y_binary = X[:100], y[:100]


@pytest.mark.parametrize(  # 使用 Pytest 的参数化装饰器，定义测试参数
    "response_method", ["decision_function", "predict_proba", "predict_log_proba"]
)
def test_get_response_values_regressor_error(response_method):
    """检查回归器在不支持的响应方法时的错误消息。"""
    my_estimator = _MockEstimatorOnOffPrediction(response_methods=[response_method])
    X = "mocking_data", "mocking_target"
    err_msg = f"{my_estimator.__class__.__name__} should either be a classifier"
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(my_estimator, X, response_method=response_method)


@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_values_regressor(return_response_method_used):
    """检查 `_get_response_values` 在回归器中的行为。"""
    X, y = make_regression(n_samples=10, random_state=0)
    regressor = LinearRegression().fit(X, y)
    results = _get_response_values(
        regressor,
        X,
        response_method="predict",
        return_response_method_used=return_response_method_used,
    )
    assert_array_equal(results[0], regressor.predict(X))  # 断言预测结果与真实值相等
    assert results[1] is None
    if return_response_method_used:
        assert results[2] == "predict"  # 断言返回的响应方法为 "predict"


@pytest.mark.parametrize(
    "response_method",
    ["predict", "decision_function", ["decision_function", "predict"]],
)
@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_values_outlier_detection(
    response_method, return_response_method_used
):
    """检查 `_get_response_values` 在异常检测器中的行为。"""
    X, y = make_classification(n_samples=50, random_state=0)
    outlier_detector = IsolationForest(random_state=0).fit(X, y)
    results = _get_response_values(
        outlier_detector,
        X,
        response_method=response_method,
        return_response_method_used=return_response_method_used,
    )
    chosen_response_method = (
        response_method[0] if isinstance(response_method, list) else response_method
    )
    prediction_method = getattr(outlier_detector, chosen_response_method)
    # 断言第一个结果与预测方法应用于 X 后的结果相等
    assert_array_equal(results[0], prediction_method(X))
    
    # 断言第二个结果为 None
    assert results[1] is None
    
    # 如果使用返回响应方法，则断言第三个结果等于选定的响应方法
    if return_response_method_used:
        assert results[2] == chosen_response_method
@pytest.mark.parametrize(
    "response_method",
    ["predict_proba", "decision_function", "predict", "predict_log_proba"],
)
def test_get_response_values_classifier_unknown_pos_label(response_method):
    """Check that `_get_response_values` raises the proper error message with
    classifier."""
    # 生成一个具有两个类的分类数据集
    X, y = make_classification(n_samples=10, n_classes=2, random_state=0)
    # 使用逻辑回归模型拟合数据
    classifier = LogisticRegression().fit(X, y)

    # 提供一个不在 `y` 中的 `pos_label`
    err_msg = r"pos_label=whatever is not a valid label: It should be one of \[0 1\]"
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(
            classifier,
            X,
            response_method=response_method,
            pos_label="whatever",
        )


@pytest.mark.parametrize("response_method", ["predict_proba", "predict_log_proba"])
def test_get_response_values_classifier_inconsistent_y_pred_for_binary_proba(
    response_method,
):
    """Check that `_get_response_values` will raise an error when `y_pred` has a
    single class with `predict_proba`."""
    # 生成一个具有两个类的分类数据集
    X, y_two_class = make_classification(n_samples=10, n_classes=2, random_state=0)
    # 创建一个只有一个类别的 `y_pred`
    y_single_class = np.zeros_like(y_two_class)
    # 使用决策树模型拟合数据
    classifier = DecisionTreeClassifier().fit(X, y_single_class)

    # 预期的错误消息
    err_msg = (
        r"Got predict_proba of shape \(10, 1\), but need classifier with "
        r"two classes"
    )
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(classifier, X, response_method=response_method)


@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_values_binary_classifier_decision_function(
    return_response_method_used,
):
    """Check the behaviour of `_get_response_values` with `decision_function`
    and binary classifier."""
    # 生成一个具有两个类的分类数据集，其中第二类样本比例较大
    X, y = make_classification(
        n_samples=10,
        n_classes=2,
        weights=[0.3, 0.7],
        random_state=0,
    )
    # 使用逻辑回归模型拟合数据
    classifier = LogisticRegression().fit(X, y)
    # 设定响应方法为 `decision_function`
    response_method = "decision_function"

    # 默认情况下，使用 `pos_label=None`
    results = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=None,
        return_response_method_used=return_response_method_used,
    )
    # 断言结果与预期一致
    assert_allclose(results[0], classifier.decision_function(X))
    assert results[1] == 1
    if return_response_method_used:
        assert results[2] == "decision_function"

    # 强制使用 `pos_label=classifier.classes_[0]`
    results = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=classifier.classes_[0],
        return_response_method_used=return_response_method_used,
    )
    # 断言结果与预期一致
    assert_allclose(results[0], classifier.decision_function(X) * -1)
    assert results[1] == 0
    if return_response_method_used:
        assert results[2] == "decision_function"
# 使用 pytest 的 parametrize 装饰器，参数化测试函数，测试两种不同的 response_method
@pytest.mark.parametrize("response_method", ["predict_proba", "predict_log_proba"])
def test_get_response_values_binary_classifier_predict_proba(
    return_response_method_used, response_method
):
    """Check that `_get_response_values` with `predict_proba` and binary
    classifier."""
    
    # 生成一个二分类数据集 X 和 y
    X, y = make_classification(
        n_samples=10,
        n_classes=2,
        weights=[0.3, 0.7],
        random_state=0,
    )
    
    # 使用 LogisticRegression 训练分类器
    classifier = LogisticRegression().fit(X, y)

    # 测试默认情况下 `pos_label` 为 None 的结果
    results = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=None,
        return_response_method_used=return_response_method_used,
    )
    
    # 断言结果的第一个元素等于 classifier 的 predict_proba(X) 的第二列
    assert_allclose(results[0], getattr(classifier, response_method)(X)[:, 1])
    
    # 断言结果的第二个元素为 1
    assert results[1] == 1
    
    # 如果 return_response_method_used 为 True，则断言结果的长度为 3，并且第三个元素为 response_method
    if return_response_method_used:
        assert len(results) == 3
        assert results[2] == response_method
    else:
        # 否则断言结果的长度为 2
        assert len(results) == 2

    # 测试强制 `pos_label=classifier.classes_[0]` 的结果
    y_pred, pos_label, *_ = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=classifier.classes_[0],
        return_response_method_used=return_response_method_used,
    )
    
    # 断言 y_pred 等于 classifier 的 predict_proba(X) 的第一列
    assert_allclose(y_pred, getattr(classifier, response_method)(X)[:, 0])
    
    # 断言 pos_label 等于 classifier 的第一个类别标签
    assert pos_label == 0


# 参数化测试函数，测试不同情况下 _get_response_values_binary 函数的错误处理
@pytest.mark.parametrize(
    "estimator, X, y, err_msg, params",
    [
        (
            DecisionTreeRegressor(),
            X_binary,
            y_binary,
            "Expected 'estimator' to be a binary classifier",
            {"response_method": "auto"},
        ),
        (
            DecisionTreeClassifier(),
            X_binary,
            y_binary,
            r"pos_label=unknown is not a valid label: It should be one of \[0 1\]",
            {"response_method": "auto", "pos_label": "unknown"},
        ),
        (
            DecisionTreeClassifier(),
            X,
            y,
            "be a binary classifier. Got 3 classes instead.",
            {"response_method": "predict_proba"},
        ),
    ],
)
def test_get_response_error(estimator, X, y, err_msg, params):
    """Check that we raise the proper error messages in _get_response_values_binary."""
    
    # 使用给定的 estimator 拟合数据集 X 和 y
    estimator.fit(X, y)
    
    # 使用 pytest 的 raises 函数断言 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values_binary(estimator, X, **params)


# 参数化测试函数，测试 _get_response_values_binary 函数在使用 predict_proba 时的行为
@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_predict_proba(return_response_method_used):
    """Check the behaviour of `_get_response_values_binary` using `predict_proba`."""
    
    # 使用 DecisionTreeClassifier 拟合二分类数据集 X_binary 和 y_binary
    classifier = DecisionTreeClassifier().fit(X_binary, y_binary)
    
    # 调用 _get_response_values_binary 函数，获取预测概率的结果
    results = _get_response_values_binary(
        classifier,
        X_binary,
        response_method="predict_proba",
        return_response_method_used=return_response_method_used,
    )
    
    # 断言结果的第一个元素等于 classifier 的 predict_proba(X_binary) 的第二列
    assert_allclose(results[0], classifier.predict_proba(X_binary)[:, 1])
    
    # 断言结果的第二个元素为 1
    assert results[1] == 1
    # 如果使用了返回响应方法的标志，进行断言检查
    if return_response_method_used:
        # 断言第三个结果是 "predict_proba"
        assert results[2] == "predict_proba"

    # 调用 _get_response_values_binary 函数，获取二进制分类器的响应值
    results = _get_response_values_binary(
        classifier,
        X_binary,
        response_method="predict_proba",  # 指定响应方法为 "predict_proba"
        pos_label=0,  # 指定正例标签为 0
        return_response_method_used=return_response_method_used,  # 是否返回响应方法标志
    )

    # 断言第一个结果的值与 classifier.predict_proba(X_binary)[:, 0] 的值非常接近
    assert_allclose(results[0], classifier.predict_proba(X_binary)[:, 0])
    
    # 断言第二个结果是 0
    assert results[1] == 0
    
    # 如果使用了返回响应方法的标志，再次进行断言检查
    if return_response_method_used:
        # 再次断言第三个结果是 "predict_proba"
        assert results[2] == "predict_proba"
# 使用 pytest 的 parametrize 装饰器定义一个测试函数，测试 `_get_response_values_binary` 函数的行为
@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_decision_function(return_response_method_used):
    """Check the behaviour of `_get_response_values_binary` using decision_function."""

    # 创建一个逻辑回归分类器，并用 X_binary 和 y_binary 进行训练
    classifier = LogisticRegression().fit(X_binary, y_binary)

    # 调用 `_get_response_values_binary` 函数，使用 decision_function 作为响应方法
    results = _get_response_values_binary(
        classifier,
        X_binary,
        response_method="decision_function",
        return_response_method_used=return_response_method_used,
    )

    # 断言第一个返回结果与分类器的 decision_function 方法计算的结果相近
    assert_allclose(results[0], classifier.decision_function(X_binary))

    # 断言第二个返回结果为 1，用于标记正类的位置
    assert results[1] == 1

    # 如果 return_response_method_used 为 True，则断言第三个返回结果为 "decision_function"
    if return_response_method_used:
        assert results[2] == "decision_function"

    # 使用不同的参数再次调用 `_get_response_values_binary` 函数，验证对 pos_label 的处理
    results = _get_response_values_binary(
        classifier,
        X_binary,
        response_method="decision_function",
        pos_label=0,
        return_response_method_used=return_response_method_used,
    )

    # 断言第一个返回结果与分类器 decision_function 方法计算的结果相近且取反
    assert_allclose(results[0], classifier.decision_function(X_binary) * -1)

    # 断言第二个返回结果为 0，用于标记负类的位置
    assert results[1] == 0

    # 如果 return_response_method_used 为 True，则断言第三个返回结果为 "decision_function"
    if return_response_method_used:
        assert results[2] == "decision_function"


# 使用 pytest 的 parametrize 装饰器定义另一个测试函数，测试 `_get_response_values` 函数在多类分类器上的行为
@pytest.mark.parametrize(
    "estimator, response_method",
    [
        (DecisionTreeClassifier(max_depth=2, random_state=0), "predict_proba"),
        (DecisionTreeClassifier(max_depth=2, random_state=0), "predict_log_proba"),
        (LogisticRegression(), "decision_function"),
    ],
)
def test_get_response_values_multiclass(estimator, response_method):
    """Check that we can call `_get_response_values` with a multiclass estimator.
    It should return the predictions untouched.
    """

    # 使用给定的估计器和响应方法进行训练
    estimator.fit(X, y)

    # 调用 `_get_response_values` 函数，获取预测值和正类标签
    predictions, pos_label = _get_response_values(
        estimator, X, response_method=response_method
    )

    # 断言正类标签为 None
    assert pos_label is None

    # 断言预测结果的形状与估计器类别数匹配
    assert predictions.shape == (X.shape[0], len(estimator.classes_))

    # 根据不同的响应方法进行进一步的断言
    if response_method == "predict_proba":
        # 断言预测概率在 [0, 1] 之间
        assert np.logical_and(predictions >= 0, predictions <= 1).all()
    elif response_method == "predict_log_proba":
        # 断言预测的对数概率都小于等于 0
        assert (predictions <= 0.0).all()


# 定义测试函数，验证 `_get_response_values` 函数处理传递响应列表的行为
def test_get_response_values_with_response_list():
    """Check the behaviour of passing a list of responses to `_get_response_values`."""

    # 创建逻辑回归分类器，并用 X_binary 和 y_binary 进行训练
    classifier = LogisticRegression().fit(X_binary, y_binary)

    # 测试使用 `predict_proba` 响应方法
    y_pred, pos_label, response_method = _get_response_values(
        classifier,
        X_binary,
        response_method=["predict_proba", "decision_function"],
        return_response_method_used=True,
    )

    # 断言预测结果与分类器的 predict_proba 方法计算的结果相近
    assert_allclose(y_pred, classifier.predict_proba(X_binary)[:, 1])

    # 断言正类标签为 1
    assert pos_label == 1

    # 断言使用的响应方法为 "predict_proba"
    assert response_method == "predict_proba"

    # 测试使用 `decision_function` 响应方法
    y_pred, pos_label, response_method = _get_response_values(
        classifier,
        X_binary,
        response_method=["decision_function", "predict_proba"],
        return_response_method_used=True,
    )

    # 断言预测结果与分类器的 decision_function 方法计算的结果相近
    assert_allclose(y_pred, classifier.decision_function(X_binary))

    # 断言正类标签为 1
    assert pos_label == 1
    # 确保变量 response_method 的值等于字符串 "decision_function"
    assert response_method == "decision_function"
# 使用 pytest.mark.parametrize 装饰器，参数化测试用例，测试不同的 response_method
@pytest.mark.parametrize(
    "response_method", ["predict_proba", "decision_function", "predict"]
)
# 定义测试函数 test_get_response_values_multilabel_indicator，用于测试获取响应值的函数
def test_get_response_values_multilabel_indicator(response_method):
    # 创建多标签分类数据集 X, Y，确保结果可重现性
    X, Y = make_multilabel_classification(random_state=0)
    # 使用 ClassifierChain 模型，基于 LogisticRegression 拟合数据 X, Y
    estimator = ClassifierChain(LogisticRegression()).fit(X, Y)

    # 调用 _get_response_values 函数获取预测值 y_pred 和正标签 pos_label
    y_pred, pos_label = _get_response_values(
        estimator, X, response_method=response_method
    )
    # 断言正标签为 None
    assert pos_label is None
    # 断言预测值 y_pred 的形状与真实标签 Y 的形状相同
    assert y_pred.shape == Y.shape

    # 根据 response_method 分类进行不同的断言
    if response_method == "predict_proba":
        # 断言预测概率 y_pred 在 [0, 1] 范围内
        assert np.logical_and(y_pred >= 0, y_pred <= 1).all()
    elif response_method == "decision_function":
        # 断言 decision_function 返回的值不在 [0, 1] 范围内
        assert (y_pred < 0).sum() > 0
        assert (y_pred > 1).sum() > 0
    else:  # response_method == "predict"
        # 断言预测值 y_pred 只包含 0 或 1
        assert np.logical_or(y_pred == 0, y_pred == 1).all()
```