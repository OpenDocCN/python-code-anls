# `D:\src\scipysrc\scikit-learn\sklearn\inspection\_plot\tests\test_boundary_decision_display.py`

```
import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 测试框架

from sklearn.base import BaseEstimator, ClassifierMixin  # 导入基类和分类器混合类
from sklearn.datasets import (  # 导入数据集加载函数
    load_diabetes,
    load_iris,
    make_classification,
    make_multilabel_classification,
)
from sklearn.ensemble import IsolationForest  # 导入隔离森林算法
from sklearn.inspection import DecisionBoundaryDisplay  # 导入决策边界显示模块
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method  # 导入边界响应方法检查函数
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.preprocessing import scale  # 导入数据缩放函数
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入决策树分类器和回归器
from sklearn.utils._testing import (  # 导入测试辅助函数
    _convert_container,
    assert_allclose,
    assert_array_equal,
)

# TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
# 标记 Pytest，忽略特定警告消息
pytestmark = pytest.mark.filterwarnings(
    "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
    "matplotlib.*"
)

# 创建一个二维分类数据集 X 和对应标签 y
X, y = make_classification(
    n_informative=1,
    n_redundant=1,
    n_clusters_per_class=1,
    n_features=2,
    random_state=42,
)


def load_iris_2d_scaled():
    """加载鸢尾花数据集并进行数据缩放，只保留前两个特征"""
    X, y = load_iris(return_X_y=True)
    X = scale(X)[:, :2]
    return X, y


@pytest.fixture(scope="module")
def fitted_clf():
    """返回在数据集 X, y 上拟合的逻辑回归模型"""
    return LogisticRegression().fit(X, y)


def test_input_data_dimension(pyplot):
    """检查当 `X` 不具有完全 2 个特征时是否引发错误。"""
    X, y = make_classification(n_samples=10, n_features=4, random_state=0)

    clf = LogisticRegression().fit(X, y)
    msg = "n_features must be equal to 2. Got 4 instead."
    with pytest.raises(ValueError, match=msg):
        DecisionBoundaryDisplay.from_estimator(estimator=clf, X=X)


def test_check_boundary_response_method_error():
    """检查 `_check_boundary_response_method` 对不支持的情况是否引发错误。"""

    class MultiLabelClassifier:
        classes_ = [np.array([0, 1]), np.array([0, 1])]

    err_msg = "Multi-label and multi-output multi-class classifiers are not supported"
    with pytest.raises(ValueError, match=err_msg):
        _check_boundary_response_method(MultiLabelClassifier(), "predict", None)

    class MulticlassClassifier:
        classes_ = [0, 1, 2]

    err_msg = "Multiclass classifiers are only supported when `response_method` is"
    for response_method in ("predict_proba", "decision_function"):
        with pytest.raises(ValueError, match=err_msg):
            _check_boundary_response_method(
                MulticlassClassifier(), response_method, None
            )


@pytest.mark.parametrize(
    "estimator, response_method, class_of_interest, expected_prediction_method",
    [
        # 使用 DecisionTreeRegressor 模型进行预测，调用 predict 方法
        (DecisionTreeRegressor(), "predict", None, "predict"),
        # 使用 DecisionTreeRegressor 模型进行预测，调用 predict 方法
        (DecisionTreeRegressor(), "auto", None, "predict"),
        # 使用 LogisticRegression 模型拟合经过缩放的鸢尾花数据集，并进行预测，调用 predict 方法
        (LogisticRegression().fit(*load_iris_2d_scaled()), "predict", None, "predict"),
        # 使用 LogisticRegression 模型拟合经过缩放的鸢尾花数据集，并进行预测，调用 predict 方法
        (LogisticRegression().fit(*load_iris_2d_scaled()), "auto", None, "predict"),
        # 使用 LogisticRegression 模型拟合经过缩放的鸢尾花数据集，并进行预测概率估计，调用 predict_proba 方法
        (
            LogisticRegression().fit(*load_iris_2d_scaled()),
            "predict_proba",
            0,
            "predict_proba",
        ),
        # 使用 LogisticRegression 模型拟合经过缩放的鸢尾花数据集，并进行决策函数计算，调用 decision_function 方法
        (
            LogisticRegression().fit(*load_iris_2d_scaled()),
            "decision_function",
            0,
            "decision_function",
        ),
        # 使用 LogisticRegression 模型拟合给定的数据集 X, y，并根据提供的方法列表进行预测和决策函数计算
        (
            LogisticRegression().fit(X, y),
            "auto",
            None,
            ["decision_function", "predict_proba", "predict"],
        ),
        # 使用 LogisticRegression 模型拟合给定的数据集 X, y，并进行预测，调用 predict 方法
        (LogisticRegression().fit(X, y), "predict", None, "predict"),
        # 使用 LogisticRegression 模型拟合给定的数据集 X, y，并根据提供的方法列表进行预测概率估计和决策函数计算
        (
            LogisticRegression().fit(X, y),
            ["predict_proba", "decision_function"],
            None,
            ["predict_proba", "decision_function"],
        ),
    ],
# 定义测试函数，用于验证 `_check_boundary_response_method` 的行为是否符合预期
def test_check_boundary_response_method(
    estimator, response_method, class_of_interest, expected_prediction_method
):
    """Check the behaviour of `_check_boundary_response_method` for the supported
    cases.
    """
    # 调用 `_check_boundary_response_method` 函数，获取预测方法
    prediction_method = _check_boundary_response_method(
        estimator, response_method, class_of_interest
    )
    # 断言预测方法与期望的预测方法相符
    assert prediction_method == expected_prediction_method


# 使用参数化装饰器，测试多类错误情况
@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_multiclass_error(pyplot, response_method):
    """Check multiclass errors."""
    # 创建具有多类别的分类数据集
    X, y = make_classification(n_classes=3, n_informative=3, random_state=0)
    X = X[:, [0, 1]]
    # 使用逻辑回归模型拟合数据集
    lr = LogisticRegression().fit(X, y)

    # 设置错误消息
    msg = (
        "Multiclass classifiers are only supported when `response_method` is 'predict'"
        " or 'auto'"
    )
    # 断言使用非支持的响应方法时抛出 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        DecisionBoundaryDisplay.from_estimator(lr, X, response_method=response_method)


# 使用参数化装饰器，测试多类情况下的预期结果
@pytest.mark.parametrize("response_method", ["auto", "predict"])
def test_multiclass(pyplot, response_method):
    """Check multiclass gives expected results."""
    # 设置网格分辨率和扩展量
    grid_resolution = 10
    eps = 1.0
    # 创建具有多类别的分类数据集
    X, y = make_classification(n_classes=3, n_informative=3, random_state=0)
    X = X[:, [0, 1]]
    # 使用逻辑回归模型拟合数据集
    lr = LogisticRegression(random_state=0).fit(X, y)

    # 从估计器生成决策边界显示对象，使用指定的响应方法、网格分辨率和扩展量
    disp = DecisionBoundaryDisplay.from_estimator(
        lr, X, response_method=response_method, grid_resolution=grid_resolution, eps=1.0
    )

    # 计算 x0 和 x1 的最小值和最大值
    x0_min, x0_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    x1_min, x1_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    # 创建网格点，用于生成决策边界响应预测
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution),
    )
    response = lr.predict(np.c_[xx0.ravel(), xx1.ravel()])
    # 断言决策边界显示对象的响应和预测结果的一致性
    assert_allclose(disp.response, response.reshape(xx0.shape))
    assert_allclose(disp.xx0, xx0)
    assert_allclose(disp.xx1, xx1)


# 使用参数化装饰器，测试输入验证错误情况
@pytest.mark.parametrize(
    "kwargs, error_msg",
    [
        (
            {"plot_method": "hello_world"},
            r"plot_method must be one of contourf, contour, pcolormesh. Got hello_world"
            r" instead.",
        ),
        (
            {"grid_resolution": 1},
            r"grid_resolution must be greater than 1. Got 1 instead",
        ),
        (
            {"grid_resolution": -1},
            r"grid_resolution must be greater than 1. Got -1 instead",
        ),
        ({"eps": -1.1}, r"eps must be greater than or equal to 0. Got -1.1 instead"),
    ],
)
def test_input_validation_errors(pyplot, kwargs, error_msg, fitted_clf):
    """Check input validation from_estimator."""
    # 断言从估计器生成决策边界显示对象时，传入不合法参数会抛出 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=error_msg):
        DecisionBoundaryDisplay.from_estimator(fitted_clf, X, **kwargs)


# 测试 `plot` 输入错误情况的输入验证
def test_display_plot_input_error(pyplot, fitted_clf):
    """Check input validation for `plot`."""
    # 从估计器生成决策边界显示对象，设置网格分辨率为 5
    disp = DecisionBoundaryDisplay.from_estimator(fitted_clf, X, grid_resolution=5)
    # 使用 pytest 的断言来测试是否会抛出 ValueError 异常，并且异常消息中包含指定的字符串 "plot_method must be 'contourf'"
    with pytest.raises(ValueError, match="plot_method must be 'contourf'"):
        # 调用 disp 对象的 plot 方法，传入一个非法的 plot_method 参数 "hello_world"
        disp.plot(plot_method="hello_world")
# 使用 pytest 的 parametrize 装饰器为测试函数提供多个参数化的输入组合，以便执行多次测试
@pytest.mark.parametrize(
    "response_method", ["auto", "predict", "predict_proba", "decision_function"]
)
@pytest.mark.parametrize("plot_method", ["contourf", "contour"])
def test_decision_boundary_display_classifier(
    pyplot, fitted_clf, response_method, plot_method
):
    """Check that decision boundary is correct."""
    # 创建一个新的图形和坐标轴
    fig, ax = pyplot.subplots()
    eps = 2.0
    # 从拟合好的分类器生成决策边界显示对象
    disp = DecisionBoundaryDisplay.from_estimator(
        fitted_clf,
        X,
        grid_resolution=5,
        response_method=response_method,
        plot_method=plot_method,
        eps=eps,
        ax=ax,
    )
    # 断言生成的决策边界表面是 QuadContourSet 类型的对象
    assert isinstance(disp.surface_, pyplot.matplotlib.contour.QuadContourSet)
    # 断言决策边界显示对象的坐标轴与之前创建的坐标轴相同
    assert disp.ax_ == ax
    # 断言决策边界显示对象的图形与之前创建的图形相同
    assert disp.figure_ == fig

    # 提取数据的第一和第二特征列
    x0, x1 = X[:, 0], X[:, 1]

    # 计算第一特征列的最小和最大值，用于生成决策边界显示对象时的网格
    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    # 计算第二特征列的最小和最大值，用于生成决策边界显示对象时的网格
    x1_min, x1_max = x1.min() - eps, x1.max() + eps

    # 断言决策边界显示对象的第一特征列网格的最小值与预期值近似相等
    assert disp.xx0.min() == pytest.approx(x0_min)
    # 断言决策边界显示对象的第一特征列网格的最大值与预期值近似相等
    assert disp.xx0.max() == pytest.approx(x0_max)
    # 断言决策边界显示对象的第二特征列网格的最小值与预期值近似相等
    assert disp.xx1.min() == pytest.approx(x1_min)
    # 断言决策边界显示对象的第二特征列网格的最大值与预期值近似相等

    assert disp.xx1.max() == pytest.approx(x1_max)

    # 创建第二个图形和坐标轴用于第二个绘图方法
    fig2, ax2 = pyplot.subplots()
    # 改变绘图方法为 'pcolormesh'，更新图形
    disp.plot(plot_method="pcolormesh", ax=ax2, shading="auto")
    # 断言生成的决策边界表面是 QuadMesh 类型的对象
    assert isinstance(disp.surface_, pyplot.matplotlib.collections.QuadMesh)
    # 断言决策边界显示对象的坐标轴与第二次创建的坐标轴相同
    assert disp.ax_ == ax2
    # 断言决策边界显示对象的图形与第二次创建的图形相同


@pytest.mark.parametrize("response_method", ["auto", "predict", "decision_function"])
@pytest.mark.parametrize("plot_method", ["contourf", "contour"])
def test_decision_boundary_display_outlier_detector(
    pyplot, response_method, plot_method
):
    """Check that decision boundary is correct for outlier detector."""
    # 创建一个新的图形和坐标轴
    fig, ax = pyplot.subplots()
    eps = 2.0
    # 使用隔离森林拟合异常检测器
    outlier_detector = IsolationForest(random_state=0).fit(X, y)
    # 从异常检测器生成决策边界显示对象
    disp = DecisionBoundaryDisplay.from_estimator(
        outlier_detector,
        X,
        grid_resolution=5,
        response_method=response_method,
        plot_method=plot_method,
        eps=eps,
        ax=ax,
    )
    # 断言生成的决策边界表面是 QuadContourSet 类型的对象
    assert isinstance(disp.surface_, pyplot.matplotlib.contour.QuadContourSet)
    # 断言决策边界显示对象的坐标轴与之前创建的坐标轴相同
    assert disp.ax_ == ax
    # 断言决策边界显示对象的图形与之前创建的图形相同

    x0, x1 = X[:, 0], X[:, 1]

    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    x1_min, x1_max = x1.min() - eps, x1.max() + eps

    # 断言决策边界显示对象的第一特征列网格的最小值与预期值近似相等
    assert disp.xx0.min() == pytest.approx(x0_min)
    # 断言决策边界显示对象的第一特征列网格的最大值与预期值近似相等
    assert disp.xx0.max() == pytest.approx(x0_max)
    # 断言决策边界显示对象的第二特征列网格的最小值与预期值近似相等
    assert disp.xx1.min() == pytest.approx(x1_min)
    # 断言决策边界显示对象的第二特征列网格的最大值与预期值近似相等


@pytest.mark.parametrize("response_method", ["auto", "predict"])
@pytest.mark.parametrize("plot_method", ["contourf", "contour"])
def test_decision_boundary_display_regressor(pyplot, response_method, plot_method):
    """Check that we can display the decision boundary for a regressor."""
    # 加载糖尿病数据集的前两个特征和目标值
    X, y = load_diabetes(return_X_y=True)
    X = X[:, :2]
    # 使用决策树回归器拟合数据
    tree = DecisionTreeRegressor().fit(X, y)
    # 创建一个新的图形和坐标轴
    fig, ax = pyplot.subplots()
    eps = 2.0
    # 使用给定的决策树模型和数据集创建 DecisionBoundaryDisplay 对象
    disp = DecisionBoundaryDisplay.from_estimator(
        tree,
        X,
        response_method=response_method,
        ax=ax,
        eps=eps,
        plot_method=plot_method,
    )
    # 断言确保生成的表面是 QuadContourSet 对象
    assert isinstance(disp.surface_, pyplot.matplotlib.contour.QuadContourSet)
    # 断言确保显示对象使用了正确的坐标轴
    assert disp.ax_ == ax
    # 断言确保显示对象使用了正确的图形对象
    assert disp.figure_ == fig

    # 提取数据集中的第一列和第二列数据
    x0, x1 = X[:, 0], X[:, 1]

    # 计算第一列数据的最小和最大值，并添加一些余量 eps
    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    # 计算第二列数据的最小和最大值，并添加一些余量 eps
    x1_min, x1_max = x1.min() - eps, x1.max() + eps

    # 断言确保显示对象的网格 xx0 和 xx1 的最小值和最大值与预期一致
    assert disp.xx0.min() == pytest.approx(x0_min)
    assert disp.xx0.max() == pytest.approx(x0_max)
    assert disp.xx1.min() == pytest.approx(x1_min)
    assert disp.xx1.max() == pytest.approx(x1_max)

    # 创建新的图形和坐标轴对象，用于第二个图形
    fig2, ax2 = pyplot.subplots()
    # 更改绘图方法为 pcolormesh，并绘制第二个图形
    disp.plot(plot_method="pcolormesh", ax=ax2, shading="auto")
    # 断言确保生成的表面是 QuadMesh 对象
    assert isinstance(disp.surface_, pyplot.matplotlib.collections.QuadMesh)
    # 断言确保显示对象使用了正确的坐标轴
    assert disp.ax_ == ax2
    # 断言确保显示对象使用了正确的图形对象
    assert disp.figure_ == fig2
@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，允许在单个测试函数中多次运行测试用例
    "response_method, msg",  # 参数化的参数名称列表
    [  # 参数化的参数值列表
        (
            "predict_proba",  # 第一个参数值: "predict_proba"
            "MyClassifier has none of the following attributes: predict_proba",  # 对应的错误消息
        ),
        (
            "decision_function",  # 第二个参数值: "decision_function"
            "MyClassifier has none of the following attributes: decision_function",  # 对应的错误消息
        ),
        (
            "auto",  # 第三个参数值: "auto"
            (  # 对应的错误消息，包含多个属性
                "MyClassifier has none of the following attributes: decision_function, "
                "predict_proba, predict"
            ),
        ),
        (
            "bad_method",  # 第四个参数值: "bad_method"
            "MyClassifier has none of the following attributes: bad_method",  # 对应的错误消息
        ),
    ],
)
def test_error_bad_response(pyplot, response_method, msg):
    """Check errors for bad response."""

    class MyClassifier(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.fitted_ = True  # 设置标志位表明分类器已经拟合
            self.classes_ = [0, 1]  # 设置分类器的类别
            return self  # 返回分类器实例

    clf = MyClassifier().fit(X, y)  # 使用自定义分类器拟合数据集 X, y

    with pytest.raises(AttributeError, match=msg):  # 使用 pytest 断言检查是否抛出特定类型和消息的异常
        DecisionBoundaryDisplay.from_estimator(clf, X, response_method=response_method)


@pytest.mark.parametrize(  # 参数化装饰器，用于测试多标签分类器
    "response_method",  # 参数化的参数名称
    ["auto", "predict", "predict_proba"],  # 参数化的参数值列表
)
def test_multilabel_classifier_error(pyplot, response_method):
    """Check that multilabel classifier raises correct error."""
    X, y = make_multilabel_classification(random_state=0)  # 生成多标签分类的样本数据
    X = X[:, :2]  # 仅使用前两个特征列
    tree = DecisionTreeClassifier().fit(X, y)  # 使用决策树分类器拟合数据

    msg = "Multi-label and multi-output multi-class classifiers are not supported"  # 错误消息
    with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言检查是否抛出特定类型和消息的异常
        DecisionBoundaryDisplay.from_estimator(
            tree,
            X,
            response_method=response_method,  # 指定响应方法参数
        )


@pytest.mark.parametrize(  # 参数化装饰器，用于测试多输出多类分类器
    "response_method",  # 参数化的参数名称
    ["auto", "predict", "predict_proba"],  # 参数化的参数值列表
)
def test_multi_output_multi_class_classifier_error(pyplot, response_method):
    """Check that multi-output multi-class classifier raises correct error."""
    X = np.asarray([[0, 1], [1, 2]])  # 特征数据
    y = np.asarray([["tree", "cat"], ["cat", "tree"]])  # 类别数据
    tree = DecisionTreeClassifier().fit(X, y)  # 使用决策树分类器拟合数据

    msg = "Multi-label and multi-output multi-class classifiers are not supported"  # 错误消息
    with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言检查是否抛出特定类型和消息的异常
        DecisionBoundaryDisplay.from_estimator(
            tree,
            X,
            response_method=response_method,  # 指定响应方法参数
        )


def test_multioutput_regressor_error(pyplot):
    """Check that multioutput regressor raises correct error."""
    X = np.asarray([[0, 1], [1, 2]])  # 特征数据
    y = np.asarray([[0, 1], [4, 1]])  # 目标数据
    tree = DecisionTreeRegressor().fit(X, y)  # 使用决策树回归器拟合数据
    with pytest.raises(ValueError, match="Multi-output regressors are not supported"):  # 使用 pytest 断言检查是否抛出特定类型和消息的异常
        DecisionBoundaryDisplay.from_estimator(tree, X, response_method="predict")


@pytest.mark.parametrize(  # 参数化装饰器，用于测试回归器不支持的响应方法
    "response_method",  # 参数化的参数名称
    ["predict_proba", "decision_function", ["predict_proba", "predict"]],  # 参数化的参数值列表
)
def test_regressor_unsupported_response(pyplot, response_method):
    """Check that regressor unsupported response methods raise correct error."""
    """Check that we can display the decision boundary for a regressor."""
    # 加载糖尿病数据集，并返回特征矩阵 X 和目标向量 y
    X, y = load_diabetes(return_X_y=True)
    # 仅保留特征矩阵 X 的前两个特征列
    X = X[:, :2]
    # 使用默认参数创建决策树回归器，并用数据 X 和目标 y 进行训练
    tree = DecisionTreeRegressor().fit(X, y)
    # 定义错误信息字符串，用于捕获 ValueError 异常
    err_msg = "should either be a classifier to be used with response_method"
    # 使用 pytest 模块断言捕获 ValueError 异常，并匹配指定的错误信息字符串
    with pytest.raises(ValueError, match=err_msg):
        # 从给定的估计器（决策树回归器）和数据 X 创建 DecisionBoundaryDisplay 对象，
        # 使用指定的响应方法 response_method
        DecisionBoundaryDisplay.from_estimator(tree, X, response_method=response_method)
@pytest.mark.filterwarnings(
    # 忽略以下警告，因为分类器是在 NumPy 数组上拟合的
    "ignore:X has feature names, but LogisticRegression was fitted without"
)
def test_dataframe_labels_used(pyplot, fitted_clf):
    """检查 pandas 是否使用了列名。"""
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个 DataFrame，使用指定的列名 "col_x" 和 "col_y"
    df = pd.DataFrame(X, columns=["col_x", "col_y"])

    # 默认情况下，使用 pandas 的列名
    _, ax = pyplot.subplots()
    # 使用 DecisionBoundaryDisplay 类从拟合好的分类器和 DataFrame 创建显示
    disp = DecisionBoundaryDisplay.from_estimator(fitted_clf, df, ax=ax)
    assert ax.get_xlabel() == "col_x"
    assert ax.get_ylabel() == "col_y"

    # 第二次调用 plot 将使用列名
    fig, ax = pyplot.subplots()
    disp.plot(ax=ax)
    assert ax.get_xlabel() == "col_x"
    assert ax.get_ylabel() == "col_y"

    # 带有标签的轴不会被覆盖
    fig, ax = pyplot.subplots()
    ax.set(xlabel="hello", ylabel="world")
    disp.plot(ax=ax)
    assert ax.get_xlabel() == "hello"
    assert ax.get_ylabel() == "world"

    # 只有在 plot 方法中提供了标签时，标签才会被覆盖
    disp.plot(ax=ax, xlabel="overwritten_x", ylabel="overwritten_y")
    assert ax.get_xlabel() == "overwritten_x"
    assert ax.get_ylabel() == "overwritten_y"

    # 如果在 from_estimator 中提供了标签，则不会推断标签
    _, ax = pyplot.subplots()
    disp = DecisionBoundaryDisplay.from_estimator(
        fitted_clf, df, ax=ax, xlabel="overwritten_x", ylabel="overwritten_y"
    )
    assert ax.get_xlabel() == "overwritten_x"
    assert ax.get_ylabel() == "overwritten_y"


def test_string_target(pyplot):
    """检查对于使用字符串标签训练的分类器，决策边界是否正常工作。"""
    iris = load_iris()
    X = iris.data[:, [0, 1]]

    # 使用字符串作为目标
    y = iris.target_names[iris.target]
    log_reg = LogisticRegression().fit(X, y)

    # 不应该引发异常
    DecisionBoundaryDisplay.from_estimator(
        log_reg,
        X,
        grid_resolution=5,
        response_method="predict",
    )


@pytest.mark.parametrize("constructor_name", ["pandas", "polars"])
def test_dataframe_support(pyplot, constructor_name):
    """检查在拟合和显示中传递 DataFrame 是否不会引发警告。

    非回归测试，针对以下问题：
    * https://github.com/scikit-learn/scikit-learn/issues/23311
    * https://github.com/scikit-learn/scikit-learn/issues/28717
    """
    # 将数据转换为指定的容器格式（例如 pandas 或 polars DataFrame），使用指定的列名 "col_x" 和 "col_y"
    df = _convert_container(
        X, constructor_name=constructor_name, columns_name=["col_x", "col_y"]
    )
    # 使用 LogisticRegression 拟合 DataFrame
    estimator = LogisticRegression().fit(df, y)

    with warnings.catch_warnings():
        # 禁止与特征名验证相关的警告
        warnings.simplefilter("error", UserWarning)
        # 使用 DecisionBoundaryDisplay 类从拟合好的估计器和 DataFrame 创建显示
        DecisionBoundaryDisplay.from_estimator(estimator, df, response_method="predict")


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_class_of_interest_binary(pyplot, response_method):
    """Check the behaviour of passing `class_of_interest` for plotting the output of
    `predict_proba` and `decision_function` in the binary case.
    """
    # 加载鸢尾花数据集的前100个样本，仅使用前两个特征
    iris = load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]
    # 确保目标变量只有两个类别：0和1
    assert_array_equal(np.unique(y), [0, 1])

    # 使用Logistic回归拟合数据
    estimator = LogisticRegression().fit(X, y)
    
    # 创建默认的决策边界展示对象，class_of_interest参数为None
    disp_default = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method=response_method,
        class_of_interest=None,
    )
    # 创建针对第二类别的决策边界展示对象，class_of_interest参数为第二类别的索引
    disp_class_1 = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method=response_method,
        class_of_interest=estimator.classes_[1],
    )

    # 断言默认展示对象和针对第二类别的展示对象的响应值相等
    assert_allclose(disp_default.response, disp_class_1.response)

    # 创建针对第一类别的决策边界展示对象，class_of_interest参数为第一类别的索引
    disp_class_0 = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method=response_method,
        class_of_interest=estimator.classes_[0],
    )

    # 如果response_method为"predict_proba"，则断言默认展示对象的响应值与1减去针对第一类别展示对象的响应值相等
    if response_method == "predict_proba":
        assert_allclose(disp_default.response, 1 - disp_class_0.response)
    else:
        # 否则，断言response_method为"decision_function"，并且默认展示对象的响应值与针对第一类别展示对象的响应值的负数相等
        assert response_method == "decision_function"
        assert_allclose(disp_default.response, -disp_class_0.response)
@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_class_of_interest_multiclass(pyplot, response_method):
    """检查在多类情况下，传递 `class_of_interest` 对于绘制 `predict_proba` 和 `decision_function` 输出的行为。
    """
    # 载入鸢尾花数据集
    iris = load_iris()
    # 只使用前两个特征
    X = iris.data[:, :2]
    # 目标变量是数值标签
    y = iris.target
    # 指定感兴趣类别的索引
    class_of_interest_idx = 2

    # 使用逻辑回归拟合模型
    estimator = LogisticRegression().fit(X, y)
    # 创建 DecisionBoundaryDisplay 对象，传入估计器、特征数据、响应方法和感兴趣类别索引
    disp = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method=response_method,
        class_of_interest=class_of_interest_idx,
    )

    # 检查绘制的响应值与预期值是否相近
    grid = np.concatenate([disp.xx0.reshape(-1, 1), disp.xx1.reshape(-1, 1)], axis=1)
    response = getattr(estimator, response_method)(grid)[:, class_of_interest_idx]
    assert_allclose(response.reshape(*disp.response.shape), disp.response)

    # 使用字符串形式的目标变量名再次进行相同的测试
    y = iris.target_names[iris.target]
    estimator = LogisticRegression().fit(X, y)

    disp = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method=response_method,
        class_of_interest=iris.target_names[class_of_interest_idx],
    )

    grid = np.concatenate([disp.xx0.reshape(-1, 1), disp.xx1.reshape(-1, 1)], axis=1)
    response = getattr(estimator, response_method)(grid)[:, class_of_interest_idx]
    assert_allclose(response.reshape(*disp.response.shape), disp.response)

    # 检查对于未知标签是否引发错误
    # 这个测试应该已经在 `_get_response_values` 中处理，但我们也可以在这里进行测试
    err_msg = "class_of_interest=2 is not a valid label: It should be one of"
    with pytest.raises(ValueError, match=err_msg):
        DecisionBoundaryDisplay.from_estimator(
            estimator,
            X,
            response_method=response_method,
            class_of_interest=class_of_interest_idx,
        )

    # TODO: 当处理 `class_of_interest=None` 的多类情况时，移除这个测试
    # 显示决策函数的最大值或预测概率的最大值。
    err_msg = "Multiclass classifiers are only supported"
    with pytest.raises(ValueError, match=err_msg):
        DecisionBoundaryDisplay.from_estimator(
            estimator,
            X,
            response_method=response_method,
            class_of_interest=None,
        )


def test_subclass_named_constructors_return_type_is_subclass(pyplot):
    """检查当子类化时，命名构造函数返回正确的类型。

    非回归测试:
    https://github.com/scikit-learn/scikit-learn/pull/27675
    """
    # 使用逻辑回归拟合模型
    clf = LogisticRegression().fit(X, y)

    # 创建 DecisionBoundaryDisplay 的子类 SubclassOfDisplay
    class SubclassOfDisplay(DecisionBoundaryDisplay):
        pass

    # 使用子类化的 DisplayBoundaryDisplay，从估计器和特征数据创建曲线对象
    curve = SubclassOfDisplay.from_estimator(estimator=clf, X=X)
    # 确保变量 curve 是 SubclassOfDisplay 的子类的实例
    assert isinstance(curve, SubclassOfDisplay)
```