# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\tests\test_predict_error_display.py`

```
import pytest  # 导入 pytest 测试框架
from numpy.testing import assert_allclose  # 导入 numpy 的数组比较函数

from sklearn.datasets import load_diabetes  # 导入糖尿病数据集
from sklearn.exceptions import NotFittedError  # 导入未拟合错误类
from sklearn.linear_model import Ridge  # 导入岭回归模型
from sklearn.metrics import PredictionErrorDisplay  # 导入预测误差显示类

X, y = load_diabetes(return_X_y=True)  # 加载糖尿病数据集并分别赋值给 X 和 y


@pytest.fixture
def regressor_fitted():
    """返回已拟合的岭回归模型的实例"""
    return Ridge().fit(X, y)


@pytest.mark.parametrize(
    "regressor, params, err_type, err_msg",
    [
        (
            Ridge().fit(X, y),
            {"subsample": -1},
            ValueError,
            "When an integer, subsample=-1 should be",
        ),
        (
            Ridge().fit(X, y),
            {"subsample": 20.0},
            ValueError,
            "When a floating-point, subsample=20.0 should be",
        ),
        (
            Ridge().fit(X, y),
            {"subsample": -20.0},
            ValueError,
            "When a floating-point, subsample=-20.0 should be",
        ),
        (
            Ridge().fit(X, y),
            {"kind": "xxx"},
            ValueError,
            "`kind` must be one of",
        ),
    ],
)
@pytest.mark.parametrize("class_method", ["from_estimator", "from_predictions"])
def test_prediction_error_display_raise_error(
    pyplot, class_method, regressor, params, err_type, err_msg
):
    """验证在参数验证时引发正确的错误。"""
    with pytest.raises(err_type, match=err_msg):
        if class_method == "from_estimator":
            PredictionErrorDisplay.from_estimator(regressor, X, y, **params)
        else:
            y_pred = regressor.predict(X)
            PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred, **params)


def test_from_estimator_not_fitted(pyplot):
    """验证当传递的回归器未拟合时，引发 `NotFittedError` 异常。"""
    regressor = Ridge()
    with pytest.raises(NotFittedError, match="is not fitted yet."):
        PredictionErrorDisplay.from_estimator(regressor, X, y)


@pytest.mark.parametrize("class_method", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("kind", ["actual_vs_predicted", "residual_vs_predicted"])
def test_prediction_error_display(pyplot, regressor_fitted, class_method, kind):
    """验证显示的默认行为。"""
    if class_method == "from_estimator":
        display = PredictionErrorDisplay.from_estimator(
            regressor_fitted, X, y, kind=kind
        )
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(
            y_true=y, y_pred=y_pred, kind=kind
        )

    if kind == "actual_vs_predicted":
        assert_allclose(display.line_.get_xdata(), display.line_.get_ydata())
        assert display.ax_.get_xlabel() == "Predicted values"
        assert display.ax_.get_ylabel() == "Actual values"
        assert display.line_ is not None
    else:
        # 断言显示对象的 X 轴标签为 "Predicted values"
        assert display.ax_.get_xlabel() == "Predicted values"
        # 断言显示对象的 Y 轴标签为 "Residuals (actual - predicted)"
        assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"
        # 断言显示对象的线条属性不为 None
        assert display.line_ is not None

    # 断言显示对象没有图例
    assert display.ax_.get_legend() is None
@pytest.mark.parametrize("class_method", ["from_estimator", "from_predictions"])
# 参数化测试，使用两个参数化：
# - class_method：指定方法来生成 PredictionErrorDisplay 对象，可以是 "from_estimator" 或 "from_predictions"
# - subsample, expected_size：用于测试不同的子采样率和期望的大小
@pytest.mark.parametrize(
    "subsample, expected_size",
    [(5, 5), (0.1, int(X.shape[0] * 0.1)), (None, X.shape[0])],
)
# 定义测试函数，测试绘制预测误差图时的子采样行为
def test_plot_prediction_error_subsample(
    pyplot, regressor_fitted, class_method, subsample, expected_size
):
    """Check the behaviour of `subsample`."""
    # 根据不同的 class_method 调用不同的方法生成 PredictionErrorDisplay 对象
    if class_method == "from_estimator":
        # 从回归器直接生成 PredictionErrorDisplay 对象
        display = PredictionErrorDisplay.from_estimator(
            regressor_fitted, X, y, subsample=subsample
        )
    else:
        # 从预测值生成 PredictionErrorDisplay 对象
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(
            y_true=y, y_pred=y_pred, subsample=subsample
        )
    # 断言生成的 scatter plot 的点数符合期望的大小
    assert len(display.scatter_.get_offsets()) == expected_size


@pytest.mark.parametrize("class_method", ["from_estimator", "from_predictions"])
# 参数化测试，使用参数：
# - class_method：指定方法来生成 PredictionErrorDisplay 对象，可以是 "from_estimator" 或 "from_predictions"
def test_plot_prediction_error_ax(pyplot, regressor_fitted, class_method):
    """Check that we can pass an axis to the display."""
    # 创建新的图形和坐标轴
    _, ax = pyplot.subplots()
    # 根据不同的 class_method 调用不同的方法生成 PredictionErrorDisplay 对象，并传入指定的坐标轴 ax
    if class_method == "from_estimator":
        display = PredictionErrorDisplay.from_estimator(regressor_fitted, X, y, ax=ax)
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(
            y_true=y, y_pred=y_pred, ax=ax
        )
    # 断言生成的 display 对象中的坐标轴与预期的 ax 对象一致
    assert display.ax_ is ax


@pytest.mark.parametrize("class_method", ["from_estimator", "from_predictions"])
# 参数化测试，使用参数：
# - class_method：指定方法来生成 PredictionErrorDisplay 对象，可以是 "from_estimator" 或 "from_predictions"
def test_prediction_error_custom_artist(pyplot, regressor_fitted, class_method):
    """Check that we can tune the style of the lines."""
    # 额外的参数，用于设置自定义的绘图风格
    extra_params = {
        "kind": "actual_vs_predicted",
        "scatter_kwargs": {"color": "red"},
        "line_kwargs": {"color": "black"},
    }
    # 根据不同的 class_method 调用不同的方法生成 PredictionErrorDisplay 对象，并传入额外的参数
    if class_method == "from_estimator":
        display = PredictionErrorDisplay.from_estimator(
            regressor_fitted, X, y, **extra_params
        )
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(
            y_true=y, y_pred=y_pred, **extra_params
        )

    # 断言生成的 display 对象中的线条颜色符合预期的黑色
    assert display.line_.get_color() == "black"
    # 断言生成的 display 对象中的 scatter plot 点的边缘颜色接近预期的红色
    assert_allclose(display.scatter_.get_edgecolor(), [[1.0, 0.0, 0.0, 0.8]])

    # 使用默认值创建一个 display 对象
    if class_method == "from_estimator":
        display = PredictionErrorDisplay.from_estimator(regressor_fitted, X, y)
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred)
    # 关闭所有图形
    pyplot.close("all")

    # 使用额外的参数重新绘制 display 对象
    display.plot(**extra_params)
    # 再次断言生成的 display 对象中的线条颜色符合预期的黑色
    assert display.line_.get_color() == "black"
    # 再次断言生成的 display 对象中的 scatter plot 点的边缘颜色接近预期的红色
    assert_allclose(display.scatter_.get_edgecolor(), [[1.0, 0.0, 0.0, 0.8]])
```