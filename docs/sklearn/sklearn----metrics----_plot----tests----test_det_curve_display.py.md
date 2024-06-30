# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\tests\test_det_curve_display.py`

```
# 导入必要的库
import numpy as np
import pytest
from numpy.testing import assert_allclose

# 导入用于测试的数据集和模型
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import DetCurveDisplay, det_curve

# 使用 pytest 的参数化装饰器，为函数 test_det_curve_display 添加参数组合测试
@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("with_strings", [True, False])
def test_det_curve_display(
    pyplot, constructor_name, response_method, with_sample_weight, with_strings
):
    # 加载鸢尾花数据集
    X, y = load_iris(return_X_y=True)
    # 仅保留两个类别的数据
    X, y = X[y < 2], y[y < 2]

    pos_label = None
    # 如果需要字符串标签，则将 y 转换为字符串数组，并设置正类标签为 "c"
    if with_strings:
        y = np.array(["c", "b"])[y]
        pos_label = "c"

    # 如果需要样本权重，则生成随机样本权重数组
    if with_sample_weight:
        rng = np.random.RandomState(42)
        sample_weight = rng.randint(1, 4, size=(X.shape[0]))
    else:
        sample_weight = None

    # 创建逻辑回归模型并拟合数据
    lr = LogisticRegression()
    lr.fit(X, y)
    # 根据选择的响应方法（概率预测或决策函数），获取预测结果
    y_pred = getattr(lr, response_method)(X)
    # 如果预测结果是二维的，则取第二列作为正类预测概率
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]

    # 对构造函数名进行断言，确保在可选的两种构造函数名中
    assert constructor_name in ("from_estimator", "from_predictions")

    # 定义通用的参数字典
    common_kwargs = {
        "name": lr.__class__.__name__,
        "alpha": 0.8,
        "sample_weight": sample_weight,
        "pos_label": pos_label,
    }
    # 根据构造函数名选择不同的方法创建 DetCurveDisplay 对象
    if constructor_name == "from_estimator":
        disp = DetCurveDisplay.from_estimator(lr, X, y, **common_kwargs)
    else:
        disp = DetCurveDisplay.from_predictions(y, y_pred, **common_kwargs)

    # 计算 DET 曲线需要的假阳性率（fpr）、假阴性率（fnr）和阈值数组
    fpr, fnr, _ = det_curve(
        y,
        y_pred,
        sample_weight=sample_weight,
        pos_label=pos_label,
    )

    # 使用 assert_allclose 断言 disp 对象中的 fpr 和计算得到的 fpr 近似相等
    assert_allclose(disp.fpr, fpr)
    # 使用 assert_allclose 断言 disp 对象中的 fnr 和计算得到的 fnr 近似相等
    assert_allclose(disp.fnr, fnr)

    # 断言 disp 对象的估计器名称为 "LogisticRegression"
    assert disp.estimator_name == "LogisticRegression"

    # 使用 pyplot fixture 进行绘图测试，这里确保不会失败
    import matplotlib as mpl  # noqal

    # 断言 disp 对象中的线条类型为 mpl.lines.Line2D 对象
    assert isinstance(disp.line_, mpl.lines.Line2D)
    # 断言 disp 对象中线条的透明度为 0.8
    assert disp.line_.get_alpha() == 0.8
    # 断言 disp 对象中的坐标轴为 mpl.axes.Axes 对象
    assert isinstance(disp.ax_, mpl.axes.Axes)
    # 断言 disp 对象中的图形为 mpl.figure.Figure 对象
    assert isinstance(disp.figure_, mpl.figure.Figure)
    # 断言 disp 对象中线条的标签为 "LogisticRegression"
    assert disp.line_.get_label() == "LogisticRegression"

    # 根据预期的正类标签，生成预期的 Y 轴标签和 X 轴标签
    expected_pos_label = 1 if pos_label is None else pos_label
    expected_ylabel = f"False Negative Rate (Positive label: {expected_pos_label})"
    expected_xlabel = f"False Positive Rate (Positive label: {expected_pos_label})"
    # 断言 disp 对象中的 Y 轴标签和预期的 Y 轴标签相等
    assert disp.ax_.get_ylabel() == expected_ylabel
    # 断言 disp 对象中的 X 轴标签和预期的 X 轴标签相等
    assert disp.ax_.get_xlabel() == expected_xlabel


# 使用 pytest 的参数化装饰器，为函数 test_det_curve_display_default_name 添加参数组合测试
@pytest.mark.parametrize(
    "constructor_name, expected_clf_name",
    [
        ("from_estimator", "LogisticRegression"),
        ("from_predictions", "Classifier"),
    ],
)
def test_det_curve_display_default_name(
    pyplot,
    constructor_name,
    expected_clf_name,
):
    # 检查当未提供 'name' 参数时，在图表中显示的默认名称
    X, y = load_iris(return_X_y=True)
    # 仅保留目标数据集中的两个类别，将特征数据 X 和标签数据 y 进行筛选
    X, y = X[y < 2], y[y < 2]

    # 使用逻辑回归模型拟合筛选后的数据
    lr = LogisticRegression().fit(X, y)

    # 对 X 进行预测，得到预测的概率值（属于类别 1 的概率）
    y_pred = lr.predict_proba(X)[:, 1]

    # 根据构造器名称选择不同的显示方式，构建 DetCurveDisplay 对象 disp
    if constructor_name == "from_estimator":
        disp = DetCurveDisplay.from_estimator(lr, X, y)
    else:
        disp = DetCurveDisplay.from_predictions(y, y_pred)

    # 断言确保 DetCurveDisplay 对象的估计器名称与预期一致
    assert disp.estimator_name == expected_clf_name

    # 断言确保 DetCurveDisplay 对象的线条标签与预期一致
    assert disp.line_.get_label() == expected_clf_name
```