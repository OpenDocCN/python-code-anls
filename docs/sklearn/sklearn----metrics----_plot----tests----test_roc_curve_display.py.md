# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\tests\test_roc_curve_display.py`

```
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import trapezoid


@pytest.fixture(scope="module")
def data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def data_binary(data):
    X, y = data
    return X[y < 2], y[y < 2]


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("drop_intermediate", [True, False])
@pytest.mark.parametrize("with_strings", [True, False])
@pytest.mark.parametrize(
    "constructor_name, default_name",
    [
        ("from_estimator", "LogisticRegression"),
        ("from_predictions", "Classifier"),
    ],
)
def test_roc_curve_display_plotting(
    pyplot,
    response_method,
    data_binary,
    with_sample_weight,
    drop_intermediate,
    with_strings,
    constructor_name,
    default_name,
):
    """Check the overall plotting behaviour."""
    X, y = data_binary

    # 如果设置了使用字符串表示类别，则将y转换为字符串数组，并设置正类标签为"c"
    pos_label = None
    if with_strings:
        y = np.array(["c", "b"])[y]
        pos_label = "c"

    # 如果设置了使用样本权重，则生成随机种子，创建样本权重数组；否则设为None
    if with_sample_weight:
        rng = np.random.RandomState(42)
        sample_weight = rng.randint(1, 4, size=(X.shape[0]))
    else:
        sample_weight = None

    # 创建LogisticRegression模型，并拟合数据
    lr = LogisticRegression()
    lr.fit(X, y)

    # 根据指定的响应方法获取预测值
    y_pred = getattr(lr, response_method)(X)
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, 1]

    # 根据constructor_name选择使用不同的方法生成ROC曲线展示对象
    if constructor_name == "from_estimator":
        display = RocCurveDisplay.from_estimator(
            lr,
            X,
            y,
            sample_weight=sample_weight,
            drop_intermediate=drop_intermediate,
            pos_label=pos_label,
            alpha=0.8,
        )
    else:
        display = RocCurveDisplay.from_predictions(
            y,
            y_pred,
            sample_weight=sample_weight,
            drop_intermediate=drop_intermediate,
            pos_label=pos_label,
            alpha=0.8,
        )

    # 计算ROC曲线的假阳率（FPR）、真阳率（TPR）及阈值（_）
    fpr, tpr, _ = roc_curve(
        y,
        y_pred,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
        pos_label=pos_label,
    )

    # 断言ROC曲线的AUC值与计算得到的AUC值接近
    assert_allclose(display.roc_auc, auc(fpr, tpr))
    # 断言展示对象中的假阳率（FPR）与计算得到的FPR接近
    assert_allclose(display.fpr, fpr)
    # 断言展示对象中的真阳率（TPR）与计算得到的TPR接近
    assert_allclose(display.tpr, tpr)

    # 断言展示对象的估计器名称等于默认名称
    assert display.estimator_name == default_name

    import matplotlib as mpl  # noqal

    # 断言展示对象中的线条类型为Line2D对象
    assert isinstance(display.line_, mpl.lines.Line2D)
    # 断言展示对象中线条的透明度为0.8
    assert display.line_.get_alpha() == 0.8
    # 断言展示对象中的坐标轴类型为Axes对象
    assert isinstance(display.ax_, mpl.axes.Axes)
    # 断言确认 display.figure_ 是 mpl.figure.Figure 类的实例
    assert isinstance(display.figure_, mpl.figure.Figure)
    # 断言确认 display.ax_ 的可调整性为 "box"
    assert display.ax_.get_adjustable() == "box"
    # 断言确认 display.ax_ 的长宽比为 "equal" 或者 1.0
    assert display.ax_.get_aspect() in ("equal", 1.0)
    # 断言确认 display.ax_ 的 x 轴和 y 轴的限制范围均为 (-0.01, 1.01)

    # 构造预期的标签，格式为 "{default_name} (AUC = {display.roc_auc:.2f})"
    expected_label = f"{default_name} (AUC = {display.roc_auc:.2f})"
    # 断言确认 display.line_ 的标签与预期标签一致
    assert display.line_.get_label() == expected_label

    # 确定预期的正类标签，如果 pos_label 为 None，则为 1；否则为 pos_label
    expected_pos_label = 1 if pos_label is None else pos_label
    # 构造预期的 y 轴标签，格式为 "True Positive Rate (Positive label: {expected_pos_label})"
    expected_ylabel = f"True Positive Rate (Positive label: {expected_pos_label})"
    # 构造预期的 x 轴标签，格式为 "False Positive Rate (Positive label: {expected_pos_label})"
    expected_xlabel = f"False Positive Rate (Positive label: {expected_pos_label})"

    # 断言确认 display.ax_ 的 y 轴标签与预期标签一致
    assert display.ax_.get_ylabel() == expected_ylabel
    # 断言确认 display.ax_ 的 x 轴标签与预期标签一致
    assert display.ax_.get_xlabel() == expected_xlabel
@pytest.mark.parametrize("plot_chance_level", [True, False])
# 使用pytest的@parametrize装饰器，为plot_chance_level参数分别测试True和False两种情况

@pytest.mark.parametrize(
    "chance_level_kw",
    [None, {"linewidth": 1, "color": "red", "label": "DummyEstimator"}],
)
# 使用pytest的@parametrize装饰器，为chance_level_kw参数分别测试None和具有特定属性的字典两种情况

@pytest.mark.parametrize(
    "constructor_name",
    ["from_estimator", "from_predictions"],
)
# 使用pytest的@parametrize装饰器，为constructor_name参数分别测试"from_estimator"和"from_predictions"两种情况
def test_roc_curve_chance_level_line(
    pyplot,
    data_binary,
    plot_chance_level,
    chance_level_kw,
    constructor_name,
):
    """Check the chance level line plotting behaviour."""
    # 检查绘制ROC曲线中的Chance Level线的行为

    X, y = data_binary
    # 解包二进制数据

    lr = LogisticRegression()
    lr.fit(X, y)
    # 使用逻辑回归拟合数据

    y_pred = getattr(lr, "predict_proba")(X)
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
    # 获取预测的概率值，并根据维度对预测结果进行处理

    if constructor_name == "from_estimator":
        display = RocCurveDisplay.from_estimator(
            lr,
            X,
            y,
            alpha=0.8,
            plot_chance_level=plot_chance_level,
            chance_level_kw=chance_level_kw,
        )
    else:
        display = RocCurveDisplay.from_predictions(
            y,
            y_pred,
            alpha=0.8,
            plot_chance_level=plot_chance_level,
            chance_level_kw=chance_level_kw,
        )
    # 根据constructor_name选择不同的方法创建RocCurveDisplay对象

    import matplotlib as mpl  # noqa
    # 导入matplotlib模块

    assert isinstance(display.line_, mpl.lines.Line2D)
    # 断言display.line_是mpl.lines.Line2D的实例
    assert display.line_.get_alpha() == 0.8
    # 断言display.line_的透明度为0.8
    assert isinstance(display.ax_, mpl.axes.Axes)
    # 断言display.ax_是mpl.axes.Axes的实例
    assert isinstance(display.figure_, mpl.figure.Figure)
    # 断言display.figure_是mpl.figure.Figure的实例

    if plot_chance_level:
        assert isinstance(display.chance_level_, mpl.lines.Line2D)
        # 如果plot_chance_level为True，断言display.chance_level_是mpl.lines.Line2D的实例
        assert tuple(display.chance_level_.get_xdata()) == (0, 1)
        # 断言display.chance_level_的x坐标数据为(0, 1)
        assert tuple(display.chance_level_.get_ydata()) == (0, 1)
        # 断言display.chance_level_的y坐标数据为(0, 1)
    else:
        assert display.chance_level_ is None
        # 如果plot_chance_level为False，断言display.chance_level_为None

    # Checking for chance level line styles
    # 检查Chance Level线的样式
    if plot_chance_level and chance_level_kw is None:
        assert display.chance_level_.get_color() == "k"
        # 如果plot_chance_level为True且chance_level_kw为None，断言display.chance_level_的颜色为"k"
        assert display.chance_level_.get_linestyle() == "--"
        # 断言display.chance_level_的线型为"--"
        assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
        # 断言display.chance_level_的标签为"Chance level (AUC = 0.5)"
    elif plot_chance_level:
        assert display.chance_level_.get_label() == chance_level_kw["label"]
        # 如果plot_chance_level为True，断言display.chance_level_的标签与chance_level_kw["label"]相符
        assert display.chance_level_.get_color() == chance_level_kw["color"]
        # 断言display.chance_level_的颜色与chance_level_kw["color"]相符
        assert display.chance_level_.get_linewidth() == chance_level_kw["linewidth"]
        # 断言display.chance_level_的线宽与chance_level_kw["linewidth"]相符


@pytest.mark.parametrize(
    "clf",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression()
        ),
    ],
)
# 使用pytest的@parametrize装饰器，为clf参数分别测试逻辑回归模型和具有复杂流水线的Pipeline模型

@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
# 使用pytest的@parametrize装饰器，为constructor_name参数分别测试"from_estimator"和"from_predictions"两种情况
def test_roc_curve_display_complex_pipeline(pyplot, data_binary, clf, constructor_name):
    """Check the behaviour with complex pipeline."""
    # 检查复杂流水线的行为

    X, y = data_binary
    # 解包二进制数据

    if constructor_name == "from_estimator":
        with pytest.raises(NotFittedError):
            RocCurveDisplay.from_estimator(clf, X, y)
        # 如果constructor_name为"from_estimator"，断言调用RocCurveDisplay.from_estimator会引发NotFittedError异常

    clf.fit(X, y)
    # 使用给定的数据拟合模型
    # 如果构造器名称是 "from_estimator"，则使用分类器 clf 和数据集 X、y 创建 ROC 曲线显示对象
    if constructor_name == "from_estimator":
        display = RocCurveDisplay.from_estimator(clf, X, y)
        # 获取分类器的类名作为名称
        name = clf.__class__.__name__
    # 否则，使用真实标签 y 和预测标签 y 创建 ROC 曲线显示对象
    else:
        display = RocCurveDisplay.from_predictions(y, y)
        # 设置默认名称为 "Classifier"
        name = "Classifier"
    
    # 断言名称 name 存在于 ROC 曲线显示对象的线条标签中
    assert name in display.line_.get_label()
    # 断言 ROC 曲线显示对象的估计器名称与名称 name 相等
    assert display.estimator_name == name
@pytest.mark.parametrize(
    "roc_auc, estimator_name, expected_label",
    [
        (0.9, None, "AUC = 0.90"),  # 参数化测试数据: ROC AUC为0.9，估计器名称为None，期望标签为"AUC = 0.90"
        (None, "my_est", "my_est"),  # 参数化测试数据: ROC AUC为None，估计器名称为"my_est"，期望标签为"my_est"
        (0.8, "my_est2", "my_est2 (AUC = 0.80)"),  # 参数化测试数据: ROC AUC为0.8，估计器名称为"my_est2"，期望标签为"my_est2 (AUC = 0.80)"
    ],
)
def test_roc_curve_display_default_labels(
    pyplot, roc_auc, estimator_name, expected_label
):
    """Check the default labels used in the display."""
    fpr = np.array([0, 0.5, 1])  # 假阳率数组
    tpr = np.array([0, 0.5, 1])  # 真阳率数组
    disp = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=estimator_name
    ).plot()  # 创建ROC曲线显示对象并绘制
    assert disp.line_.get_label() == expected_label  # 断言曲线显示对象的标签与期望标签一致


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_plot_roc_curve_pos_label(pyplot, response_method, constructor_name):
    """Check the plotting of ROC curve with positive label handling."""
    # check that we can provide the positive label and display the proper
    # statistics
    X, y = load_breast_cancer(return_X_y=True)  # 加载乳腺癌数据集
    # create an highly imbalanced
    idx_positive = np.flatnonzero(y == 1)  # 获取正类索引
    idx_negative = np.flatnonzero(y == 0)  # 获取负类索引
    idx_selected = np.hstack([idx_negative, idx_positive[:25]])  # 选择部分样本，保持高度不平衡
    X, y = X[idx_selected], y[idx_selected]  # 根据选择的索引重新划分数据集
    X, y = shuffle(X, y, random_state=42)  # 打乱数据集顺序
    # only use 2 features to make the problem even harder
    X = X[:, :2]  # 仅使用前两个特征列进行训练
    y = np.array(["cancer" if c == 1 else "not cancer" for c in y], dtype=object)  # 将标签转换为"cancer"或"not cancer"
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        random_state=0,
    )  # 划分训练集和测试集

    classifier = LogisticRegression()  # 创建逻辑回归分类器
    classifier.fit(X_train, y_train)  # 使用训练数据拟合分类器

    # sanity check to be sure the positive class is classes_[0] and that we
    # are betrayed by the class imbalance
    assert classifier.classes_.tolist() == ["cancer", "not cancer"]  # 断言分类器的类别列表顺序为["cancer", "not cancer"]

    y_pred = getattr(classifier, response_method)(X_test)  # 根据response_method预测测试数据的标签或概率
    # we select the corresponding probability columns or reverse the decision
    # function otherwise
    y_pred_cancer = -1 * y_pred if y_pred.ndim == 1 else y_pred[:, 0]  # 选择与"cancer"类别相关的概率列或反转决策函数
    y_pred_not_cancer = y_pred if y_pred.ndim == 1 else y_pred[:, 1]  # 选择与"not cancer"类别相关的概率列

    if constructor_name == "from_estimator":
        display = RocCurveDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            pos_label="cancer",
            response_method=response_method,
        )  # 从估计器创建ROC曲线显示对象，指定正类标签为"cancer"
    else:
        display = RocCurveDisplay.from_predictions(
            y_test,
            y_pred_cancer,
            pos_label="cancer",
        )  # 从预测结果创建ROC曲线显示对象，指定正类标签为"cancer"

    roc_auc_limit = 0.95679  # 预期的ROC AUC值的上限

    assert display.roc_auc == pytest.approx(roc_auc_limit)  # 断言显示对象的ROC AUC近似等于预期的上限值
    assert trapezoid(display.tpr, display.fpr) == pytest.approx(roc_auc_limit)  # 断言根据TPR和FPR计算的面积近似等于预期的上限值

    if constructor_name == "from_estimator":
        display = RocCurveDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            response_method=response_method,
            pos_label="not cancer",
        )  # 从估计器创建ROC曲线显示对象，指定正类标签为"not cancer"
    else:
        # 如果不是癌症预测，则使用 RocCurveDisplay 类从预测结果生成 ROC 曲线显示对象
        display = RocCurveDisplay.from_predictions(
            y_test,
            y_pred_not_cancer,
            pos_label="not cancer",
        )

    # 断言 ROC 曲线的 AUC 值与预期的 roc_auc_limit 接近
    assert display.roc_auc == pytest.approx(roc_auc_limit)
    # 使用 trapezoid 函数计算得到的 ROC 曲线下的面积（AUC），并断言其与 roc_auc_limit 接近
    assert trapezoid(display.tpr, display.fpr) == pytest.approx(roc_auc_limit)
```