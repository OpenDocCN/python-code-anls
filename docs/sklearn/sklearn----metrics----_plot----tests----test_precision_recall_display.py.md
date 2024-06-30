# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\tests\test_precision_recall_display.py`

```
# 导入必要的库和模块
from collections import Counter
import numpy as np
import pytest

# 导入 sklearn 相关模块和函数
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import trapezoid

# 忽略特定警告，直到问题解决 https://github.com/numpy/numpy/issues/14397
pytestmark = pytest.mark.filterwarnings(
    "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
    "matplotlib.*"
)

# 使用 pytest 的 parametrize 装饰器进行多参数化测试
@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
@pytest.mark.parametrize("drop_intermediate", [True, False])
def test_precision_recall_display_plotting(
    pyplot, constructor_name, response_method, drop_intermediate
):
    """Check the overall plotting rendering."""
    # 创建一个二分类数据集 X, y
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    pos_label = 1

    # 训练逻辑回归模型
    classifier = LogisticRegression().fit(X, y)
    classifier.fit(X, y)

    # 根据不同的 response_method 调用模型的方法生成预测结果
    y_pred = getattr(classifier, response_method)(X)
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, pos_label]

    # 断言 constructor_name 必须是指定的两个字符串之一
    assert constructor_name in ("from_estimator", "from_predictions")

    # 根据 constructor_name 创建 PrecisionRecallDisplay 对象
    if constructor_name == "from_estimator":
        display = PrecisionRecallDisplay.from_estimator(
            classifier,
            X,
            y,
            response_method=response_method,
            drop_intermediate=drop_intermediate,
        )
    else:
        display = PrecisionRecallDisplay.from_predictions(
            y, y_pred, pos_label=pos_label, drop_intermediate=drop_intermediate
        )

    # 计算精确率-召回率曲线
    precision, recall, _ = precision_recall_curve(
        y, y_pred, pos_label=pos_label, drop_intermediate=drop_intermediate
    )
    # 计算平均精确率
    average_precision = average_precision_score(y, y_pred, pos_label=pos_label)

    # 使用 np.testing 断言检查精确率和召回率的计算结果
    np.testing.assert_allclose(display.precision, precision)
    np.testing.assert_allclose(display.recall, recall)
    # 使用 pytest.approx 断言检查平均精确率的计算结果
    assert display.average_precision == pytest.approx(average_precision)

    # 导入 matplotlib 并进行类型检查
    import matplotlib as mpl

    assert isinstance(display.line_, mpl.lines.Line2D)
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert isinstance(display.figure_, mpl.figure.Figure)

    # 检查图形对象的标签和调整参数
    assert display.ax_.get_xlabel() == "Recall (Positive label: 1)"
    assert display.ax_.get_ylabel() == "Precision (Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    # 调用 plot 方法进行绘图，传入 alpha 参数为 0.8，name 参数为 "MySpecialEstimator"
    display.plot(alpha=0.8, name="MySpecialEstimator")

    # 生成期望的标签字符串，包含特定格式化的平均精度值
    expected_label = f"MySpecialEstimator (AP = {average_precision:0.2f})"

    # 断言绘图对象的线条标签与期望的标签字符串相等
    assert display.line_.get_label() == expected_label

    # 断言绘图对象的线条透明度接近于 0.8（使用 pytest 的 approx 方法进行近似比较）
    assert display.line_.get_alpha() == pytest.approx(0.8)

    # 断言检查默认情况下不绘制机会水平线
    assert display.chance_level_ is None
# 使用 pytest 的 parametrize 装饰器，为 chance_level_kw 参数提供两种情况的测试：None 和 {"color": "r"}
# 为 constructor_name 参数提供两种情况的测试："from_estimator" 和 "from_predictions"
@pytest.mark.parametrize("chance_level_kw", [None, {"color": "r"}])
@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_precision_recall_chance_level_line(
    pyplot,  # 传入 pyplot 对象作为测试函数的参数
    chance_level_kw,  # 测试用的 chance_level_kw 参数
    constructor_name,  # 测试用的 constructor_name 参数
):
    """Check the chance level line plotting behavior."""
    # 生成一个二分类的样本集 X 和对应的标签 y
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    # 计算正类的样本比例
    pos_prevalence = Counter(y)[1] / len(y)

    # 初始化逻辑回归模型
    lr = LogisticRegression()
    # 训练模型并预测类别概率
    y_pred = lr.fit(X, y).predict_proba(X)[:, 1]

    # 根据 constructor_name 的不同选择不同的 PrecisionRecallDisplay 实例化方法
    if constructor_name == "from_estimator":
        display = PrecisionRecallDisplay.from_estimator(
            lr,
            X,
            y,
            plot_chance_level=True,
            chance_level_kw=chance_level_kw,
        )
    else:
        display = PrecisionRecallDisplay.from_predictions(
            y,
            y_pred,
            plot_chance_level=True,
            chance_level_kw=chance_level_kw,
        )

    import matplotlib as mpl  # 导入 matplotlib 库，用于绘图操作

    # 断言 display.chance_level_ 是 matplotlib 的 Line2D 对象
    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    # 断言 display.chance_level_ 的 x 数据是 (0, 1)
    assert tuple(display.chance_level_.get_xdata()) == (0, 1)
    # 断言 display.chance_level_ 的 y 数据是 (pos_prevalence, pos_prevalence)
    assert tuple(display.chance_level_.get_ydata()) == (pos_prevalence, pos_prevalence)

    # 检查 chance level 线条的样式
    if chance_level_kw is None:
        assert display.chance_level_.get_color() == "k"  # 如果 chance_level_kw 是 None，则颜色应为黑色
    else:
        assert display.chance_level_.get_color() == "r"  # 否则颜色应为红色


# 使用 pytest 的 parametrize 装饰器，为 constructor_name 和 default_label 参数提供两种情况的测试
@pytest.mark.parametrize(
    "constructor_name, default_label",
    [
        ("from_estimator", "LogisticRegression (AP = {:.2f})"),
        ("from_predictions", "Classifier (AP = {:.2f})"),
    ],
)
def test_precision_recall_display_name(pyplot, constructor_name, default_label):
    """Check the behaviour of the name parameters"""
    # 生成一个二分类的样本集 X 和对应的标签 y
    X, y = make_classification(n_classes=2, n_samples=100, random_state=0)
    pos_label = 1

    # 初始化逻辑回归模型并拟合数据
    classifier = LogisticRegression().fit(X, y)
    classifier.fit(X, y)

    y_pred = classifier.predict_proba(X)[:, pos_label]

    # 断言 constructor_name 在合法的选项中
    assert constructor_name in ("from_estimator", "from_predictions")

    # 根据 constructor_name 的不同选择不同的 PrecisionRecallDisplay 实例化方法
    if constructor_name == "from_estimator":
        display = PrecisionRecallDisplay.from_estimator(classifier, X, y)
    else:
        display = PrecisionRecallDisplay.from_predictions(
            y, y_pred, pos_label=pos_label
        )

    # 计算平均精度
    average_precision = average_precision_score(y, y_pred, pos_label=pos_label)

    # 断言使用默认标签
    assert display.line_.get_label() == default_label.format(average_precision)

    # 设置名称后再次断言
    display.plot(name="MySpecialEstimator")
    assert (
        display.line_.get_label()
        == f"MySpecialEstimator (AP = {average_precision:.2f})"
    )


# 使用 pytest 的 parametrize 装饰器，为 clf 参数提供两种不同的机器学习模型进行测试
@pytest.mark.parametrize(
    "clf",
    [
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression()
        ),
    ],
)
def test_precision_recall_display_pipeline(pyplot, clf):
    # 生成一个二分类的合成数据集，用于测试
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    # 测试在模型未拟合的情况下是否会引发 NotFittedError 异常
    with pytest.raises(NotFittedError):
        PrecisionRecallDisplay.from_estimator(clf, X, y)
    # 拟合分类器模型
    clf.fit(X, y)
    # 使用 PrecisionRecallDisplay 从已拟合的分类器和数据集生成展示对象
    display = PrecisionRecallDisplay.from_estimator(clf, X, y)
    # 断言展示对象的估计器名称与分类器类名相同
    assert display.estimator_name == clf.__class__.__name__


def test_precision_recall_display_string_labels(pyplot):
    # 回归测试 #15738
    # 加载乳腺癌数据集
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target_names[cancer.target]

    # 构建流水线，包括标准化和逻辑回归
    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(X, y)
    # 断言所有类别在逻辑回归的类列表中
    for klass in cancer.target_names:
        assert klass in lr.classes_
    # 使用 PrecisionRecallDisplay 从已拟合的分类器和数据集生成展示对象
    display = PrecisionRecallDisplay.from_estimator(lr, X, y)

    # 计算逻辑回归预测的平均精度
    y_pred = lr.predict_proba(X)[:, 1]
    avg_prec = average_precision_score(y, y_pred, pos_label=lr.classes_[1])

    # 断言展示对象的平均精度与预期的平均精度近似相等
    assert display.average_precision == pytest.approx(avg_prec)
    # 断言展示对象的估计器名称与逻辑回归类名相同
    assert display.estimator_name == lr.__class__.__name__

    # 预期的错误消息，指示 y_true 的取值应为 {'benign', 'malignant'}
    err_msg = r"y_true takes value in {'benign', 'malignant'}"
    # 测试从预测值生成 PrecisionRecallDisplay 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=err_msg):
        PrecisionRecallDisplay.from_predictions(y, y_pred)

    # 使用 PrecisionRecallDisplay 从预测值生成展示对象，指定正类标签
    display = PrecisionRecallDisplay.from_predictions(
        y, y_pred, pos_label=lr.classes_[1]
    )
    # 断言展示对象的平均精度与预期的平均精度近似相等
    assert display.average_precision == pytest.approx(avg_prec)


@pytest.mark.parametrize(
    "average_precision, estimator_name, expected_label",
    [
        (0.9, None, "AP = 0.90"),
        (None, "my_est", "my_est"),
        (0.8, "my_est2", "my_est2 (AP = 0.80)"),
    ],
)
def test_default_labels(pyplot, average_precision, estimator_name, expected_label):
    """Check the default labels used in the display."""
    # 构造精度和召回率数组
    precision = np.array([1, 0.5, 0])
    recall = np.array([0, 0.5, 1])
    # 使用给定的参数创建 PrecisionRecallDisplay 对象
    display = PrecisionRecallDisplay(
        precision,
        recall,
        average_precision=average_precision,
        estimator_name=estimator_name,
    )
    # 绘制展示对象
    display.plot()
    # 断言展示对象的线条标签与预期标签相同
    assert display.line_.get_label() == expected_label


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_plot_precision_recall_pos_label(pyplot, constructor_name, response_method):
    # 检查是否能提供正类标签，并显示正确的统计信息
    # 加载乳腺癌数据集
    X, y = load_breast_cancer(return_X_y=True)
    # 创建一个高度不平衡的乳腺癌数据集版本
    idx_positive = np.flatnonzero(y == 1)
    idx_negative = np.flatnonzero(y == 0)
    idx_selected = np.hstack([idx_negative, idx_positive[:25]])
    X, y = X[idx_selected], y[idx_selected]
    X, y = shuffle(X, y, random_state=42)
    # 只使用前两个特征使问题更加困难
    X = X[:, :2]
    y = np.array(["cancer" if c == 1 else "not cancer" for c in y], dtype=object)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        random_state=0,
    )

    # 创建一个逻辑回归分类器对象
    classifier = LogisticRegression()
    # 使用训练数据 X_train 和标签 y_train 来拟合分类器
    classifier.fit(X_train, y_train)

    # 对分类器的类别顺序进行断言，确保正类是 classes_[0]，并检查类别不平衡问题
    assert classifier.classes_.tolist() == ["cancer", "not cancer"]

    # 根据指定的响应方法调用分类器预测函数，得到预测结果 y_pred
    y_pred = getattr(classifier, response_method)(X_test)
    # 如果 y_pred 是一维数组，则 y_pred_cancer 是 y_pred 的负值；否则是第一列数据
    y_pred_cancer = -1 * y_pred if y_pred.ndim == 1 else y_pred[:, 0]
    # 如果 y_pred 是一维数组，则 y_pred_not_cancer 是 y_pred；否则是第二列数据
    y_pred_not_cancer = y_pred if y_pred.ndim == 1 else y_pred[:, 1]

    # 根据构造函数名称选择不同的 Precision-Recall 显示对象
    if constructor_name == "from_estimator":
        # 从分类器创建 Precision-Recall 显示对象，用于评估模型在测试数据上的性能
        display = PrecisionRecallDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            pos_label="cancer",
            response_method=response_method,
        )
    else:
        # 从预测结果创建 Precision-Recall 显示对象，用于评估模型在测试数据上的性能
        display = PrecisionRecallDisplay.from_predictions(
            y_test,
            y_pred_cancer,
            pos_label="cancer",
        )
    # 断言 Precision-Recall 曲线下的平均精度低于预设的阈值
    avg_prec_limit = 0.65
    assert display.average_precision < avg_prec_limit
    # 断言 Precision-Recall 曲线下的负面积低于预设的阈值
    assert -trapezoid(display.precision, display.recall) < avg_prec_limit

    # 否则，获取“非癌症”类别的 Precision-Recall 统计数据
    if constructor_name == "from_estimator":
        # 从分类器创建 Precision-Recall 显示对象，用于评估模型在测试数据上的性能
        display = PrecisionRecallDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            response_method=response_method,
            pos_label="not cancer",
        )
    else:
        # 从预测结果创建 Precision-Recall 显示对象，用于评估模型在测试数据上的性能
        display = PrecisionRecallDisplay.from_predictions(
            y_test,
            y_pred_not_cancer,
            pos_label="not cancer",
        )
    # 断言 Precision-Recall 曲线下的平均精度高于预设的阈值
    avg_prec_limit = 0.95
    assert display.average_precision > avg_prec_limit
    # 断言 Precision-Recall 曲线下的负面积高于预设的阈值
    assert -trapezoid(display.precision, display.recall) > avg_prec_limit
# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_precision_recall_prevalence_pos_label_reusable 提供参数化测试
@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_precision_recall_prevalence_pos_label_reusable(pyplot, constructor_name):
    # 检查即使在第一次调用时传递 plot_chance_level=False，
    # 仍然可以调用 disp.plot，并且通过 plot_chance_level=True 获得 chance level 线条
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    # 创建逻辑回归模型并预测概率
    lr = LogisticRegression()
    y_pred = lr.fit(X, y).predict_proba(X)[:, 1]

    # 根据 constructor_name 的值选择使用 PrecisionRecallDisplay.from_estimator 或 PrecisionRecallDisplay.from_predictions 创建 display 对象
    if constructor_name == "from_estimator":
        display = PrecisionRecallDisplay.from_estimator(
            lr, X, y, plot_chance_level=False
        )
    else:
        display = PrecisionRecallDisplay.from_predictions(
            y, y_pred, plot_chance_level=False
        )
    # 断言 display.chance_level_ 为 None
    assert display.chance_level_ is None

    import matplotlib as mpl  # 导入 matplotlib 库

    # 调用 display.plot 方法，传入 plot_chance_level=True
    # 验证 display.chance_level_ 是否为 mpl.lines.Line2D 类型的对象
    display.plot(plot_chance_level=True)
    assert isinstance(display.chance_level_, mpl.lines.Line2D)


# 定义测试函数 test_precision_recall_raise_no_prevalence
def test_precision_recall_raise_no_prevalence(pyplot):
    # 检查当没有提供 prevalence_pos_label 的情况下绘制 chance level 时是否会正确引发异常
    precision = np.array([1, 0.5, 0])
    recall = np.array([0, 0.5, 1])
    # 创建 PrecisionRecallDisplay 对象 display，只传入 precision 和 recall
    display = PrecisionRecallDisplay(precision, recall)

    # 定义异常消息
    msg = (
        "You must provide prevalence_pos_label when constructing the "
        "PrecisionRecallDisplay object in order to plot the chance "
        "level line. Alternatively, you may use "
        "PrecisionRecallDisplay.from_estimator or "
        "PrecisionRecallDisplay.from_predictions "
        "to automatically set prevalence_pos_label"
    )

    # 使用 pytest.raises 检查是否会引发 ValueError 异常，并匹配异常消息 msg
    with pytest.raises(ValueError, match=msg):
        display.plot(plot_chance_level=True)
```