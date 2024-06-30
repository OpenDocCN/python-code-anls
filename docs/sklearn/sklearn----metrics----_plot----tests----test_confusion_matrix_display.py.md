# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\tests\test_confusion_matrix_display.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 测试框架，用于单元测试
from numpy.testing import (  # 导入 NumPy 测试模块中的断言函数
    assert_allclose,  # 用于检查两个数组或数值的近似相等
    assert_array_equal,  # 用于检查两个数组是否完全相等
)

from sklearn.compose import make_column_transformer  # 导入数据预处理模块
from sklearn.datasets import make_classification  # 导入生成分类数据的函数
from sklearn.exceptions import NotFittedError  # 导入模型未拟合异常
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # 导入混淆矩阵相关类和函数
from sklearn.pipeline import make_pipeline  # 导入创建管道的函数
from sklearn.preprocessing import StandardScaler  # 导入数据标准化模块
from sklearn.svm import SVC, SVR  # 导入支持向量机分类和回归模型

# TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
# 设置 Pytest 标记，忽略指定的警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
    "matplotlib.*"
)


def test_confusion_matrix_display_validation(pyplot):
    """Check that we raise the proper error when validating parameters."""
    # 生成分类数据，用于测试
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=5, random_state=0
    )

    # 测试未拟合模型时，从估计器中生成混淆矩阵显示对象是否会引发 NotFittedError 异常
    with pytest.raises(NotFittedError):
        ConfusionMatrixDisplay.from_estimator(SVC(), X, y)

    # 训练 SVR 模型进行回归，用于后续测试
    regressor = SVR().fit(X, y)
    y_pred_regressor = regressor.predict(X)
    # 训练 SVC 模型进行分类，用于后续测试
    y_pred_classifier = SVC().fit(X, y).predict(X)

    # 测试使用回归器作为参数时，从估计器中生成混淆矩阵显示对象是否会引发 ValueError 异常
    err_msg = "ConfusionMatrixDisplay.from_estimator only supports classifiers"
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_estimator(regressor, X, y)

    # 测试混合类型的 y 参数时，从预测结果中生成混淆矩阵显示对象是否会引发 ValueError 异常
    err_msg = "Mix type of y not allowed, got types"
    with pytest.raises(ValueError, match=err_msg):
        # 强制将 `y_true` 视为回归问题
        ConfusionMatrixDisplay.from_predictions(y + 0.5, y_pred_classifier)
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_predictions(y, y_pred_regressor)

    # 测试输入变量样本数不一致时，从预测结果中生成混淆矩阵显示对象是否会引发 ValueError 异常
    err_msg = "Found input variables with inconsistent numbers of samples"
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_predictions(y, y_pred_classifier[::2])


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("with_labels", [True, False])
@pytest.mark.parametrize("with_display_labels", [True, False])
def test_confusion_matrix_display_custom_labels(
    pyplot, constructor_name, with_labels, with_display_labels
):
    """Check the resulting plot when labels are given."""
    n_classes = 5
    # 生成分类数据，用于测试
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    # 训练分类器模型
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)

    # 检查是否在二进制 if/else 结构中安全地使用构造函数名称
    assert constructor_name in ("from_estimator", "from_predictions")

    # 获取当前坐标轴
    ax = pyplot.gca()
    # 根据条件设置标签和显示标签
    labels = [2, 1, 0, 3, 4] if with_labels else None
    display_labels = ["b", "d", "a", "e", "f"] if with_display_labels else None

    # 计算混淆矩阵
    cm = confusion_matrix(y, y_pred, labels=labels)
    # 共同的参数
    common_kwargs = {
        "ax": ax,
        "display_labels": display_labels,
        "labels": labels,
    }
    # 如果构造函数名称为 "from_estimator"，则使用分类器、输入特征 X 和目标标签 y 创建混淆矩阵显示对象
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        # 否则，使用预测结果 y_pred 创建混淆矩阵显示对象
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)
    
    # 确保显示对象的混淆矩阵与预期的混淆矩阵 cm 非常接近
    assert_allclose(disp.confusion_matrix, cm)

    # 根据参数选择显示的标签
    if with_display_labels:
        expected_display_labels = display_labels
    elif with_labels:
        expected_display_labels = labels
    else:
        expected_display_labels = list(range(n_classes))

    # 将预期显示的标签转换为字符串形式
    expected_display_labels_str = [str(name) for name in expected_display_labels]

    # 获取显示对象的 x 轴和 y 轴的刻度标签文本
    x_ticks = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    y_ticks = [tick.get_text() for tick in disp.ax_.get_yticklabels()]

    # 确保混淆矩阵显示对象的 display_labels 与预期的显示标签一致
    assert_array_equal(disp.display_labels, expected_display_labels)
    # 确保 x 轴的刻度标签与预期的显示标签字符串数组一致
    assert_array_equal(x_ticks, expected_display_labels_str)
    # 确保 y 轴的刻度标签与预期的显示标签字符串数组一致
    assert_array_equal(y_ticks, expected_display_labels_str)
@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
@pytest.mark.parametrize("include_values", [True, False])
def test_confusion_matrix_display_plotting(
    pyplot,
    constructor_name,
    normalize,
    include_values,
):
    """Check the overall plotting rendering."""
    # 生成一个具有指定参数化参数的测试函数，用于测试混淆矩阵的显示和绘制
    n_classes = 5
    # 生成一个具有指定特征的分类样本集
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    # 使用支持向量机对样本集进行拟合
    classifier = SVC().fit(X, y)
    # 对样本集进行预测
    y_pred = classifier.predict(X)

    # 对二进制的 if/else 结构进行安全保护
    assert constructor_name in ("from_estimator", "from_predictions")

    # 获取当前图形的坐标轴
    ax = pyplot.gca()
    # 指定颜色映射
    cmap = "plasma"

    # 计算混淆矩阵
    cm = confusion_matrix(y, y_pred)
    # 定义常见参数字典
    common_kwargs = {
        "normalize": normalize,
        "cmap": cmap,
        "ax": ax,
        "include_values": include_values,
    }
    # 根据构造函数的名称选择不同的混淆矩阵显示方法
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)

    # 断言显示对象的坐标轴与当前坐标轴一致
    assert disp.ax_ == ax

    # 根据不同的归一化方式调整混淆矩阵的值
    if normalize == "true":
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif normalize == "all":
        cm = cm / cm.sum()

    # 断言混淆矩阵显示对象的混淆矩阵与调整后的混淆矩阵近似相等
    assert_allclose(disp.confusion_matrix, cm)
    import matplotlib as mpl

    # 断言显示对象的图像部分是 Matplotlib 的 AxesImage 类型
    assert isinstance(disp.im_, mpl.image.AxesImage)
    # 断言显示对象的颜色映射名称与指定的颜色映射一致
    assert disp.im_.get_cmap().name == cmap
    # 断言显示对象的坐标轴是 Matplotlib 的 Axes 类型
    assert isinstance(disp.ax_, pyplot.Axes)
    # 断言显示对象的图形是 Matplotlib 的 Figure 类型
    assert isinstance(disp.figure_, pyplot.Figure)

    # 断言显示对象的 y 轴标签为 "True label"
    assert disp.ax_.get_ylabel() == "True label"
    # 断言显示对象的 x 轴标签为 "Predicted label"

    # 获取显示对象的 x 轴刻度标签文本，并转换为列表
    x_ticks = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    # 获取显示对象的 y 轴刻度标签文本，并转换为列表
    y_ticks = [tick.get_text() for tick in disp.ax_.get_yticklabels()]

    # 期望的显示标签为 0 到 n_classes-1 的列表
    expected_display_labels = list(range(n_classes))

    # 将期望的显示标签转换为字符串列表
    expected_display_labels_str = [str(name) for name in expected_display_labels]

    # 断言显示对象的显示标签与期望的显示标签数组相等
    assert_array_equal(disp.display_labels, expected_display_labels)
    # 断言 x 轴刻度标签与期望的显示标签字符串数组相等
    assert_array_equal(x_ticks, expected_display_labels_str)
    # 断言 y 轴刻度标签与期望的显示标签字符串数组相等
    assert_array_equal(y_ticks, expected_display_labels_str)

    # 获取显示对象图像数据的数组
    image_data = disp.im_.get_array().data
    # 断言显示对象图像数据与调整后的混淆矩阵数据近似相等
    assert_allclose(image_data, cm)

    # 如果包含值，则断言显示对象的文本数组形状为 (n_classes, n_classes)
    if include_values:
        assert disp.text_.shape == (n_classes, n_classes)
        # 设置格式字符串
        fmt = ".2g"
        # 期望的文本数组为格式化后的调整后的混淆矩阵数据
        expected_text = np.array([format(v, fmt) for v in cm.ravel(order="C")])
        # 获取显示对象的文本数组，并转换为字符串数组
        text_text = np.array([t.get_text() for t in disp.text_.ravel(order="C")])
        # 断言期望的文本数组与显示对象的文本数组相等
        assert_array_equal(expected_text, text_text)
    else:
        # 如果不包含值，则断言显示对象的文本数组为空
        assert disp.text_ is None


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_confusion_matrix_display(pyplot, constructor_name):
    """Check the behaviour of the default constructor without using the class
    methods."""
    # 测试默认构造函数的行为，不使用类方法
    n_classes = 5
    # 使用 make_classification 生成具有指定特征的分类数据集
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    # 使用支持向量机 (SVC) 对数据集进行拟合
    classifier = SVC().fit(X, y)
    # 对训练集 X 进行预测
    y_pred = classifier.predict(X)

    # 检查构造函数名称是否符合预期值，用于二进制的 if/else 结构的保护
    assert constructor_name in ("from_estimator", "from_predictions")

    # 计算真实标签 y 和预测标签 y_pred 的混淆矩阵
    cm = confusion_matrix(y, y_pred)
    # 定义常见参数字典
    common_kwargs = {
        "normalize": None,
        "include_values": True,
        "cmap": "viridis",
        "xticks_rotation": 45.0,
    }
    # 根据 constructor_name 的值选择不同的 ConfusionMatrixDisplay 方法
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)

    # 检查显示的混淆矩阵是否与计算的混淆矩阵非常接近
    assert_allclose(disp.confusion_matrix, cm)
    # 检查显示文本的形状是否符合预期的类别数量
    assert disp.text_.shape == (n_classes, n_classes)

    # 获取显示的 x 轴刻度标签的旋转角度，并检查是否接近 45.0 度
    rotations = [tick.get_rotation() for tick in disp.ax_.get_xticklabels()]
    assert_allclose(rotations, 45.0)

    # 获取显示的图像数据，并检查与混淆矩阵的数值是否非常接近
    image_data = disp.im_.get_array().data
    assert_allclose(image_data, cm)

    # 使用不同的 colormap 绘制混淆矩阵，并检查使用的 colormap 是否为 "plasma"
    disp.plot(cmap="plasma")
    assert disp.im_.get_cmap().name == "plasma"

    # 绘制混淆矩阵时不包括数值，并检查文本显示是否为 None
    disp.plot(include_values=False)
    assert disp.text_ is None

    # 绘制混淆矩阵时旋转 x 轴刻度标签为 90.0 度，并检查旋转角度是否符合预期
    disp.plot(xticks_rotation=90.0)
    rotations = [tick.get_rotation() for tick in disp.ax_.get_xticklabels()]
    assert_allclose(rotations, 90.0)

    # 绘制混淆矩阵时使用科学计数法格式化数值文本，并检查生成的文本数组是否符合预期
    disp.plot(values_format="e")
    expected_text = np.array([format(v, "e") for v in cm.ravel(order="C")])
    text_text = np.array([t.get_text() for t in disp.text_.ravel(order="C")])
    assert_array_equal(expected_text, text_text)
# 定义测试函数，用于测试混淆矩阵的对比
def test_confusion_matrix_contrast(pyplot):
    """Check that the text color is appropriate depending on background."""
    
    # 创建一个2x2的单位矩阵，并对每个元素除以2，得到混淆矩阵
    cm = np.eye(2) / 2
    
    # 创建一个混淆矩阵显示对象，指定显示标签为[0, 1]
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    
    # 绘制混淆矩阵，使用灰度色图
    disp.plot(cmap=pyplot.cm.gray)
    # 断言：对角线文本颜色为黑色
    assert_allclose(disp.text_[0, 0].get_color(), [0.0, 0.0, 0.0, 1.0])
    assert_allclose(disp.text_[1, 1].get_color(), [0.0, 0.0, 0.0, 1.0])
    
    # 断言：非对角线文本颜色为白色
    assert_allclose(disp.text_[0, 1].get_color(), [1.0, 1.0, 1.0, 1.0])
    assert_allclose(disp.text_[1, 0].get_color(), [1.0, 1.0, 1.0, 1.0])
    
    # 绘制混淆矩阵，使用反向灰度色图
    disp.plot(cmap=pyplot.cm.gray_r)
    # 断言：对角线文本颜色为白色
    assert_allclose(disp.text_[0, 1].get_color(), [0.0, 0.0, 0.0, 1.0])
    assert_allclose(disp.text_[1, 0].get_color(), [0.0, 0.0, 0.0, 1.0])
    
    # 断言：非对角线文本颜色为黑色
    assert_allclose(disp.text_[0, 0].get_color(), [1.0, 1.0, 1.0, 1.0])
    assert_allclose(disp.text_[1, 1].get_color(), [1.0, 1.0, 1.0, 1.0])
    
    # 回归测试 #15920
    # 创建一个具体的混淆矩阵，进行显示
    cm = np.array([[19, 34], [32, 58]])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    
    # 绘制混淆矩阵，使用蓝色色图
    disp.plot(cmap=pyplot.cm.Blues)
    min_color = pyplot.cm.Blues(0)
    max_color = pyplot.cm.Blues(255)
    # 断言：文本颜色与最大和最小颜色匹配
    assert_allclose(disp.text_[0, 0].get_color(), max_color)
    assert_allclose(disp.text_[0, 1].get_color(), max_color)
    assert_allclose(disp.text_[1, 0].get_color(), max_color)
    assert_allclose(disp.text_[1, 1].get_color(), min_color)


# 参数化装饰器，用于测试复杂流水线的混淆矩阵行为
@pytest.mark.parametrize(
    "clf",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])),
            LogisticRegression(),
        ),
    ],
    ids=["clf", "pipeline-clf", "pipeline-column_transformer-clf"],
)
def test_confusion_matrix_pipeline(pyplot, clf):
    """Check the behaviour of the plotting with more complex pipeline."""
    
    n_classes = 5
    # 生成具有5个类别的分类数据集
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    
    # 使用断言检查未拟合错误
    with pytest.raises(NotFittedError):
        ConfusionMatrixDisplay.from_estimator(clf, X, y)
    
    # 拟合分类器
    clf.fit(X, y)
    y_pred = clf.predict(X)
    
    # 从估计器创建混淆矩阵显示对象
    disp = ConfusionMatrixDisplay.from_estimator(clf, X, y)
    cm = confusion_matrix(y, y_pred)
    
    # 断言：混淆矩阵的值与预期一致
    assert_allclose(disp.confusion_matrix, cm)
    # 断言：文本数组的形状应为(n_classes, n_classes)
    assert disp.text_.shape == (n_classes, n_classes)


# 参数化装饰器，用于测试未知标签时的混淆矩阵行为
@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_confusion_matrix_with_unknown_labels(pyplot, constructor_name):
    """Check that when labels=None, the unique values in `y_pred` and `y_true`
    will be used.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/18405
    """
    
    n_classes = 5
    # 生成具有5个类别的分类数据集
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    
    # 训练支持向量分类器
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)
    # 将 `y_true` 中未在拟合过程中出现且不在 'classifier.classes_' 中的标签加一处理
    y = y + 1

    # 断言，确保 constructor_name 只能是 "from_estimator" 或 "from_predictions" 中的一个
    assert constructor_name in ("from_estimator", "from_predictions")

    # 设置公共参数字典，初始值 labels 为 None
    common_kwargs = {"labels": None}

    # 根据 constructor_name 的不同选择不同的显示方式
    if constructor_name == "from_estimator":
        # 从分类器和数据中创建混淆矩阵显示对象
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        # 根据预测结果创建混淆矩阵显示对象
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)

    # 获取显示对象的 x 轴刻度标签文本
    display_labels = [tick.get_text() for tick in disp.ax_.get_xticklabels()]

    # 生成预期的标签列表，包括数字 0 到 n_classes
    expected_labels = [str(i) for i in range(n_classes + 1)]

    # 断言，确保显示的标签与预期的标签一致
    assert_array_equal(expected_labels, display_labels)
# 检查颜色映射中的最大颜色是否用于文本的颜色
def test_colormap_max(pyplot):
    # 获取灰度颜色映射对象，分成 1024 个不同的颜色
    gray = pyplot.get_cmap("gray", 1024)
    # 创建一个2x2的混淆矩阵，对角线上为1.0，其它位置为0.0
    confusion_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])

    # 创建混淆矩阵的展示对象
    disp = ConfusionMatrixDisplay(confusion_matrix)
    # 使用灰度颜色映射绘制混淆矩阵
    disp.plot(cmap=gray)

    # 获取混淆矩阵中特定位置的文本对象的颜色
    color = disp.text_[1, 0].get_color()
    # 断言文本的颜色接近 [1.0, 1.0, 1.0, 1.0]（白色）
    assert_allclose(color, [1.0, 1.0, 1.0, 1.0])


# 检查imshow函数是否正确传递了im_kw参数
def test_im_kw_adjust_vmin_vmax(pyplot):
    # 创建一个2x2的混淆矩阵，元素值介于0.0到0.8之间
    confusion_matrix = np.array([[0.48, 0.04], [0.08, 0.4]])
    # 创建混淆矩阵的展示对象
    disp = ConfusionMatrixDisplay(confusion_matrix)
    # 使用im_kw参数设置imshow函数的vmin和vmax值
    disp.plot(im_kw=dict(vmin=0.0, vmax=0.8))

    # 获取imshow对象的颜色限制范围
    clim = disp.im_.get_clim()
    # 断言颜色限制的下限接近0.0
    assert clim[0] == pytest.approx(0.0)
    # 断言颜色限制的上限接近0.8
    assert clim[1] == pytest.approx(0.8)


# 检查text_kw参数是否正确传递给text函数调用
def test_confusion_matrix_text_kw(pyplot):
    # 设置字体大小为15.0
    font_size = 15.0
    # 创建一个随机的分类数据集
    X, y = make_classification(random_state=0)
    # 使用SVC分类器拟合数据
    classifier = SVC().fit(X, y)

    # 使用分类器创建混淆矩阵展示对象，并传递text_kw参数设置字体大小
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier, X, y, text_kw={"fontsize": font_size}
    )
    # 遍历所有文本对象，断言它们的字体大小与预期相符
    for text in disp.text_.reshape(-1):
        assert text.get_fontsize() == font_size

    # 使用text_kw参数设置新的字体大小，并重新绘制混淆矩阵
    new_font_size = 20.0
    disp.plot(text_kw={"fontsize": new_font_size})
    # 遍历所有文本对象，断言它们的字体大小已更新为新的字体大小
    for text in disp.text_.reshape(-1):
        assert text.get_fontsize() == new_font_size

    # 使用预测结果创建混淆矩阵展示对象，并传递text_kw参数设置字体大小
    y_pred = classifier.predict(X)
    disp = ConfusionMatrixDisplay.from_predictions(
        y, y_pred, text_kw={"fontsize": font_size}
    )
    # 遍历所有文本对象，断言它们的字体大小与预期相符
    for text in disp.text_.reshape(-1):
        assert text.get_fontsize() == font_size
```