# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\tests\test_plot.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

from sklearn.datasets import load_iris  # 导入 load_iris 函数，用于加载鸢尾花数据集
from sklearn.model_selection import (  # 导入交叉验证相关函数和类
    LearningCurveDisplay,  # 学习曲线显示类
    ValidationCurveDisplay,  # 验证曲线显示类
    learning_curve,  # 计算学习曲线函数
    validation_curve,  # 计算验证曲线函数
)
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.utils import shuffle  # 导入 shuffle 函数，用于数据集洗牌
from sklearn.utils._testing import assert_allclose, assert_array_equal  # 导入断言函数，用于测试数组相等性


@pytest.fixture
def data():
    # 返回经过洗牌后的鸢尾花数据集
    return shuffle(*load_iris(return_X_y=True), random_state=0)


@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        ({"std_display_style": "invalid"}, ValueError, "Unknown std_display_style:"),  # 参数化测试，测试不合法的参数值
        ({"score_type": "invalid"}, ValueError, "Unknown score_type:"),  # 参数化测试，测试不合法的参数值
    ],
)
@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),  # 参数化测试，验证曲线显示类的参数
        (LearningCurveDisplay, {"train_sizes": [0.3, 0.6, 0.9]}),  # 参数化测试，学习曲线显示类的参数
    ],
)
def test_curve_display_parameters_validation(
    pyplot, data, params, err_type, err_msg, CurveDisplay, specific_params
):
    """Check that we raise a proper error when passing invalid parameters."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    # 使用 pytest.raises 检测是否抛出预期异常
    with pytest.raises(err_type, match=err_msg):
        CurveDisplay.from_estimator(estimator, X, y, **specific_params, **params)


def test_learning_curve_display_default_usage(pyplot, data):
    """Check the default usage of the LearningCurveDisplay class."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    train_sizes = [0.3, 0.6, 0.9]
    display = LearningCurveDisplay.from_estimator(
        estimator, X, y, train_sizes=train_sizes
    )

    import matplotlib as mpl

    # 断言显示误差条为空
    assert display.errorbar_ is None

    # 断言 display.lines_ 是一个列表，并且每个元素是 Line2D 对象
    assert isinstance(display.lines_, list)
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)

    # 断言 display.fill_between_ 是一个列表，并且每个元素是 PolyCollection 对象，alpha 值为 0.5
    assert isinstance(display.fill_between_, list)
    for fill in display.fill_between_:
        assert isinstance(fill, mpl.collections.PolyCollection)
        assert fill.get_alpha() == 0.5

    # 断言 score_name 属性为 "Score"
    assert display.score_name == "Score"
    # 断言 x 轴标签为 "Number of samples in the training set"
    assert display.ax_.get_xlabel() == "Number of samples in the training set"
    # 断言 y 轴标签为 "Score"
    assert display.ax_.get_ylabel() == "Score"

    # 获取图例标签，断言为 ["Train", "Test"]
    _, legend_labels = display.ax_.get_legend_handles_labels()
    assert legend_labels == ["Train", "Test"]

    # 计算学习曲线上的训练样本数、训练分数和测试分数，进行断言比较
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes
    )

    assert_array_equal(display.train_sizes, train_sizes_abs)  # 断言训练样本大小数组相等
    assert_allclose(display.train_scores, train_scores)  # 断言训练分数数组近似相等
    assert_allclose(display.test_scores, test_scores)  # 断言测试分数数组近似相等


def test_validation_curve_display_default_usage(pyplot, data):
    """Check the default usage of the ValidationCurveDisplay class."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    param_name, param_range = "max_depth", [1, 3, 5]
    # 使用给定的评估器、数据集和参数范围，创建一个验证曲线展示对象
    display = ValidationCurveDisplay.from_estimator(
        estimator, X, y, param_name=param_name, param_range=param_range
    )
    
    # 导入 matplotlib 库作为 mpl 别名
    import matplotlib as mpl
    
    # 断言展示对象中的 errorbar_ 属性为空
    assert display.errorbar_ is None
    
    # 断言展示对象中的 lines_ 属性是列表类型
    assert isinstance(display.lines_, list)
    for line in display.lines_:
        # 断言每个 line 对象是 matplotlib 中的 Line2D 类型
        assert isinstance(line, mpl.lines.Line2D)
    
    # 断言展示对象中的 fill_between_ 属性是列表类型
    assert isinstance(display.fill_between_, list)
    for fill in display.fill_between_:
        # 断言每个 fill 对象是 matplotlib 中的 PolyCollection 类型
        assert isinstance(fill, mpl.collections.PolyCollection)
        # 断言每个 fill 对象的透明度为 0.5
        assert fill.get_alpha() == 0.5
    
    # 断言展示对象的 score_name 属性为 "Score"
    assert display.score_name == "Score"
    
    # 断言展示对象所关联的坐标轴的 X 标签为 param_name 参数的字符串表示
    assert display.ax_.get_xlabel() == f"{param_name}"
    
    # 断言展示对象所关联的坐标轴的 Y 标签为 "Score"
    assert display.ax_.get_ylabel() == "Score"
    
    # 获取展示对象所关联坐标轴的图例标签和句柄
    _, legend_labels = display.ax_.get_legend_handles_labels()
    
    # 断言图例标签为 ["Train", "Test"]
    assert legend_labels == ["Train", "Test"]
    
    # 使用给定的评估器、数据集和参数范围计算验证曲线的训练分数和测试分数
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range
    )
    
    # 断言展示对象的 param_range 属性与给定的 param_range 参数相等
    assert_array_equal(display.param_range, param_range)
    
    # 断言展示对象的 train_scores 属性与计算得到的 train_scores 相近
    assert_allclose(display.train_scores, train_scores)
    
    # 断言展示对象的 test_scores 属性与计算得到的 test_scores 相近
    assert_allclose(display.test_scores, test_scores)
# 使用 pytest 的 mark.parametrize 装饰器为 test_curve_display_negate_score 函数指定参数组合测试
@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [
        # 指定不同的 CurveDisplay 类型和特定参数的组合
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),
        (LearningCurveDisplay, {"train_sizes": [0.3, 0.6, 0.9]}),
    ],
)
# 定义测试函数 test_curve_display_negate_score，用于验证 negate_score 参数的行为
def test_curve_display_negate_score(pyplot, data, CurveDisplay, specific_params):
    """Check the behaviour of the `negate_score` parameter calling `from_estimator` and
    `plot`.
    """
    # 解包测试数据
    X, y = data
    # 创建决策树分类器实例
    estimator = DecisionTreeClassifier(max_depth=1, random_state=0)

    # 初始设定 negate_score 为 False
    negate_score = False
    # 调用 CurveDisplay 的 from_estimator 方法，生成显示对象 display
    display = CurveDisplay.from_estimator(
        estimator, X, y, **specific_params, negate_score=negate_score
    )

    # 获取正分数的数据线
    positive_scores = display.lines_[0].get_data()[1]
    # 断言所有分数均大于等于 0
    assert (positive_scores >= 0).all()
    # 断言显示对象的 y 轴标签为 "Score"
    assert display.ax_.get_ylabel() == "Score"

    # 设置 negate_score 为 True
    negate_score = True
    # 重新调用 from_estimator 方法，生成新的显示对象 display
    display = CurveDisplay.from_estimator(
        estimator, X, y, **specific_params, negate_score=negate_score
    )

    # 获取负分数的数据线
    negative_scores = display.lines_[0].get_data()[1]
    # 断言所有分数均小于等于 0
    assert (negative_scores <= 0).all()
    # 使用 assert_allclose 断言负分数与正分数的绝对值接近
    assert_allclose(negative_scores, -positive_scores)
    # 断言显示对象的 y 轴标签为 "Negative score"
    assert display.ax_.get_ylabel() == "Negative score"

    # 重置 negate_score 为 False
    negate_score = False
    # 再次调用 from_estimator 方法，生成新的显示对象 display
    display = CurveDisplay.from_estimator(
        estimator, X, y, **specific_params, negate_score=negate_score
    )
    # 断言显示对象的 y 轴标签为 "Score"
    assert display.ax_.get_ylabel() == "Score"
    # 调用 display 的 plot 方法，绘制曲线，参数 negate_score 取反
    display.plot(negate_score=not negate_score)
    # 断言显示对象的 y 轴标签为 "Score"
    assert display.ax_.get_ylabel() == "Score"
    # 断言所有分数均小于 0
    assert (display.lines_[0].get_data()[1] < 0).all()


# 使用 pytest 的 mark.parametrize 装饰器为 test_curve_display_score_name 函数指定参数组合测试
@pytest.mark.parametrize(
    "score_name, ylabel", [(None, "Score"), ("Accuracy", "Accuracy")]
)
# 使用 pytest 的 mark.parametrize 装饰器为 test_curve_display_score_name 函数指定参数组合测试
@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [
        # 指定不同的 CurveDisplay 类型和特定参数的组合
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),
        (LearningCurveDisplay, {"train_sizes": [0.3, 0.6, 0.9]}),
    ],
)
# 定义测试函数 test_curve_display_score_name，用于验证 score_name 参数的行为
def test_curve_display_score_name(
    pyplot, data, score_name, ylabel, CurveDisplay, specific_params
):
    """Check that we can overwrite the default score name shown on the y-axis."""
    # 解包测试数据
    X, y = data
    # 创建决策树分类器实例
    estimator = DecisionTreeClassifier(random_state=0)

    # 调用 CurveDisplay 的 from_estimator 方法，生成显示对象 display
    display = CurveDisplay.from_estimator(
        estimator, X, y, **specific_params, score_name=score_name
    )

    # 断言显示对象的 y 轴标签为预期的 ylabel
    assert display.ax_.get_ylabel() == ylabel
    # 重新定义 estimator 为具有限定最大深度的决策树分类器实例
    estimator = DecisionTreeClassifier(max_depth=1, random_state=0)

    # 重新调用 from_estimator 方法，生成新的显示对象 display
    display = CurveDisplay.from_estimator(
        estimator, X, y, **specific_params, score_name=score_name
    )

    # 断言显示对象的 score_name 属性与预期的 ylabel 相等
    assert display.score_name == ylabel


# 使用 pytest 的 mark.parametrize 装饰器为 test_learning_curve_display_score_type 函数指定参数测试
@pytest.mark.parametrize("std_display_style", (None, "errorbar"))
# 定义测试函数 test_learning_curve_display_score_type，用于验证 score_type 参数的行为
def test_learning_curve_display_score_type(pyplot, data, std_display_style):
    """Check the behaviour of setting the `score_type` parameter."""
    # 解包测试数据
    X, y = data
    # 创建决策树分类器实例
    estimator = DecisionTreeClassifier(random_state=0)

    # 定义训练样本大小列表
    train_sizes = [0.3, 0.6, 0.9]
    # 计算学习曲线的训练样本大小、训练分数和测试分数
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes
    )

    # 设置 score_type 参数为 "train"
    score_type = "train"
    # 使用给定的评估器、数据集和参数，创建一个学习曲线展示对象
    display = LearningCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    # 获取图例的标签，并确保其包含了预期的训练集标签
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Train"]

    # 如果没有指定标准差显示风格
    if std_display_style is None:
        # 确保只有一个线条（无误差线）
        assert len(display.lines_) == 1
        assert display.errorbar_ is None
        # 获取训练集数据的 x 和 y 值
        x_data, y_data = display.lines_[0].get_data()
    else:
        # 确保没有线条（有误差线）
        assert display.lines_ is None
        assert len(display.errorbar_) == 1
        # 获取误差线中第一个线条的数据
        x_data, y_data = display.errorbar_[0].lines[0].get_data()

    # 确保获取的 x 数据与给定的训练集大小一致
    assert_array_equal(x_data, train_sizes_abs)
    # 确保获取的 y 数据与训练分数的均值（沿轴 1）一致
    assert_allclose(y_data, train_scores.mean(axis=1))

    # 切换评分类型为测试集
    score_type = "test"
    # 重新创建学习曲线展示对象，用于测试集数据
    display = LearningCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    # 获取图例的标签，并确保其包含了预期的测试集标签
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Test"]

    # 如果没有指定标准差显示风格
    if std_display_style is None:
        # 确保只有一个线条（无误差线）
        assert len(display.lines_) == 1
        assert display.errorbar_ is None
        # 获取测试集数据的 x 和 y 值
        x_data, y_data = display.lines_[0].get_data()
    else:
        # 确保没有线条（有误差线）
        assert display.lines_ is None
        assert len(display.errorbar_) == 1
        # 获取误差线中第一个线条的数据
        x_data, y_data = display.errorbar_[0].lines[0].get_data()

    # 确保获取的 x 数据与给定的训练集大小一致
    assert_array_equal(x_data, train_sizes_abs)
    # 确保获取的 y 数据与测试分数的均值（沿轴 1）一致
    assert_allclose(y_data, test_scores.mean(axis=1))

    # 切换评分类型为训练集和测试集都包含
    score_type = "both"
    # 重新创建学习曲线展示对象，用于同时展示训练集和测试集数据
    display = LearningCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    # 获取图例的标签，并确保其包含了预期的训练集和测试集标签
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Train", "Test"]

    # 如果没有指定标准差显示风格
    if std_display_style is None:
        # 确保有两个线条（无误差线）
        assert len(display.lines_) == 2
        assert display.errorbar_ is None
        # 获取训练集数据的 x 和 y 值
        x_data_train, y_data_train = display.lines_[0].get_data()
        # 获取测试集数据的 x 和 y 值
        x_data_test, y_data_test = display.lines_[1].get_data()
    else:
        # 确保没有线条（有误差线）
        assert display.lines_ is None
        assert len(display.errorbar_) == 2
        # 获取误差线中第一个线条（对应训练集）的数据
        x_data_train, y_data_train = display.errorbar_[0].lines[0].get_data()
        # 获取误差线中第二个线条（对应测试集）的数据
        x_data_test, y_data_test = display.errorbar_[1].lines[0].get_data()

    # 确保获取的训练集 x 数据与给定的训练集大小一致
    assert_array_equal(x_data_train, train_sizes_abs)
    # 确保获取的训练集 y 数据与训练分数的均值（沿轴 1）一致
    assert_allclose(y_data_train, train_scores.mean(axis=1))
    # 确保获取的测试集 x 数据与给定的训练集大小一致
    assert_array_equal(x_data_test, train_sizes_abs)
    # 确保获取的测试集 y 数据与测试分数的均值（沿轴 1）一致
    assert_allclose(y_data_test, test_scores.mean(axis=1))
@pytest.mark.parametrize("std_display_style", (None, "errorbar"))
def test_validation_curve_display_score_type(pyplot, data, std_display_style):
    """Check the behaviour of setting the `score_type` parameter."""

    # 从测试数据中获取特征 X 和标签 y
    X, y = data

    # 创建一个决策树分类器作为评估器
    estimator = DecisionTreeClassifier(random_state=0)

    # 设置参数名称和参数范围
    param_name, param_range = "max_depth", [1, 3, 5]

    # 使用验证曲线函数计算训练集和测试集的得分
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range
    )

    # 设置评分类型为 "train"，创建显示对象
    score_type = "train"
    display = ValidationCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    # 检查图例标签是否为 ["Train"]
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Train"]

    # 根据 std_display_style 的不同情况，检查绘制的曲线或误差条
    if std_display_style is None:
        assert len(display.lines_) == 1
        assert display.errorbar_ is None
        x_data, y_data = display.lines_[0].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 1
        x_data, y_data = display.errorbar_[0].lines[0].get_data()

    # 检查 x_data 是否与 param_range 相等
    assert_array_equal(x_data, param_range)
    # 检查 y_data 是否接近于 train_scores 的均值
    assert_allclose(y_data, train_scores.mean(axis=1))

    # 设置评分类型为 "test"，创建显示对象
    score_type = "test"
    display = ValidationCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    # 检查图例标签是否为 ["Test"]
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Test"]

    # 根据 std_display_style 的不同情况，检查绘制的曲线或误差条
    if std_display_style is None:
        assert len(display.lines_) == 1
        assert display.errorbar_ is None
        x_data, y_data = display.lines_[0].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 1
        x_data, y_data = display.errorbar_[0].lines[0].get_data()

    # 检查 x_data 是否与 param_range 相等
    assert_array_equal(x_data, param_range)
    # 检查 y_data 是否接近于 test_scores 的均值
    assert_allclose(y_data, test_scores.mean(axis=1))

    # 设置评分类型为 "both"，创建显示对象
    score_type = "both"
    display = ValidationCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    # 检查图例标签是否为 ["Train", "Test"]
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Train", "Test"]

    # 根据 std_display_style 的不同情况，检查绘制的曲线或误差条
    if std_display_style is None:
        assert len(display.lines_) == 2
        assert display.errorbar_ is None
        x_data_train, y_data_train = display.lines_[0].get_data()
        x_data_test, y_data_test = display.lines_[1].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 2
        x_data_train, y_data_train = display.errorbar_[0].lines[0].get_data()
        x_data_test, y_data_test = display.errorbar_[1].lines[0].get_data()

    # 检查 x_data_train 是否与 param_range 相等
    assert_array_equal(x_data_train, param_range)
    # 断言：验证 y_data_train 与 train_scores 按行均值相等
    assert_allclose(y_data_train, train_scores.mean(axis=1))
    
    # 断言：验证 x_data_test 与 param_range 相等
    assert_array_equal(x_data_test, param_range)
    
    # 断言：验证 y_data_test 与 test_scores 按行均值相等
    assert_allclose(y_data_test, test_scores.mean(axis=1))
@pytest.mark.parametrize(
    "CurveDisplay, specific_params, expected_xscale",
    [  # 参数化测试，用不同的参数组合多次运行测试函数
        (
            ValidationCurveDisplay,  # 使用 ValidationCurveDisplay 类进行验证曲线显示
            {"param_name": "max_depth", "param_range": np.arange(1, 5)},  # 参数设置：最大深度参数范围为1到4
            "linear",  # 预期的 x 轴缩放类型为线性
        ),
        (
            LearningCurveDisplay,  # 使用 LearningCurveDisplay 类进行学习曲线显示
            {"train_sizes": np.linspace(0.1, 0.9, num=5)},  # 参数设置：训练样本大小从0.1到0.9均匀分布，共5个点
            "linear",  # 预期的 x 轴缩放类型为线性
        ),
        (
            ValidationCurveDisplay,  # 使用 ValidationCurveDisplay 类进行验证曲线显示
            {
                "param_name": "max_depth",  # 参数名称为 max_depth
                "param_range": np.round(np.logspace(0, 2, num=5)).astype(np.int64),  # 参数范围为对数空间中四舍五入后的整数
            },
            "log",  # 预期的 x 轴缩放类型为对数
        ),
        (
            LearningCurveDisplay,  # 使用 LearningCurveDisplay 类进行学习曲线显示
            {"train_sizes": np.logspace(-1, 0, num=5)},  # 参数设置：训练样本大小从0.1到1对数分布，共5个点
            "log",  # 预期的 x 轴缩放类型为对数
        ),
    ],
)
def test_curve_display_xscale_auto(
    pyplot, data, CurveDisplay, specific_params, expected_xscale
):
    """Check the behaviour of the x-axis scaling depending on the data provided."""
    X, y = data  # 获取测试数据集 X 和 y
    estimator = DecisionTreeClassifier(random_state=0)  # 创建决策树分类器对象

    display = CurveDisplay.from_estimator(estimator, X, y, **specific_params)  # 使用给定参数从估计器创建显示对象
    assert display.ax_.get_xscale() == expected_xscale  # 断言预期的 x 轴缩放类型与实际显示对象的 x 轴缩放类型一致


@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [  # 参数化测试，用不同的参数组合多次运行测试函数
        (
            ValidationCurveDisplay,  # 使用 ValidationCurveDisplay 类进行验证曲线显示
            {"param_name": "max_depth", "param_range": [1, 3, 5]},  # 参数设置：最大深度参数范围为 1, 3, 5
        ),
        (
            LearningCurveDisplay,  # 使用 LearningCurveDisplay 类进行学习曲线显示
            {"train_sizes": [0.3, 0.6, 0.9]},  # 参数设置：训练样本大小为 0.3, 0.6, 0.9
        ),
    ],
)
def test_curve_display_std_display_style(pyplot, data, CurveDisplay, specific_params):
    """Check the behaviour of the parameter `std_display_style`."""
    X, y = data  # 获取测试数据集 X 和 y
    estimator = DecisionTreeClassifier(random_state=0)  # 创建决策树分类器对象

    import matplotlib as mpl  # 导入 matplotlib 库

    std_display_style = None  # 初始化标准显示样式为 None
    display = CurveDisplay.from_estimator(
        estimator,
        X,
        y,
        **specific_params,
        std_display_style=std_display_style,  # 使用指定的标准显示样式参数创建显示对象
    )

    assert len(display.lines_) == 2  # 断言显示对象中存在两条线
    for line in display.lines_:  # 遍历显示对象中的每条线
        assert isinstance(line, mpl.lines.Line2D)  # 断言每条线都是 Line2D 对象
    assert display.errorbar_ is None  # 断言 errorbar 对象为空
    assert display.fill_between_ is None  # 断言 fill_between 对象为空
    _, legend_label = display.ax_.get_legend_handles_labels()  # 获取图例句柄和标签
    assert len(legend_label) == 2  # 断言图例标签数量为2个

    std_display_style = "fill_between"  # 设置标准显示样式为 "fill_between"
    display = CurveDisplay.from_estimator(
        estimator,
        X,
        y,
        **specific_params,
        std_display_style=std_display_style,  # 使用指定的标准显示样式参数创建显示对象
    )

    assert len(display.lines_) == 2  # 断言显示对象中存在两条线
    for line in display.lines_:  # 遍历显示对象中的每条线
        assert isinstance(line, mpl.lines.Line2D)  # 断言每条线都是 Line2D 对象
    assert display.errorbar_ is None  # 断言 errorbar 对象为空
    assert len(display.fill_between_) == 2  # 断言 fill_between 对象存在两个
    for fill_between in display.fill_between_:  # 遍历 fill_between 对象列表
        assert isinstance(fill_between, mpl.collections.PolyCollection)  # 断言每个 fill_between 对象是 PolyCollection 对象
    _, legend_label = display.ax_.get_legend_handles_labels()  # 获取图例句柄和标签
    assert len(legend_label) == 2  # 断言图例标签数量为2个

    std_display_style = "errorbar"  # 设置标准显示样式为 "errorbar"
    display = CurveDisplay.from_estimator(
        estimator,
        X,
        y,
        **specific_params,
        std_display_style=std_display_style,  # 使用指定的标准显示样式参数创建显示对象
    )

    assert display.lines_ is None  # 断言 lines 对象为空
    assert len(display.errorbar_) == 2  # 断言 errorbar 对象存在两个
    # 对于 display.errorbar_ 中的每一个元素进行类型断言，确保是 ErrorbarContainer 类型
    for errorbar in display.errorbar_:
        assert isinstance(errorbar, mpl.container.ErrorbarContainer)
    
    # 确保 display.fill_between_ 为 None
    assert display.fill_between_ is None
    
    # 获取图形对象 display.ax_ 的图例句柄和标签
    _, legend_label = display.ax_.get_legend_handles_labels()
    
    # 断言图例标签的长度为 2
    assert len(legend_label) == 2
@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [  # 使用 pytest.mark.parametrize 装饰器标记测试函数，定义参数化测试
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),
        (LearningCurveDisplay, {"train_sizes": [0.3, 0.6, 0.9]}),
    ],
)
def test_curve_display_plot_kwargs(pyplot, data, CurveDisplay, specific_params):
    """Check the behaviour of the different plotting keyword arguments: `line_kw`,
    `fill_between_kw`, and `errorbar_kw`."""
    X, y = data  # 获取测试数据 X, y
    estimator = DecisionTreeClassifier(random_state=0)  # 初始化决策树分类器

    std_display_style = "fill_between"  # 设置显示样式为 "fill_between"
    line_kw = {"color": "red"}  # 设置线条属性字典，指定颜色为红色
    fill_between_kw = {"color": "red", "alpha": 1.0}  # 设置填充区域属性字典，指定颜色为红色，不透明度为1.0
    display = CurveDisplay.from_estimator(  # 使用 CurveDisplay 类的静态方法创建显示对象
        estimator,
        X,
        y,
        **specific_params,  # 传递特定的参数到显示对象的构造函数
        std_display_style=std_display_style,
        line_kw=line_kw,
        fill_between_kw=fill_between_kw,
    )

    assert display.lines_[0].get_color() == "red"  # 断言线条颜色是否为红色
    assert_allclose(
        display.fill_between_[0].get_facecolor(),
        [[1.0, 0.0, 0.0, 1.0]],  # 断言填充颜色是否为红色，使用数值表示颜色，[1.0, 0.0, 0.0, 1.0] 对应红色
    )

    std_display_style = "errorbar"  # 设置显示样式为 "errorbar"
    errorbar_kw = {"color": "red"}  # 设置误差条属性字典，指定颜色为红色
    display = CurveDisplay.from_estimator(
        estimator,
        X,
        y,
        **specific_params,
        std_display_style=std_display_style,
        errorbar_kw=errorbar_kw,
    )

    assert display.errorbar_[0].lines[0].get_color() == "red"  # 断言误差条颜色是否为红色


@pytest.mark.parametrize(
    "param_range, xscale",
    [  # 使用 pytest.mark.parametrize 装饰器标记测试函数，定义参数化测试
        ([5, 10, 15], "linear"),  # 参数范围和 X 轴比例为线性
        ([-50, 5, 50, 500], "symlog"),  # 参数范围和 X 轴比例为对数对称
        ([5, 50, 500], "log"),  # 参数范围和 X 轴比例为对数
    ],
)
def test_validation_curve_xscale_from_param_range_provided_as_a_list(
    pyplot, data, param_range, xscale
):
    """Check the induced xscale from the provided param_range values."""
    X, y = data  # 获取测试数据 X, y
    estimator = DecisionTreeClassifier(random_state=0)  # 初始化决策树分类器

    param_name = "max_depth"  # 设置参数名称为 "max_depth"
    display = ValidationCurveDisplay.from_estimator(  # 使用 ValidationCurveDisplay 类的静态方法创建显示对象
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
    )

    assert display.ax_.get_xscale() == xscale  # 断言 X 轴的比例是否符合预期


@pytest.mark.parametrize(
    "Display, params",
    [  # 使用 pytest.mark.parametrize 装饰器标记测试函数，定义参数化测试
        (LearningCurveDisplay, {}),  # 测试 LearningCurveDisplay 类，不传入额外参数
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),  # 测试 ValidationCurveDisplay 类，传入特定参数
    ],
)
def test_subclassing_displays(pyplot, data, Display, params):
    """Check that named constructors return the correct type when subclassed.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/27675
    """
    X, y = data  # 获取测试数据 X, y
    estimator = DecisionTreeClassifier(random_state=0)  # 初始化决策树分类器

    class SubclassOfDisplay(Display):  # 定义 Display 的子类
        pass

    display = SubclassOfDisplay.from_estimator(estimator, X, y, **params)  # 使用子类的静态方法创建显示对象
    assert isinstance(display, SubclassOfDisplay)  # 断言显示对象是否为子类类型
```