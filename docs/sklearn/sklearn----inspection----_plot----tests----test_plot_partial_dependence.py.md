# `D:\src\scipysrc\scikit-learn\sklearn\inspection\_plot\tests\test_plot_partial_dependence.py`

```
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats.mstats import mquantiles

from sklearn.compose import make_column_transformer  # 导入 make_column_transformer 模块
from sklearn.datasets import (  # 导入多个数据集模块
    load_diabetes,  # 加载糖尿病数据集
    load_iris,  # 加载鸢尾花数据集
    make_classification,  # 生成分类数据集
    make_regression,  # 生成回归数据集
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  # 导入梯度提升分类器和回归器
from sklearn.inspection import PartialDependenceDisplay  # 导入部分依赖显示模块
from sklearn.linear_model import LinearRegression  # 导入线性回归模块
from sklearn.pipeline import make_pipeline  # 导入构建管道的模块
from sklearn.preprocessing import OneHotEncoder  # 导入独热编码模块
from sklearn.utils._testing import _convert_container  # 导入测试工具函数

# TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
# 设定 pytest 的标记来过滤警告信息，以便忽略特定的警告
pytestmark = pytest.mark.filterwarnings(
    (
        "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
        "matplotlib.*"
    ),
)


@pytest.fixture(scope="module")
def diabetes():
    # 糖尿病数据集，为了加快速度而进行了子采样
    data = load_diabetes()
    data.data = data.data[:50]  # 只保留前50个样本的数据
    data.target = data.target[:50]  # 只保留前50个样本的目标值
    return data


@pytest.fixture(scope="module")
def clf_diabetes(diabetes):
    # 使用梯度提升回归器对糖尿病数据进行训练
    clf = GradientBoostingRegressor(n_estimators=10, random_state=1)
    clf.fit(diabetes.data, diabetes.target)
    return clf


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize("grid_resolution", [10, 20])
def test_plot_partial_dependence(grid_resolution, pyplot, clf_diabetes, diabetes):
    # 测试部分依赖图绘制函数
    # 使用列 0 和 2，因为列 1 不是定量特征（性别）
    feature_names = diabetes.feature_names
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 2, (0, 2)],
        grid_resolution=grid_resolution,
        feature_names=feature_names,
        contour_kw={"cmap": "jet"},
    )
    fig = pyplot.gcf()  # 获取当前图形
    axs = fig.get_axes()  # 获取图形的所有轴
    assert disp.figure_ is fig  # 断言部分依赖显示对象的图形与当前图形一致
    assert len(axs) == 4  # 断言图形的轴数为 4

    assert disp.bounding_ax_ is not None  # 断言边界轴不为 None
    assert disp.axes_.shape == (1, 3)  # 断言轴数组的形状为 (1, 3)
    assert disp.lines_.shape == (1, 3)  # 断言线条数组的形状为 (1, 3)
    assert disp.contours_.shape == (1, 3)  # 断言等高线数组的形状为 (1, 3)
    assert disp.deciles_vlines_.shape == (1, 3)  # 断言分位数垂直线数组的形状为 (1, 3)
    assert disp.deciles_hlines_.shape == (1, 3)  # 断言分位数水平线数组的形状为 (1, 3)

    assert disp.lines_[0, 2] is None  # 断言特定位置的线条为 None
    assert disp.contours_[0, 0] is None  # 断言特定位置的等高线为 None
    assert disp.contours_[0, 1] is None  # 断言特定位置的等高线为 None

    # 分位数线：始终显示在 x 轴上，如果是二维部分依赖图，则在 y 轴上显示
    for i in range(3):
        assert disp.deciles_vlines_[0, i] is not None
    assert disp.deciles_hlines_[0, 0] is None
    assert disp.deciles_hlines_[0, 1] is None
    assert disp.deciles_hlines_[0, 2] is not None

    assert disp.features == [(0,), (2,), (0, 2)]  # 断言特征的组合
    assert np.all(disp.feature_names == feature_names)  # 断言特征名称数组与数据集的特征名称一致
    assert len(disp.deciles) == 2  # 断言分位数数组的长度为 2
    for i in [0, 2]:
        assert_allclose(
            disp.deciles[i],
            mquantiles(diabetes.data[:, i], prob=np.arange(0.1, 1.0, 0.1)),
        )

    single_feature_positions = [(0, (0, 0)), (2, (0, 1))]  # 单一特征的位置信息
    # 预期的 y 轴标签列表，包括部分依赖和空字符串
    expected_ylabels = ["Partial dependence", ""]

    # 遍历单个特征的位置及其索引
    for i, (feat_col, pos) in enumerate(single_feature_positions):
        # 获取当前位置的子图对象
        ax = disp.axes_[pos]
        # 断言 y 轴标签与预期的相符
        assert ax.get_ylabel() == expected_ylabels[i]
        # 断言 x 轴标签与糖尿病数据集中特定列的名称相符
        assert ax.get_xlabel() == diabetes.feature_names[feat_col]

        # 获取当前位置的线条对象
        line = disp.lines_[pos]

        # 获取平均预测值
        avg_preds = disp.pd_results[i]
        # 断言平均预测值的形状符合预期
        assert avg_preds.average.shape == (1, grid_resolution)
        # 获取目标索引
        target_idx = disp.target_idx

        # 获取线条数据
        line_data = line.get_data()
        # 断言线条数据的第一个数组与预测结果的网格值相近
        assert_allclose(line_data[0], avg_preds["grid_values"][0])
        # 断言线条数据的第二个数组与平均预测值中目标索引的展平值相近
        assert_allclose(line_data[1], avg_preds.average[target_idx].ravel())

    # 两个特征位置
    # 获取指定位置的子图对象
    ax = disp.axes_[0, 2]
    # 获取指定位置的等高线对象
    contour = disp.contours_[0, 2]
    # 断言等高线对象的颜色映射名称为 "jet"
    assert contour.get_cmap().name == "jet"
    # 断言 x 轴标签与糖尿病数据集中第一个特征的名称相符
    assert ax.get_xlabel() == diabetes.feature_names[0]
    # 断言 y 轴标签与糖尿病数据集中第三个特征的名称相符
    assert ax.get_ylabel() == diabetes.feature_names[2]
@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
# 使用 pytest 的标记忽略特定警告信息
@pytest.mark.parametrize(
    "kind, centered, subsample, shape",
    # 参数化测试，定义多组输入参数及其期望输出形状
    [
        ("average", False, None, (1, 3)),
        ("individual", False, None, (1, 3, 50)),
        ("both", False, None, (1, 3, 51)),
        ("individual", False, 20, (1, 3, 20)),
        ("both", False, 20, (1, 3, 21)),
        ("individual", False, 0.5, (1, 3, 25)),
        ("both", False, 0.5, (1, 3, 26)),
        ("average", True, None, (1, 3)),
        ("individual", True, None, (1, 3, 50)),
        ("both", True, None, (1, 3, 51)),
        ("individual", True, 20, (1, 3, 20)),
        ("both", True, 20, (1, 3, 21)),
    ],
)
def test_plot_partial_dependence_kind(
    pyplot,
    kind,
    centered,
    subsample,
    shape,
    clf_diabetes,
    diabetes,
):
    # 根据估计器和数据集生成部分依赖图展示对象
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 1, 2],
        kind=kind,
        centered=centered,
        subsample=subsample,
    )

    # 断言图对象的轴的形状符合预期
    assert disp.axes_.shape == (1, 3)
    # 断言图对象的线条形状符合预期
    assert disp.lines_.shape == shape
    # 断言图对象的等高线形状符合预期
    assert disp.contours_.shape == (1, 3)

    # 断言特定位置的等高线对象为空
    assert disp.contours_[0, 0] is None
    assert disp.contours_[0, 1] is None
    assert disp.contours_[0, 2] is None

    if centered:
        # 如果设置了居中参数，断言所有线条对象的首个点的 y 坐标为 0.0
        assert all([ln._y[0] == 0.0 for ln in disp.lines_.ravel() if ln is not None])
    else:
        # 否则断言所有线条对象的首个点的 y 坐标不为 0.0
        assert all([ln._y[0] != 0.0 for ln in disp.lines_.ravel() if ln is not None])


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
# 使用 pytest 的标记忽略特定警告信息
@pytest.mark.parametrize(
    "input_type, feature_names_type",
    # 参数化测试，定义多组输入参数及其期望输出形状
    [
        ("dataframe", None),
        ("dataframe", "list"),
        ("list", "list"),
        ("array", "list"),
        ("dataframe", "array"),
        ("list", "array"),
        ("array", "array"),
        ("dataframe", "series"),
        ("list", "series"),
        ("array", "series"),
        ("dataframe", "index"),
        ("list", "index"),
        ("array", "index"),
    ],
)
def test_plot_partial_dependence_str_features(
    pyplot,
    clf_diabetes,
    diabetes,
    input_type,
    feature_names_type,
):
    if input_type == "dataframe":
        # 如果输入类型为 dataframe，则导入 pandas 库
        pd = pytest.importorskip("pandas")
        # 创建 DataFrame 对象 X，使用糖尿病数据和特征名称
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    elif input_type == "list":
        # 如果输入类型为 list，则将糖尿病数据转换为列表形式
        X = diabetes.data.tolist()
    else:
        # 否则直接使用糖尿病数据
        X = diabetes.data

    if feature_names_type is None:
        # 如果特征名称类型为 None，则设置特征名称为 None
        feature_names = None
    else:
        # 否则根据特征名称类型转换特征名称容器
        feature_names = _convert_container(diabetes.feature_names, feature_names_type)

    grid_resolution = 25
    # 使用估计器生成部分依赖图展示对象，包括图形分辨率和特征名称
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        X,
        [("age", "bmi"), "bmi"],
        grid_resolution=grid_resolution,
        feature_names=feature_names,
        n_cols=1,
        line_kw={"alpha": 0.8},
    )
    # 获取当前图形对象
    fig = pyplot.gcf()
    # 获取图形对象的坐标轴
    axs = fig.get_axes()
    # 断言坐标轴数量符合预期
    assert len(axs) == 3

    # 断言图形对象与预期图形对象相等
    assert disp.figure_ is fig
    # 确保显示对象的轴的形状为 (2, 1)
    assert disp.axes_.shape == (2, 1)
    # 确保显示对象的线条形状为 (2, 1)
    assert disp.lines_.shape == (2, 1)
    # 确保显示对象的等高线形状为 (2, 1)
    assert disp.contours_.shape == (2, 1)
    # 确保显示对象的十分位竖线形状为 (2, 1)
    assert disp.deciles_vlines_.shape == (2, 1)
    # 确保显示对象的十分位横线形状为 (2, 1)
    assert disp.deciles_hlines_.shape == (2, 1)

    # 确保第一条线为 None
    assert disp.lines_[0, 0] is None
    # 确保第一条十分位竖线不为 None
    assert disp.deciles_vlines_[0, 0] is not None
    # 确保第一条十分位横线不为 None
    assert disp.deciles_hlines_[0, 0] is not None
    # 确保第二条等高线为 None
    assert disp.contours_[1, 0] is None
    # 确保第二条十分位横线为 None
    assert disp.deciles_hlines_[1, 0] is None
    # 确保第二条十分位竖线不为 None
    assert disp.deciles_vlines_[1, 0] is not None

    # 确保轴为显示对象的第二行第一列轴
    ax = disp.axes_[1, 0]
    # 确保轴的 x 轴标签为 "bmi"
    assert ax.get_xlabel() == "bmi"
    # 确保轴的 y 轴标签为 "Partial dependence"
    assert ax.get_ylabel() == "Partial dependence"

    # 获取第二条线
    line = disp.lines_[1, 0]
    # 获取平均预测值结果
    avg_preds = disp.pd_results[1]
    # 获取目标索引
    target_idx = disp.target_idx
    # 确保线条的透明度为 0.8
    assert line.get_alpha() == 0.8

    # 获取线条的数据
    line_data = line.get_data()
    # 确保线条数据的第一个元素与平均预测的网格值的第一个元素非常接近
    assert_allclose(line_data[0], avg_preds["grid_values"][0])
    # 确保线条数据的第二个元素与平均预测的目标索引的平均值展开后非常接近
    assert_allclose(line_data[1], avg_preds.average[target_idx].ravel())

    # 确保轴为显示对象的第一行第一列轴
    ax = disp.axes_[0, 0]
    # 确保轴的 x 轴标签为 "age"
    assert ax.get_xlabel() == "age"
    # 确保轴的 y 轴标签为 "bmi"
    assert ax.get_ylabel() == "bmi"
# 标记测试以忽略特定警告信息
@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
# 测试绘制部分依赖图，自定义坐标轴
def test_plot_partial_dependence_custom_axes(pyplot, clf_diabetes, diabetes):
    # 定义网格分辨率
    grid_resolution = 25
    # 创建包含两个子图的图形对象
    fig, (ax1, ax2) = pyplot.subplots(1, 2)
    # 从估计器生成部分依赖展示对象
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        ["age", ("age", "bmi")],
        grid_resolution=grid_resolution,
        feature_names=diabetes.feature_names,
        ax=[ax1, ax2],
    )
    # 断言确保图形对象正确生成
    assert fig is disp.figure_
    # 断言确保没有边界坐标轴
    assert disp.bounding_ax_ is None
    # 断言确保子图形状为 (2,)
    assert disp.axes_.shape == (2,)
    # 断言确保第一个子图是 ax1
    assert disp.axes_[0] is ax1
    # 断言确保第二个子图是 ax2

    assert disp.axes_[1] is ax2

    # 获取第一个子图的引用
    ax = disp.axes_[0]
    # 断言第一个子图的 x 轴标签为 "age"
    assert ax.get_xlabel() == "age"
    # 断言第一个子图的 y 轴标签为 "Partial dependence"
    assert ax.get_ylabel() == "Partial dependence"

    # 获取部分依赖曲线对象
    line = disp.lines_[0]
    # 获取平均预测值
    avg_preds = disp.pd_results[0]
    # 获取目标索引
    target_idx = disp.target_idx

    # 获取部分依赖曲线数据
    line_data = line.get_data()
    # 使用 assert_allclose 检查部分依赖曲线 x 轴数据的一致性
    assert_allclose(line_data[0], avg_preds["grid_values"][0])
    # 使用 assert_allclose 检查部分依赖曲线 y 轴数据的一致性
    assert_allclose(line_data[1], avg_preds.average[target_idx].ravel())

    # 获取第二个子图的引用
    ax = disp.axes_[1]
    # 断言第二个子图的 x 轴标签为 "age"
    assert ax.get_xlabel() == "age"
    # 断言第二个子图的 y 轴标签为 "bmi"


# 标记测试以忽略特定警告信息，并使用参数化测试传递不同的 kind 和 lines 值
@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize(
    "kind, lines", [("average", 1), ("individual", 50), ("both", 51)]
)
# 测试绘制部分依赖图，使用 NumPy 数组作为坐标轴
def test_plot_partial_dependence_passing_numpy_axes(
    pyplot, clf_diabetes, diabetes, kind, lines
):
    # 定义网格分辨率
    grid_resolution = 25
    # 获取特征名称
    feature_names = diabetes.feature_names
    # 从估计器生成部分依赖展示对象
    disp1 = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        ["age", "bmi"],
        kind=kind,
        grid_resolution=grid_resolution,
        feature_names=feature_names,
    )
    # 断言确保子图形状为 (1, 2)
    assert disp1.axes_.shape == (1, 2)
    # 断言确保第一个子图的 y 轴标签为 "Partial dependence"
    assert disp1.axes_[0, 0].get_ylabel() == "Partial dependence"
    # 断言确保第二个子图的 y 轴标签为空字符串
    assert disp1.axes_[0, 1].get_ylabel() == ""
    # 断言确保第一个子图的线条数量符合预期
    assert len(disp1.axes_[0, 0].get_lines()) == lines
    # 断言确保第二个子图的线条数量符合预期
    assert len(disp1.axes_[0, 1].get_lines()) == lines

    # 创建线性回归模型
    lr = LinearRegression()
    lr.fit(diabetes.data, diabetes.target)

    # 从线性回归模型生成部分依赖展示对象，并共享前一个展示对象的坐标轴
    disp2 = PartialDependenceDisplay.from_estimator(
        lr,
        diabetes.data,
        ["age", "bmi"],
        kind=kind,
        grid_resolution=grid_resolution,
        feature_names=feature_names,
        ax=disp1.axes_,
    )

    # 断言确保两个展示对象的坐标轴相同
    assert np.all(disp1.axes_ == disp2.axes_)
    # 断言确保第一个子图的线条数量加倍
    assert len(disp2.axes_[0, 0].get_lines()) == 2 * lines
    # 断言确保第二个子图的线条数量加倍
    assert len(disp2.axes_[0, 1].get_lines()) == 2 * lines


# 标记测试以忽略特定警告信息，并使用参数化测试传递不同的 nrows 和 ncols 值
@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize("nrows, ncols", [(2, 2), (3, 1)])
# 测试绘制部分依赖图，检查不正确的坐标轴数量
def test_plot_partial_dependence_incorrent_num_axes(
    pyplot, clf_diabetes, diabetes, nrows, ncols
):
    # 定义网格分辨率
    grid_resolution = 5
    # 创建具有指定行和列数的子图形对象
    fig, axes = pyplot.subplots(nrows, ncols)
    # 构建不同的 axes 格式列表
    axes_formats = [list(axes.ravel()), tuple(axes.ravel()), axes]

    # 根据实际获得的坐标轴数量生成错误消息
    msg = "Expected ax to have 2 axes, got {}".format(nrows * ncols)
    # 根据分类器 clf_diabetes 计算偏依赖图显示对象
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,                     # 使用指定的分类器对象 clf_diabetes
        diabetes.data,                    # 使用糖尿病数据集中的数据
        ["age", "bmi"],                   # 计算"年龄"和"BMI"两个特征的偏依赖
        grid_resolution=grid_resolution,   # 使用指定的网格分辨率进行计算
        feature_names=diabetes.feature_names,  # 使用糖尿病数据集中的特征名称
    )
    
    # 对于每个指定的轴格式，验证部分依赖图显示对象的创建是否引发值错误并包含特定消息
    for ax_format in axes_formats:
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言引发值错误，并匹配预期消息
            PartialDependenceDisplay.from_estimator(
                clf_diabetes,                 # 再次使用分类器对象 clf_diabetes
                diabetes.data,                # 使用糖尿病数据集中的数据
                ["age", "bmi"],               # 计算"年龄"和"BMI"两个特征的偏依赖
                grid_resolution=grid_resolution,  # 使用指定的网格分辨率进行计算
                feature_names=diabetes.feature_names,  # 使用糖尿病数据集中的特征名称
                ax=ax_format,                 # 使用指定的轴格式进行绘图
            )
    
        # 使用预先创建的偏依赖图显示对象 disp，在指定的轴上绘制图形
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言引发值错误，并匹配预期消息
            disp.plot(ax=ax_format)  # 在指定的轴上绘制偏依赖图
@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
# 定义一个测试函数，用于测试部分依赖图在相同轴上的绘制情况，同时忽略特定警告信息
def test_plot_partial_dependence_with_same_axes(pyplot, clf_diabetes, diabetes):
    # 第一次调用 plot_partial_dependence 会创建两个新的轴，放置在传入的轴的空间中，
    # 这样图中总共会有三个轴。
    # 目前的 API 不允许第二次调用 plot_partial_dependence 再次使用同一个轴，
    # 因为它会在空间中创建两个新的轴，导致总共五个轴。为了获得预期的行为，需要将生成的轴传递给第二次调用：
    # disp1 = plot_partial_dependence(...)
    # disp2 = plot_partial_dependence(..., ax=disp1.axes_)

    grid_resolution = 25
    # 创建一个新的图形和轴对象
    fig, ax = pyplot.subplots()
    # 使用给定的分类器绘制部分依赖图，并指定要分析的特征
    PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        ["age", "bmi"],
        grid_resolution=grid_resolution,
        feature_names=diabetes.feature_names,
        ax=ax,
    )

    # 准备一条错误消息，用于在预期值错误时抛出异常
    msg = (
        "The ax was already used in another plot function, please set "
        "ax=display.axes_ instead"
    )

    # 使用 pytest 检查是否抛出了预期的 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        PartialDependenceDisplay.from_estimator(
            clf_diabetes,
            diabetes.data,
            ["age", "bmi"],
            grid_resolution=grid_resolution,
            feature_names=diabetes.feature_names,
            ax=ax,
        )


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
# 定义一个测试函数，用于测试部分依赖图在重复使用特征名称时的情况，同时忽略特定警告信息
def test_plot_partial_dependence_feature_name_reuse(pyplot, clf_diabetes, diabetes):
    # 第二次调用 plot_partial_dependence 不会改变特征名称，仍然沿用第一次调用时的名称

    feature_names = diabetes.feature_names
    # 使用给定的分类器绘制部分依赖图，并指定要分析的特征
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 1],
        grid_resolution=10,
        feature_names=feature_names,
    )

    # 第二次使用同一个轴绘制部分依赖图
    PartialDependenceDisplay.from_estimator(
        clf_diabetes, diabetes.data, [0, 1], grid_resolution=10, ax=disp.axes_
    )

    # 验证每个轴对象的 x 轴标签是否与特征名称相匹配
    for i, ax in enumerate(disp.axes_.ravel()):
        assert ax.get_xlabel() == feature_names[i]


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
# 定义一个测试函数，用于测试多类别情况下的部分依赖图绘制功能，同时忽略特定警告信息
def test_plot_partial_dependence_multiclass(pyplot):
    grid_resolution = 25
    # 创建一个梯度提升分类器
    clf_int = GradientBoostingClassifier(n_estimators=10, random_state=1)
    # 加载鸢尾花数据集
    iris = load_iris()

    # 在多类别输入上测试部分依赖图函数
    clf_int.fit(iris.data, iris.target)
    # 绘制目标为第一个类别的部分依赖图
    disp_target_0 = PartialDependenceDisplay.from_estimator(
        clf_int, iris.data, [0, 3], target=0, grid_resolution=grid_resolution
    )
    # 验证返回的图形对象是否与当前图形对象相同
    assert disp_target_0.figure_ is pyplot.gcf()
    # 验证返回的轴对象的形状是否为 (1, 2)
    assert disp_target_0.axes_.shape == (1, 2)
    # 验证返回的线条对象的形状是否为 (1, 2)
    assert disp_target_0.lines_.shape == (1, 2)
    # 验证返回的等高线对象的形状是否为 (1, 2)
    assert disp_target_0.contours_.shape == (1, 2)
    # 验证返回的分位数垂直线对象的形状是否为 (1, 2)
    assert disp_target_0.deciles_vlines_.shape == (1, 2)
    # 验证返回的分位数水平线对象的形状是否为 (1, 2)
    assert disp_target_0.deciles_hlines_.shape == (1, 2)
    # 验证所有等高线对象是否都为 None
    assert all(c is None for c in disp_target_0.contours_.flat)
    # 确保 disp_target_0 的 target_idx 属性为 0
    assert disp_target_0.target_idx == 0
    
    # 使用 iris 数据集的目标标签作为目标变量
    target = iris.target_names[iris.target]
    
    # 使用 GradientBoostingClassifier 创建分类器对象 clf_symbol
    clf_symbol = GradientBoostingClassifier(n_estimators=10, random_state=1)
    
    # 使用 iris 数据和目标标签训练分类器 clf_symbol
    clf_symbol.fit(iris.data, target)
    
    # 使用 clf_symbol 创建 PartialDependenceDisplay 对象 disp_symbol，显示部分依赖
    disp_symbol = PartialDependenceDisplay.from_estimator(
        clf_symbol, iris.data, [0, 3], target="setosa", grid_resolution=grid_resolution
    )
    
    # 确保 disp_symbol 的 figure_ 属性与当前 pyplot 的图形对象相同
    assert disp_symbol.figure_ is pyplot.gcf()
    
    # 确保 disp_symbol 的 axes_ 属性形状为 (1, 2)
    assert disp_symbol.axes_.shape == (1, 2)
    
    # 确保 disp_symbol 的 lines_ 属性形状为 (1, 2)
    assert disp_symbol.lines_.shape == (1, 2)
    
    # 确保 disp_symbol 的 contours_ 属性形状为 (1, 2)
    assert disp_symbol.contours_.shape == (1, 2)
    
    # 确保 disp_symbol 的 deciles_vlines_ 属性形状为 (1, 2)
    assert disp_symbol.deciles_vlines_.shape == (1, 2)
    
    # 确保 disp_symbol 的 deciles_hlines_ 属性形状为 (1, 2)
    assert disp_symbol.deciles_hlines_.shape == (1, 2)
    
    # 确保 disp_symbol 的 contours_ 属性中所有元素都为 None
    assert all(c is None for c in disp_symbol.contours_.flat)
    
    # 确保 disp_symbol 的 target_idx 属性为 0
    assert disp_symbol.target_idx == 0
    
    # 比较 disp_target_0 和 disp_symbol 的部分依赖结果
    for int_result, symbol_result in zip(
        disp_target_0.pd_results, disp_symbol.pd_results
    ):
        # 确保平均值相近
        assert_allclose(int_result.average, symbol_result.average)
        # 确保 grid_values 相近
        assert_allclose(int_result["grid_values"], symbol_result["grid_values"])
    
    # 使用另一个目标值创建 PartialDependenceDisplay 对象 disp_target_1
    disp_target_1 = PartialDependenceDisplay.from_estimator(
        clf_int, iris.data, [0, 3], target=1, grid_resolution=grid_resolution
    )
    
    # 获取目标为 0 和 1 时的线数据，并确保它们不完全相等
    target_0_data_y = disp_target_0.lines_[0, 0].get_data()[1]
    target_1_data_y = disp_target_1.lines_[0, 0].get_data()[1]
    assert any(target_0_data_y != target_1_data_y)
multioutput_regression_data = make_regression(n_samples=50, n_targets=2, random_state=0)
# 生成一个具有两个目标变量的多输出回归数据集

@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize("target", [0, 1])
def test_plot_partial_dependence_multioutput(pyplot, target):
    # 测试多输出输入下的偏依赖图函数
    X, y = multioutput_regression_data
    clf = LinearRegression().fit(X, y)

    grid_resolution = 25
    # 从估计器（分类器）中创建偏依赖展示对象
    disp = PartialDependenceDisplay.from_estimator(
        clf, X, [0, 1], target=target, grid_resolution=grid_resolution
    )
    fig = pyplot.gcf()
    axs = fig.get_axes()
    assert len(axs) == 3
    assert disp.target_idx == target
    assert disp.bounding_ax_ is not None

    positions = [(0, 0), (0, 1)]
    expected_label = ["Partial dependence", ""]

    for i, pos in enumerate(positions):
        ax = disp.axes_[pos]
        assert ax.get_ylabel() == expected_label[i]
        assert ax.get_xlabel() == f"x{i}"

@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
def test_plot_partial_dependence_dataframe(pyplot, clf_diabetes, diabetes):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    grid_resolution = 25

    PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        df,
        ["bp", "s1"],
        grid_resolution=grid_resolution,
        feature_names=df.columns.tolist(),
    )

dummy_classification_data = make_classification(random_state=0)
# 生成一个分类数据集，用于测试分类问题

@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize(
    "data, params, err_msg",
    [
        # 第一个测试用例：多输出回归数据，未指定目标变量
        (
            multioutput_regression_data,
            {"target": None, "features": [0]},
            "target must be specified for multi-output",
        ),
        # 第二个测试用例：多输出回归数据，目标变量为-1，超出允许范围
        (
            multioutput_regression_data,
            {"target": -1, "features": [0]},
            r"target must be in \[0, n_tasks\]",
        ),
        # 第三个测试用例：多输出回归数据，目标变量为100，超出允许范围
        (
            multioutput_regression_data,
            {"target": 100, "features": [0]},
            r"target must be in \[0, n_tasks\]",
        ),
        # 第四个测试用例：虚拟分类数据，指定的特征名 'foobar' 不在特征列表中
        (
            dummy_classification_data,
            {"features": ["foobar"], "feature_names": None},
            "Feature 'foobar' not in feature_names",
        ),
        # 第五个测试用例：虚拟分类数据，指定的特征名 'foobar' 不在特征列表中
        (
            dummy_classification_data,
            {"features": ["foobar"], "feature_names": ["abcd", "def"]},
            "Feature 'foobar' not in feature_names",
        ),
        # 第六个测试用例：虚拟分类数据，特征列表中的元素不是整数或浮点数
        (
            dummy_classification_data,
            {"features": [(1, 2, 3)]},
            "Each entry in features must be either an int, ",
        ),
        # 第七个测试用例：虚拟分类数据，特征列表中的元素不是整数或浮点数
        (
            dummy_classification_data,
            {"features": [1, {}]},
            "Each entry in features must be either an int, ",
        ),
        # 第八个测试用例：虚拟分类数据，特征列表中的元素不是整数或浮点数
        (
            dummy_classification_data,
            {"features": [tuple()]},
            "Each entry in features must be either an int, ",
        ),
        # 第九个测试用例：虚拟分类数据，特征列表中的所有元素应小于某个值
        (
            dummy_classification_data,
            {"features": [123], "feature_names": ["blahblah"]},
            "All entries of features must be less than ",
        ),
        # 第十个测试用例：虚拟分类数据，特征名称列表中有重复的元素
        (
            dummy_classification_data,
            {"features": [0, 1, 2], "feature_names": ["a", "b", "a"]},
            "feature_names should not contain duplicates",
        ),
        # 第十一个测试用例：虚拟分类数据，`kind` 参数提供为字符串列表，但不满足条件
        (
            dummy_classification_data,
            {"features": [1, 2], "kind": ["both"]},
            "When `kind` is provided as a list of strings, it should contain",
        ),
        # 第十二个测试用例：虚拟分类数据，子样本参数为负数
        (
            dummy_classification_data,
            {"features": [1], "subsample": -1},
            "When an integer, subsample=-1 should be positive.",
        ),
        # 第十三个测试用例：虚拟分类数据，子样本参数超出允许的范围
        (
            dummy_classification_data,
            {"features": [1], "subsample": 1.2},
            r"When a floating-point, subsample=1.2 should be in the \(0, 1\) range",
        ),
        # 第十四个测试用例：虚拟分类数据，分类特征参数不是布尔数组
        (
            dummy_classification_data,
            {"features": [1, 2], "categorical_features": [1.0, 2.0]},
            "Expected `categorical_features` to be an array-like of boolean,",
        ),
        # 第十五个测试用例：虚拟分类数据，特征列表中包含元组，不支持二元部分依赖图
        (
            dummy_classification_data,
            {"features": [(1, 2)], "categorical_features": [2]},
            "Two-way partial dependence plots are not supported for pairs",
        ),
        # 第十六个测试用例：虚拟分类数据，不支持显示个体效应
        (
            dummy_classification_data,
            {"features": [1], "categorical_features": [1], "kind": "individual"},
            "It is not possible to display individual effects",
        ),
    ],
# 测试函数，用于测试绘制偏依赖图时是否能正确处理数值回归模型的错误情况
def test_plot_partial_dependence_error(pyplot, data, params, err_msg):
    # 从测试数据中获取特征和目标值
    X, y = data
    # 使用线性回归模型拟合数据
    estimator = LinearRegression().fit(X, y)

    # 断言捕获到 ValueError 异常，并匹配给定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        PartialDependenceDisplay.from_estimator(estimator, X, **params)


# 使用 pytest 标记，忽略特定警告消息类别
@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
# 参数化测试函数，针对不同的参数和错误消息进行测试
@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"target": 4, "features": [0]}, "target not in est.classes_, got 4"),
        ({"target": None, "features": [0]}, "target must be specified for multi-class"),
        (
            {"target": 1, "features": [4.5]},
            "Each entry in features must be either an int,",
        ),
    ],
)
# 测试函数，用于测试绘制偏依赖图时是否能正确处理多类别分类模型的错误情况
def test_plot_partial_dependence_multiclass_error(pyplot, params, err_msg):
    # 加载鸢尾花数据集
    iris = load_iris()
    # 使用梯度提升分类器拟合数据
    clf = GradientBoostingClassifier(n_estimators=10, random_state=1)
    clf.fit(iris.data, iris.target)

    # 断言捕获到 ValueError 异常，并匹配给定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        PartialDependenceDisplay.from_estimator(clf, iris.data, **params)


# 测试函数，用于确保在不覆盖 ylabel 的情况下绘制偏依赖图
def test_plot_partial_dependence_does_not_override_ylabel(
    pyplot, clf_diabetes, diabetes
):
    # 创建包含两个子图的图像
    _, axes = pyplot.subplots(1, 2)
    # 在第一个子图上设置 ylabel
    axes[0].set_ylabel("Hello world")
    # 绘制偏依赖图，并传入自定义的坐标轴
    PartialDependenceDisplay.from_estimator(
        clf_diabetes, diabetes.data, [0, 1], ax=axes
    )

    # 断言第一个子图的 ylabel 保持不变
    assert axes[0].get_ylabel() == "Hello world"
    # 断言第二个子图的 ylabel 是默认值 "Partial dependence"
    assert axes[1].get_ylabel() == "Partial dependence"


# 参数化测试函数，用于测试在包含分类特征的情况下绘制偏依赖图的功能
@pytest.mark.parametrize(
    "categorical_features, array_type",
    [
        (["col_A", "col_C"], "dataframe"),
        ([0, 2], "array"),
        ([True, False, True], "array"),
    ],
)
def test_plot_partial_dependence_with_categorical(
    pyplot, categorical_features, array_type
):
    # 定义包含分类特征的数据集 X 和目标值 y
    X = [[1, 1, "A"], [2, 0, "C"], [3, 2, "B"]]
    column_name = ["col_A", "col_B", "col_C"]
    # 转换数据集 X 的容器类型
    X = _convert_container(X, array_type, columns_name=column_name)
    y = np.array([1.2, 0.5, 0.45]).T

    # 创建数据预处理器，将分类特征进行独热编码
    preprocessor = make_column_transformer((OneHotEncoder(), categorical_features))
    # 创建包含预处理和线性回归模型的管道
    model = make_pipeline(preprocessor, LinearRegression())
    # 使用模型拟合数据
    model.fit(X, y)

    # 绘制单个特征的偏依赖图，并传入相关参数
    disp = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=["col_C"],
        feature_names=column_name,
        categorical_features=categorical_features,
    )

    # 断言绘制的图像与当前 pyplot 的图形是同一个对象
    assert disp.figure_ is pyplot.gcf()
    # 断言绘制的柱状图的形状为 (1, 1)
    assert disp.bars_.shape == (1, 1)
    # 断言绘制的柱状图的第一个条形图不为 None
    assert disp.bars_[0][0] is not None
    # 断言绘制的线条的形状为 (1, 1)
    assert disp.lines_.shape == (1, 1)
    # 断言绘制的线条的第一个线条为 None
    assert disp.lines_[0][0] is None
    # 断言绘制的等高线图的形状为 (1, 1)
    assert disp.contours_.shape == (1, 1)
    # 断言绘制的等高线图的第一个等高线为 None
    assert disp.contours_[0][0] is None
    # 断言绘制的十分位线的垂直线形状为 (1, 1)
    assert disp.deciles_vlines_.shape == (1, 1)
    # 断言绘制的十分位线的第一个垂直线为 None
    assert disp.deciles_vlines_[0][0] is None
    # 断言绘制的十分位线的水平线形状为 (1, 1)
    assert disp.deciles_hlines_.shape == (1, 1)
    # 断言绘制的十分位线的第一个水平线为 None
    assert disp.deciles_hlines_[0][0] is None
    # 断言绘制的坐标轴不包含图例
    assert disp.axes_[0, 0].get_legend() is None
    # 使用给定的模型和数据集 X，生成部分依赖图的显示对象
    disp = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=[("col_A", "col_C")],  # 指定要计算部分依赖的特征对
        feature_names=column_name,  # 特征名称列表
        categorical_features=categorical_features,  # 分类特征列表
    )

    # 断言部分依赖图的主图是当前 pyplot 的图形对象
    assert disp.figure_ is pyplot.gcf()

    # 断言柱状图的形状为 (1, 1)，表明只有一个柱状图
    assert disp.bars_.shape == (1, 1)

    # 断言柱状图的值为 None
    assert disp.bars_[0][0] is None

    # 断言线图的形状为 (1, 1)，表明只有一个线图
    assert disp.lines_.shape == (1, 1)

    # 断言线图的值为 None
    assert disp.lines_[0][0] is None

    # 断言轮廓图的形状为 (1, 1)，表明只有一个轮廓图
    assert disp.contours_.shape == (1, 1)

    # 断言轮廓图的值为 None
    assert disp.contours_[0][0] is None

    # 断言十分位垂直线的形状为 (1, 1)，表明只有一个十分位垂直线
    assert disp.deciles_vlines_.shape == (1, 1)

    # 断言十分位垂直线的值为 None
    assert disp.deciles_vlines_[0][0] is None

    # 断言十分位水平线的形状为 (1, 1)，表明只有一个十分位水平线
    assert disp.deciles_hlines_.shape == (1, 1)

    # 断言十分位水平线的值为 None
    assert disp.deciles_hlines_[0][0] is None

    # 断言显示对象的第一个子图没有图例
    assert disp.axes_[0, 0].get_legend() is None
def test_plot_partial_dependence_legend(pyplot):
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    
    # 创建包含三列的 DataFrame 对象 X，每列包含不同类型的数据
    X = pd.DataFrame(
        {
            "col_A": ["A", "B", "C"],
            "col_B": [1, 0, 2],
            "col_C": ["C", "B", "A"],
        }
    )
    
    # 创建目标变量 y，这里是一个 numpy 数组
    y = np.array([1.2, 0.5, 0.45]).T

    # 指定分类特征的列名
    categorical_features = ["col_A", "col_C"]
    
    # 创建列转换器，对指定的分类特征使用 OneHotEncoder 进行转换
    preprocessor = make_column_transformer((OneHotEncoder(), categorical_features))
    
    # 创建包含预处理和线性回归模型的管道
    model = make_pipeline(preprocessor, LinearRegression())
    
    # 使用 X 和 y 训练模型
    model.fit(X, y)

    # 从模型创建 PartialDependenceDisplay 对象
    disp = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=["col_B", "col_C"],
        categorical_features=categorical_features,
        kind=["both", "average"],
    )

    # 获取第一个子图的图例文本，并断言其内容为 "average"
    legend_text = disp.axes_[0, 0].get_legend().get_texts()
    assert len(legend_text) == 1
    assert legend_text[0].get_text() == "average"
    
    # 断言第二个子图没有图例
    assert disp.axes_[0, 1].get_legend() is None


@pytest.mark.parametrize(
    "kind, expected_shape",
    [("average", (1, 2)), ("individual", (1, 2, 20)), ("both", (1, 2, 21))],
)
def test_plot_partial_dependence_subsampling(
    pyplot, clf_diabetes, diabetes, kind, expected_shape
):
    # 检查子采样是否正常工作
    # 针对非回归测试：
    # https://github.com/scikit-learn/scikit-learn/pull/18359
    matplotlib = pytest.importorskip("matplotlib")
    
    # 定义网格分辨率
    grid_resolution = 25
    
    # 获取糖尿病数据集的特征名称
    feature_names = diabetes.feature_names

    # 从估计器创建 PartialDependenceDisplay 对象 disp1
    disp1 = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        ["age", "bmi"],
        kind=kind,
        grid_resolution=grid_resolution,
        feature_names=feature_names,
        subsample=20,
        random_state=0,
    )

    # 断言 disp1.lines_ 的形状符合预期
    assert disp1.lines_.shape == expected_shape
    # 断言 disp1.lines_ 中的所有元素都是 Line2D 对象
    assert all(
        [isinstance(line, matplotlib.lines.Line2D) for line in disp1.lines_.ravel()]
    )


@pytest.mark.parametrize(
    "kind, line_kw, label",
    [
        ("individual", {}, None),
        ("individual", {"label": "xxx"}, None),
        ("average", {}, None),
        ("average", {"label": "xxx"}, "xxx"),
        ("both", {}, "average"),
        ("both", {"label": "xxx"}, "xxx"),
    ],
)
def test_partial_dependence_overwrite_labels(
    pyplot,
    clf_diabetes,
    diabetes,
    kind,
    line_kw,
    label,
):
    """Test that make sure that we can overwrite the label of the PDP plot"""
    # 从估计器创建 PartialDependenceDisplay 对象 disp
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 2],
        grid_resolution=25,
        feature_names=diabetes.feature_names,
        kind=kind,
        line_kw=line_kw,
    )

    # 遍历 disp.axes_ 中的每个子图 ax
    for ax in disp.axes_.ravel():
        # 如果 label 为 None，断言 ax 没有图例
        if label is None:
            assert ax.get_legend() is None
        else:
            # 否则，获取 ax 的图例文本，并断言其为 label
            legend_text = ax.get_legend().get_texts()
            assert len(legend_text) == 1
            assert legend_text[0].get_text() == label
    [
        # 元组1: 包含列名 "col_A" 和 "col_C"，以及数据类型为 "dataframe"
        (["col_A", "col_C"], "dataframe"),
        # 元组2: 包含索引为 0 和 2 的元素，数据类型为 "array"
        ([0, 2], "array"),
        # 元组3: 包含布尔值 True, False, True，数据类型为 "array"
        ([True, False, True], "array"),
    ],
@pytest.mark.parametrize("kind", ["individual", "average", "both"])
@pytest.mark.parametrize("centered", [True, False])
def test_partial_dependence_plot_limits_one_way(
    pyplot, clf_diabetes, diabetes, kind, centered
):
    """Check that the PD limit on the plots are properly set on one-way plots."""
    # 从分类器和数据中创建部分依赖显示对象
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        features=(0, 1),
        kind=kind,
        grid_resolution=25,
        feature_names=diabetes.feature_names,
    )

    # 定义部分依赖范围
    range_pd = np.array([-1, 1], dtype=np.float64)
    for pd in disp.pd_results:
        # 设置平均部分依赖值的上下限
        if "average" in pd:
            pd["average"][...] = range_pd[1]
            pd["average"][0, 0] = range_pd[0]
        # 设置个体部分依赖值的上下限
        if "individual" in pd:
            pd["individual"][...] = range_pd[1]
            pd["individual"][0, 0, 0] = range_pd[0]

    # 绘制部分依赖图
    disp.plot(centered=centered)
    
    # 当中心化时，检查是否锚定到零点的 x 轴
    y_lim = range_pd - range_pd[0] if centered else range_pd
    padding = 0.05 * (y_lim[1] - y_lim[0])
    y_lim[0] -= padding
    y_lim[1] += padding
    for ax in disp.axes_.ravel():
        # 断言确保 y 轴的限制范围
        assert_allclose(ax.get_ylim(), y_lim)


@pytest.mark.parametrize("centered", [True, False])
def test_partial_dependence_plot_limits_two_way(
    pyplot, clf_diabetes, diabetes, centered
):
    """Check that the PD limit on the plots are properly set on two-way plots."""
    # 从分类器和数据中创建部分依赖显示对象
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        features=[(0, 1)],
        kind="average",
        grid_resolution=25,
        feature_names=diabetes.feature_names,
    )

    # 定义部分依赖范围
    range_pd = np.array([-1, 1], dtype=np.float64)
    for pd in disp.pd_results:
        # 设置平均部分依赖值的上下限
        pd["average"][...] = range_pd[1]
        pd["average"][0, 0] = range_pd[0]

    # 绘制部分依赖图
    disp.plot(centered=centered)
    
    # 获取等高线并定义级别
    contours = disp.contours_[0, 0]
    levels = range_pd - range_pd[0] if centered else range_pd
    # 计算填充值，为当前等级范围的5%
    padding = 0.05 * (levels[1] - levels[0])
    # 调整第一个等级的下限，使其减少填充值
    levels[0] -= padding
    # 调整第二个等级的上限，使其增加填充值
    levels[1] += padding
    # 使用线性分布生成预期的等级数组，包含8个数值
    expect_levels = np.linspace(*levels, num=8)
    # 断言实际等高线的等级数组与预期生成的等级数组非常接近
    assert_allclose(contours.levels, expect_levels)
# 定义测试函数，用于验证 `PartialDependenceDisplay` 是否能够处理 `kind` 参数为列表的情况。
def test_partial_dependence_kind_list(
    pyplot,
    clf_diabetes,
    diabetes,
):
    """Check that we can provide a list of strings to kind parameter."""
    # 导入 matplotlib 库，如果导入失败则跳过测试
    matplotlib = pytest.importorskip("matplotlib")

    # 从分类器 `clf_diabetes` 和数据 `diabetes.data` 创建 PartialDependenceDisplay 对象
    # 使用 features 参数指定要计算偏依赖的特征，kind 参数设为列表，包含不同的计算类型
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        features=[0, 2, (1, 2)],
        grid_resolution=20,
        kind=["both", "both", "average"],
    )

    # 验证前两个特征绘制的线条类型是否为 matplotlib.lines.Line2D 对象
    for idx in [0, 1]:
        assert all(
            [
                isinstance(line, matplotlib.lines.Line2D)
                for line in disp.lines_[0, idx].ravel()
            ]
        )
        # 验证对应的轮廓图是否为 None
        assert disp.contours_[0, idx] is None

    # 验证第三个特征绘制的轮廓图不为 None
    assert disp.contours_[0, 2] is not None
    # 验证第三个特征绘制的线条是否全部为 None
    assert all([line is None for line in disp.lines_[0, 2].ravel()])


# 使用 pytest 的参数化装饰器，定义多组参数化测试用例
@pytest.mark.parametrize(
    "features, kind",
    [
        ([0, 2, (1, 2)], "individual"),
        ([0, 2, (1, 2)], "both"),
        ([(0, 1), (0, 2), (1, 2)], "individual"),
        ([(0, 1), (0, 2), (1, 2)], "both"),
        ([0, 2, (1, 2)], ["individual", "individual", "individual"]),
        ([0, 2, (1, 2)], ["both", "both", "both"]),
    ],
)
# 定义测试函数，验证在请求二维偏依赖同时请求一维偏依赖或 ICE 图时是否能抛出信息丰富的错误
def test_partial_dependence_kind_error(
    pyplot,
    clf_diabetes,
    diabetes,
    features,
    kind,
):
    """Check that we raise an informative error when 2-way PD is requested
    together with 1-way PD/ICE"""
    # 预期的警告信息
    warn_msg = (
        "ICE plot cannot be rendered for 2-way feature interactions. 2-way "
        "feature interactions mandates PD plots using the 'average' kind"
    )
    # 使用 pytest 的断言检查是否抛出 ValueError 异常，并匹配预期的警告信息
    with pytest.raises(ValueError, match=warn_msg):
        # 从分类器 `clf_diabetes` 和数据 `diabetes.data` 创建 PartialDependenceDisplay 对象
        # features 参数指定要计算偏依赖的特征，kind 参数设置为传入的参数
        PartialDependenceDisplay.from_estimator(
            clf_diabetes,
            diabetes.data,
            features=features,
            grid_resolution=20,
            kind=kind,
        )


# 使用 pytest 的过滤警告装饰器，忽略特定警告信息
@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
# 使用 pytest 的参数化装饰器，定义多组参数化测试用例
@pytest.mark.parametrize(
    "line_kw, pd_line_kw, ice_lines_kw, expected_colors",
    [
        ({"color": "r"}, {"color": "g"}, {"color": "b"}, ("g", "b")),
        (None, {"color": "g"}, {"color": "b"}, ("g", "b")),
        ({"color": "r"}, None, {"color": "b"}, ("r", "b")),
        ({"color": "r"}, {"color": "g"}, None, ("g", "r")),
        ({"color": "r"}, None, None, ("r", "r")),
        ({"color": "r"}, {"linestyle": "--"}, {"linestyle": "-."}, ("r", "r")),
    ],
)
# 定义测试函数，验证传递 `pd_line_kw` 和 `ice_lines_kw` 是否会影响绘图中的特定线条
def test_plot_partial_dependence_lines_kw(
    pyplot,
    clf_diabetes,
    diabetes,
    line_kw,
    pd_line_kw,
    ice_lines_kw,
    expected_colors,
):
    """Check that passing `pd_line_kw` and `ice_lines_kw` will act on the
    specific lines in the plot.
    """
    # 从分类器 `clf_diabetes` 和数据 `diabetes.data` 创建 PartialDependenceDisplay 对象
    # features 参数指定要计算偏依赖的特征，kind 参数设为 "both"，其他参数用于自定义绘图样式
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 2],
        grid_resolution=20,
        feature_names=diabetes.feature_names,
        n_cols=2,
        kind="both",
        line_kw=line_kw,
        pd_line_kw=pd_line_kw,
        ice_lines_kw=ice_lines_kw,
    )

    # 获取绘制的最后一条线条对象
    line = disp.lines_[0, 0, -1]
    # 断言线条的颜色是否符合预期
    assert line.get_color() == expected_colors[0]
    # 检查是否定义了 pd_line_kw 并且其包含 "linestyle" 键，如果是则断言当前线条的样式与 pd_line_kw["linestyle"] 相符
    if pd_line_kw is not None and "linestyle" in pd_line_kw:
        assert line.get_linestyle() == pd_line_kw["linestyle"]
    else:
        # 如果未定义 pd_line_kw 或者未包含 "linestyle" 键，则断言当前线条的样式为虚线 "--"
        assert line.get_linestyle() == "--"
    
    # 获取第一个图形对象 disp 的第一个线条对象 lines_[0, 0, 0]
    line = disp.lines_[0, 0, 0]
    # 断言当前线条的颜色与预期颜色列表 expected_colors 中的第二个颜色相符
    assert line.get_color() == expected_colors[1]
    
    # 检查是否定义了 ice_lines_kw 并且其包含 "linestyle" 键，如果是则断言当前线条的样式与 ice_lines_kw["linestyle"] 相符
    if ice_lines_kw is not None and "linestyle" in ice_lines_kw:
        assert line.get_linestyle() == ice_lines_kw["linestyle"]
    else:
        # 如果未定义 ice_lines_kw 或者未包含 "linestyle" 键，则断言当前线条的样式为实线 "-"
        assert line.get_linestyle() == "-"
# 检查当 `kind` 是一个长度错误的列表时，是否会引发错误。
# 只能通过 `PartialDependenceDisplay.from_estimator` 方法触发此情况。
def test_partial_dependence_display_wrong_len_kind(
    pyplot,
    clf_diabetes,
    diabetes,
):
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        features=[0, 2],
        grid_resolution=20,
        kind="average",  # len(kind) != len(features)
    )

    # 将 `kind` 修改为包含与 `features` 长度不同的列表
    disp.kind = ["average"]
    err_msg = (
        r"When `kind` is provided as a list of strings, it should contain as many"
        r" elements as `features`. `kind` contains 1 element\(s\) and `features`"
        r" contains 2 element\(s\)."
    )
    # 使用 pytest 来检查是否引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        disp.plot()


@pytest.mark.parametrize(
    "kind",
    ["individual", "both", "average", ["average", "both"], ["individual", "both"]],
)
# 检查当将 `kind` 作为字符串或列表传递时，ICE 和 PD 是否能正确居中。
def test_partial_dependence_display_kind_centered_interaction(
    pyplot,
    kind,
    clf_diabetes,
    diabetes,
):
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 1],
        kind=kind,
        centered=True,
        subsample=5,
    )

    # 断言所有的线条起点是否都为 0.0
    assert all([ln._y[0] == 0.0 for ln in disp.lines_.ravel() if ln is not None])


# 检查使用恒定样本权重时是否保持标准行为。
def test_partial_dependence_display_with_constant_sample_weight(
    pyplot,
    clf_diabetes,
    diabetes,
):
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 1],
        kind="average",
        method="brute",
    )

    # 创建与目标数据形状相同的全一数组作为样本权重
    sample_weight = np.ones_like(diabetes.target)
    disp_sw = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 1],
        sample_weight=sample_weight,
        kind="average",
        method="brute",
    )

    # 断言两个 PartialDependenceDisplay 对象的平均部分依赖结果是否完全相等
    assert np.array_equal(
        disp.pd_results[0]["average"], disp_sw.pd_results[0]["average"]
    )


# 检查当子类化时，命名构造函数返回正确的子类类型。
# 非回归测试，用于检查子类化相关功能。
def test_subclass_named_constructors_return_type_is_subclass(
    pyplot, diabetes, clf_diabetes
):
    # 定义一个继承自 PartialDependenceDisplay 的子类
    class SubclassOfDisplay(PartialDependenceDisplay):
        pass

    # 使用子类的命名构造函数创建对象
    curve = SubclassOfDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 2, (0, 2)],
    )

    # 断言 curve 对象是否为 SubclassOfDisplay 的实例
    assert isinstance(curve, SubclassOfDisplay)
```