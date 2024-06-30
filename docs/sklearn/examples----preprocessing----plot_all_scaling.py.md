# `D:\src\scipysrc\scikit-learn\examples\preprocessing\plot_all_scaling.py`

```
"""
=============================================================
Compare the effect of different scalers on data with outliers
=============================================================

Feature 0 (median income in a block) and feature 5 (average house occupancy) of
the :ref:`california_housing_dataset` have very
different scales and contain some very large outliers. These two
characteristics lead to difficulties to visualize the data and, more
importantly, they can degrade the predictive performance of many machine
learning algorithms. Unscaled data can also slow down or even prevent the
convergence of many gradient-based estimators.

Indeed many estimators are designed with the assumption that each feature takes
values close to zero or more importantly that all features vary on comparable
scales. In particular, metric-based and gradient-based estimators often assume
approximately standardized data (centered features with unit variances). A
notable exception are decision tree-based estimators that are robust to
arbitrary scaling of the data.

This example uses different scalers, transformers, and normalizers to bring the
data within a pre-defined range.

Scalers are linear (or more precisely affine) transformers and differ from each
other in the way they estimate the parameters used to shift and scale each
feature.

:class:`~sklearn.preprocessing.QuantileTransformer` provides non-linear
transformations in which distances
between marginal outliers and inliers are shrunk.
:class:`~sklearn.preprocessing.PowerTransformer` provides
non-linear transformations in which data is mapped to a normal distribution to
stabilize variance and minimize skewness.

Unlike the previous transformations, normalization refers to a per sample
transformation instead of a per feature transformation.

The following code is a bit verbose, feel free to jump directly to the analysis
of the results_.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib as mpl  # 导入 matplotlib 库并简写为 mpl
import numpy as np  # 导入 numpy 库并简写为 np
from matplotlib import cm  # 从 matplotlib 中导入 cm 模块
from matplotlib import pyplot as plt  # 从 matplotlib 中导入 pyplot 模块并简写为 plt

from sklearn.datasets import fetch_california_housing  # 从 sklearn.datasets 中导入 fetch_california_housing 函数
from sklearn.preprocessing import (  # 从 sklearn.preprocessing 中导入以下预处理模块
    MaxAbsScaler,  # 最大值绝对值缩放器
    MinMaxScaler,  # 最小-最大缩放器
    Normalizer,  # 归一化器
    PowerTransformer,  # 功率变换器
    QuantileTransformer,  # 分位数变换器
    RobustScaler,  # 鲁棒缩放器
    StandardScaler,  # 标准化器
    minmax_scale,  # 最小-最大缩放函数
)

dataset = fetch_california_housing()  # 获取加利福尼亚房屋数据集
X_full, y_full = dataset.data, dataset.target  # 分别获取数据集特征和目标值
feature_names = dataset.feature_names  # 获取数据集特征名称

feature_mapping = {  # 特征映射字典，将特征名称映射到描述
    "MedInc": "Median income in block",  # 中位收入
    "HouseAge": "Median house age in block",  # 房龄中位数
    "AveRooms": "Average number of rooms",  # 平均房间数
    "AveBedrms": "Average number of bedrooms",  # 平均卧室数
    "Population": "Block population",  # 区块人口
    "AveOccup": "Average house occupancy",  # 平均房屋占用率
    "Latitude": "House block latitude",  # 房屋区块纬度
    "Longitude": "House block longitude",  # 房屋区块经度
}

# Take only 2 features to make visualization easier
# Feature MedInc has a long tail distribution.
# Feature AveOccup has a few but very large outliers.
# 定义要使用的特征列表
features = ["MedInc", "AveOccup"]
# 根据特征名称获取它们在数据集中的索引
features_idx = [feature_names.index(feature) for feature in features]
# 从完整数据集中提取指定特征列
X = X_full[:, features_idx]

# 定义不同数据变换的列表
distributions = [
    ("Unscaled data", X),  # 原始数据
    ("Data after standard scaling", StandardScaler().fit_transform(X)),  # 标准化数据
    ("Data after min-max scaling", MinMaxScaler().fit_transform(X)),  # 最小-最大缩放数据
    ("Data after max-abs scaling", MaxAbsScaler().fit_transform(X)),  # 最大绝对值缩放数据
    (
        "Data after robust scaling",
        RobustScaler(quantile_range=(25, 75)).fit_transform(X),  # 鲁棒缩放数据
    ),
    (
        "Data after power transformation (Yeo-Johnson)",
        PowerTransformer(method="yeo-johnson").fit_transform(X),  # Yeo-Johnson 幂变换数据
    ),
    (
        "Data after power transformation (Box-Cox)",
        PowerTransformer(method="box-cox").fit_transform(X),  # Box-Cox 幂变换数据
    ),
    (
        "Data after quantile transformation (uniform pdf)",
        QuantileTransformer(output_distribution="uniform", random_state=42).fit_transform(X),  # 分位数变换（均匀分布）数据
    ),
    (
        "Data after quantile transformation (gaussian pdf)",
        QuantileTransformer(output_distribution="normal", random_state=42).fit_transform(X),  # 分位数变换（正态分布）数据
    ),
    ("Data after sample-wise L2 normalizing", Normalizer().fit_transform(X)),  # 样本-wise L2 正则化数据
]

# 将输出值 y 进行 0 到 1 的缩放，以便用于色条
y = minmax_scale(y_full)

# 根据 matplotlib 的版本选择合适的色彩映射，使用 plasma_r 或者 hot_r
cmap = getattr(cm, "plasma_r", cm.hot_r)


def create_axes(title, figsize=(16, 6)):
    # 创建包含子图的图形对象，并设置总体标题
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # 定义第一个子图的位置和尺寸
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]
    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # 定义放大子图的位置和尺寸
    left = width + left + 0.2
    left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]
    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # 定义色条的位置和尺寸
    left, width = width + left + 0.13, 0.01
    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
        ax_colorbar,
    )


def plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    # 设置子图的标题和轴标签
    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # 绘制散点图
    colors = cmap(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c=colors)

    # 美学上去掉顶部和右侧的脊柱
    # 设置合适的坐标轴布局
    ax.spines["top"].set_visible(False)  # 隐藏顶部边框
    ax.spines["right"].set_visible(False)  # 隐藏右侧边框
    ax.get_xaxis().tick_bottom()  # 设置X轴刻度在底部
    ax.get_yaxis().tick_left()  # 设置Y轴刻度在左侧
    ax.spines["left"].set_position(("outward", 10))  # 左侧边框向外偏移10个单位
    ax.spines["bottom"].set_position(("outward", 10))  # 底部边框向外偏移10个单位
    
    # 为X1轴（特征5）绘制直方图
    hist_X1.set_ylim(ax.get_ylim())  # 设置X1轴直方图与当前轴相同的Y轴范围
    hist_X1.hist(
        X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
    )  # 绘制水平方向的直方图，灰色填充和边框
    hist_X1.axis("off")  # 关闭X1轴的坐标轴显示
    
    # 为X0轴（特征0）绘制直方图
    hist_X0.set_xlim(ax.get_xlim())  # 设置X0轴直方图与当前轴相同的X轴范围
    hist_X0.hist(
        X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
    )  # 绘制垂直方向的直方图，灰色填充和边框
    hist_X0.axis("off")  # 关闭X0轴的坐标轴显示
# %%
# 对每个缩放器/归一化器/转换器生成两个图。左图展示完整数据集的散点图，右图排除了极端值，
# 仅考虑数据集的 99%，排除了边缘异常值。此外，还会在散点图的两侧显示每个特征的边缘分布。

def make_plot(item_idx):
    # 获取数据集的标题和特征矩阵
    title, X = distributions[item_idx]
    # 创建三个子图：全局视图、局部放大视图和颜色条
    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    
    # 绘制全局视图
    plot_distribution(
        axarr[0],
        X,
        y,
        hist_nbins=200,
        x0_label=feature_mapping[features[0]],
        x1_label=feature_mapping[features[1]],
        title="Full data",
    )

    # zoom-in
    zoom_in_percentile_range = (0, 99)
    # 计算每个特征的百分位数范围作为放大视图的截断值
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    # 创建一个布尔掩码，排除边缘异常值
    non_outliers_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(
        X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1
    )
    # 绘制局部放大视图
    plot_distribution(
        axarr[1],
        X[non_outliers_mask],
        y[non_outliers_mask],
        hist_nbins=50,
        x0_label=feature_mapping[features[0]],
        x1_label=feature_mapping[features[1]],
        title="Zoom-in",
    )

    # 创建颜色条
    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(
        ax_colorbar,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
        label="Color mapping for values of y",
    )


# %%
# .. _results:
#
# 原始数据
# -------------
#
# 每个变换方法都将显示两个转换后的特征，左图显示整个数据集，右图放大显示去除了边缘异常值的数据集。
# 大多数样本集中在特定范围内，如中位收入为 [0, 10]，平均住房占用率为 [0, 6]。注意，有些边缘异常值
# （某些街区的平均住房占用率超过 1200）存在。因此，具体的预处理方法根据应用场景可能非常有益。接下来，
# 我们将展示这些预处理方法在存在边缘异常值时的一些见解和行为。

make_plot(0)

# %%
# .. _plot_all_scaling_standard_scaler_section:
#
# StandardScaler
# --------------
#
# :class:`~sklearn.preprocessing.StandardScaler` 移除均值并将数据缩放到单位方差。缩放会缩小
# 特征值的范围，如下图左图所示。然而，异常值在计算经验均值和标准差时会产生影响。特别注意，
# 因为每个特征的异常值具有不同的幅度，所以在每个特征上转换后数据的分布范围会有很大差异：
# 大多数数据在转换后的中位收入特征上的范围为 [-2, 4]，而同样的数据在转换后的平均住房占用率
# 特征上则被挤压在...
# smaller [-0.2, 0.2] range for the transformed average house occupancy.
#
# :class:`~sklearn.preprocessing.StandardScaler` therefore cannot guarantee
# balanced feature scales in the
# presence of outliers.
make_plot(1)

# %%
# .. _plot_all_scaling_minmax_scaler_section:
#
# MinMaxScaler
# ------------
#
# :class:`~sklearn.preprocessing.MinMaxScaler` rescales the data set such that
# all feature values are in
# the range [0, 1] as shown in the right panel below. However, this scaling
# compresses all inliers into the narrow range [0, 0.005] for the transformed
# average house occupancy.
#
# Both :class:`~sklearn.preprocessing.StandardScaler` and
# :class:`~sklearn.preprocessing.MinMaxScaler` are very sensitive to the
# presence of outliers.
make_plot(2)

# %%
# .. _plot_all_scaling_max_abs_scaler_section:
#
# MaxAbsScaler
# ------------
#
# :class:`~sklearn.preprocessing.MaxAbsScaler` is similar to
# :class:`~sklearn.preprocessing.MinMaxScaler` except that the
# values are mapped across several ranges depending on whether negative
# OR positive values are present. If only positive values are present, the
# range is [0, 1]. If only negative values are present, the range is [-1, 0].
# If both negative and positive values are present, the range is [-1, 1].
# On positive only data, both :class:`~sklearn.preprocessing.MinMaxScaler`
# and :class:`~sklearn.preprocessing.MaxAbsScaler` behave similarly.
# :class:`~sklearn.preprocessing.MaxAbsScaler` therefore also suffers from
# the presence of large outliers.
make_plot(3)

# %%
# .. _plot_all_scaling_robust_scaler_section:
#
# RobustScaler
# ------------
#
# Unlike the previous scalers, the centering and scaling statistics of
# :class:`~sklearn.preprocessing.RobustScaler`
# are based on percentiles and are therefore not influenced by a small
# number of very large marginal outliers. Consequently, the resulting range of
# the transformed feature values is larger than for the previous scalers and,
# more importantly, are approximately similar: for both features most of the
# transformed values lie in a [-2, 3] range as seen in the zoomed-in figure.
# Note that the outliers themselves are still present in the transformed data.
# If a separate outlier clipping is desirable, a non-linear transformation is
# required (see below).
make_plot(4)

# %%
# .. _plot_all_scaling_power_transformer_section:
#
# PowerTransformer
# ----------------
#
# :class:`~sklearn.preprocessing.PowerTransformer` applies a power
# transformation to each feature to make the data more Gaussian-like in order
# to stabilize variance and minimize skewness. Currently the Yeo-Johnson
# and Box-Cox transforms are supported and the optimal
# scaling factor is determined via maximum likelihood estimation in both
# methods. By default, :class:`~sklearn.preprocessing.PowerTransformer` applies
# zero-mean, unit variance normalization. Note that
# Box-Cox can only be applied to strictly positive data. Income and average
# 绘制图表，显示 5 号数据的可视化结果
make_plot(5)
# 绘制图表，显示 6 号数据的可视化结果
make_plot(6)

# %%
# .. _plot_all_scaling_quantile_transformer_section:
#
# QuantileTransformer (uniform output)
# ------------------------------------
#
# :class:`~sklearn.preprocessing.QuantileTransformer` 应用非线性变换，
# 使得每个特征的概率密度函数被映射到均匀分布或高斯分布。在此情况下，
# 所有数据（包括异常值）将被映射到范围 [0, 1] 的均匀分布，使得异常值与非异常值难以区分。
#
# :class:`~sklearn.preprocessing.RobustScaler` 和
# :class:`~sklearn.preprocessing.QuantileTransformer` 对异常值具有鲁棒性，
# 添加或移除训练集中的异常值将产生近似相同的变换效果。但与
# :class:`~sklearn.preprocessing.RobustScaler` 不同的是，
# :class:`~sklearn.preprocessing.QuantileTransformer` 会自动将任何异常值折叠到
# 预先定义的范围边界（0 和 1），这可能导致极端值的饱和伪影。

make_plot(7)

##############################################################################
# QuantileTransformer (Gaussian output)
# -------------------------------------
#
# 若要映射到高斯分布，请设置参数 ``output_distribution='normal'``。

make_plot(8)

# %%
# .. _plot_all_scaling_normalizer_section:
#
# Normalizer
# ----------
#
# :class:`~sklearn.preprocessing.Normalizer` 将每个样本的向量重新缩放为单位范数，
# 独立于样本的分布。可以在下面的图中看到所有样本都被映射到单位圆上。
# 在我们的示例中，所选特征只有正值；因此，转换后的数据仅位于正象限。
# 如果一些原始特征具有正值和负值的混合，则情况将不同。

make_plot(9)

plt.show()
```