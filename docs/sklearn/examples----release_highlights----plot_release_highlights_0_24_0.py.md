# `D:\src\scipysrc\scikit-learn\examples\release_highlights\plot_release_highlights_0_24_0.py`

```
# ruff: noqa
"""
========================================
Release Highlights for scikit-learn 0.24
========================================

.. currentmodule:: sklearn

We are pleased to announce the release of scikit-learn 0.24! Many bug fixes
and improvements were added, as well as some new key features. We detail
below a few of the major features of this release. **For an exhaustive list of
all the changes**, please refer to the :ref:`release notes <release_notes_0_24>`.

To install the latest version (with pip)::

    pip install --upgrade scikit-learn

or with conda::

    conda install -c conda-forge scikit-learn

"""

##############################################################################
# Successive Halving estimators for tuning hyper-parameters
# ---------------------------------------------------------
# Successive Halving, a state of the art method, is now available to
# explore the space of the parameters and identify their best combination.
# :class:`~sklearn.model_selection.HalvingGridSearchCV` and
# :class:`~sklearn.model_selection.HalvingRandomSearchCV` can be
# used as drop-in replacement for
# :class:`~sklearn.model_selection.GridSearchCV` and
# :class:`~sklearn.model_selection.RandomizedSearchCV`.
# Successive Halving is an iterative selection process illustrated in the
# figure below. The first iteration is run with a small amount of resources,
# where the resource typically corresponds to the number of training samples,
# but can also be an arbitrary integer parameter such as `n_estimators` in a
# random forest. Only a subset of the parameter candidates are selected for the
# next iteration, which will be run with an increasing amount of allocated
# resources. Only a subset of candidates will last until the end of the
# iteration process, and the best parameter candidate is the one that has the
# highest score on the last iteration.
#
# Read more in the :ref:`User Guide <successive_halving_user_guide>` (note:
# the Successive Halving estimators are still :term:`experimental
# <experimental>`).
#
# .. figure:: ../model_selection/images/sphx_glr_plot_successive_halving_iterations_001.png
#   :target: ../model_selection/plot_successive_halving_iterations.html
#   :align: center

import numpy as np                           # 导入NumPy库，用于数值计算
from scipy.stats import randint              # 从SciPy库中导入randint函数，用于生成随机整数
from sklearn.experimental import enable_halving_search_cv  # 启用Successive Halving CV的实验性支持
from sklearn.model_selection import HalvingRandomSearchCV  # 导入Successive Halving Random Search CV
from sklearn.ensemble import RandomForestClassifier         # 导入随机森林分类器
from sklearn.datasets import make_classification            # 导入数据生成工具

rng = np.random.RandomState(0)              # 创建随机数生成器对象rng，用于复现随机结果

X, y = make_classification(n_samples=700, random_state=rng)  # 生成分类数据集X和标签y

clf = RandomForestClassifier(n_estimators=10, random_state=rng)  # 创建随机森林分类器对象clf

param_dist = {                             # 定义参数分布字典param_dist，包含随机森林的各种参数选择
    "max_depth": [3, None],                # 决策树的最大深度参数选择
    "max_features": randint(1, 11),        # 最大特征数的随机选择范围
    "min_samples_split": randint(2, 11),   # 内部节点再划分所需最小样本数的随机选择范围
    "bootstrap": [True, False],            # 是否进行bootstrap的布尔参数选择
    "criterion": ["gini", "entropy"],      # 分裂标准的选择：基尼系数或信息增益
}

rsh = HalvingRandomSearchCV(                # 创建Successive Halving Random Search CV对象rsh
    estimator=clf,                         # 使用随机森林分类器clf作为估计器
    param_distributions=param_dist,         # 使用定义好的参数分布字典param_dist
    factor=2,                              # Successive Halving算法的资源分配因子
    random_state=rng                        # 随机状态对象rng，用于复现随机结果
)
rsh.fit(X, y)                              # 在数据集X, y上拟合rsh模型
# 输出 rsh 对象的 best_params_ 属性，该属性包含 HistGradientBoosting 模型的最佳参数
rsh.best_params_

##############################################################################
# HistGradientBoosting 估计器对分类特征的本地支持
# --------------------------------------------------------------------------
# :class:`~sklearn.ensemble.HistGradientBoostingClassifier` 和
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` 现在本地支持分类特征：
# 它们可以考虑在非有序的分类数据上进行分割。在 :ref:`用户指南
# <categorical_support_gbdt>` 中了解更多。
#
# .. figure:: ../ensemble/images/sphx_glr_plot_gradient_boosting_categorical_001.png
#   :target: ../ensemble/plot_gradient_boosting_categorical.html
#   :align: center
#
# 上图显示，对分类特征的新本地支持导致拟合时间与将类别视为有序量（简单的序数编码）的模型相当。
# 本地支持也比独热编码和序数编码更具表达性。然而，要使用新的 `categorical_features` 参数，
# 仍然需要像这个 :ref:`示例中演示的那样，在管道中预处理数据 <sphx_glr_auto_examples_ensemble_plot_gradient_boosting_categorical.py>`。

##############################################################################
# HistGradientBoosting 估计器的性能改进
# --------------------------------------------------------
# 在调用 `fit` 时，:class:`ensemble.HistGradientBoostingRegressor` 和
# :class:`ensemble.HistGradientBoostingClassifier` 的内存占用显著改进。
# 此外，直方图初始化现在是并行进行的，这导致轻微的速度改进。
# 在 `基准页面 <https://scikit-learn.org/scikit-learn-benchmarks/>`_ 中了解更多信息。

##############################################################################
# 新的自训练元估计器
# --------------------------------
# 基于 `Yarowski's algorithm <https://doi.org/10.3115/981658.981684>`_ 实现了
# 一种新的自训练方法，现在可以与任何实现 :term:`predict_proba` 的分类器一起使用。
# 子分类器将表现为半监督分类器，允许其从未标记的数据中学习。
# 在 :ref:`用户指南 <self_training>` 中了解更多信息。

import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

rng = np.random.RandomState(42)
iris = datasets.load_iris()
random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.3
iris.target[random_unlabeled_points] = -1
svc = SVC(probability=True, gamma="auto")
self_training_model = SelfTrainingClassifier(svc)
self_training_model.fit(iris.data, iris.target)

##############################################################################
# 新的 SequentialFeatureSelector 转换器
# -----------------------------------------
# 导入所需的类和函数
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# 加载鸢尾花数据集，返回特征矩阵 X 和目标向量 y
X, y = load_iris(return_X_y=True, as_frame=True)
# 获取特征列的名称
feature_names = X.columns
# 创建 KNN 分类器对象，设置邻居数为 3
knn = KNeighborsClassifier(n_neighbors=3)
# 创建顺序特征选择器对象，选择要保留的特征数为 2
sfs = SequentialFeatureSelector(knn, n_features_to_select=2)
# 在数据集上拟合顺序特征选择器
sfs.fit(X, y)
# 打印通过顺序特征选择器选择的特征名列表
print(
    "Features selected by forward sequential selection: "
    f"{feature_names[sfs.get_support()].tolist()}"
)

##############################################################################
# 新的 PolynomialCountSketch 核近似函数
# ---------------------------------------
# 新的 PolynomialCountSketch 类用于在线性模型中近似特征空间的多项式展开，
# 但比 PolynomialFeatures 使用更少的内存。

from sklearn.datasets import fetch_covtype
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import LogisticRegression

# 加载森林类型数据集，返回特征矩阵 X 和目标向量 y
X, y = fetch_covtype(return_X_y=True)
# 创建处理流水线，包括数据归一化、多项式 CountSketch 核近似和逻辑回归
pipe = make_pipeline(
    MinMaxScaler(),
    PolynomialCountSketch(degree=2, n_components=300),
    LogisticRegression(max_iter=1000),
)
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=5000, test_size=10000, random_state=42
)
# 在训练集上拟合处理流水线，并计算在测试集上的得分
pipe.fit(X_train, y_train).score(X_test, y_test)

##############################################################################
# 作为对比，这里是相同数据的线性基准分数：

linear_baseline = make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=1000))
# 在训练集上拟合线性基准模型，并计算在测试集上的得分
linear_baseline.fit(X_train, y_train).score(X_test, y_test)

##############################################################################
# 个体条件期望（ICE）图
# --------------------------
# 新的偏依赖图形式：个体条件期望（ICE）图。ICE 图可视化每个样本中预测对一个特征的依赖关系，
# 每个样本对应一条曲线。

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# 从加州住房数据集中获取特征矩阵 X 和目标向量 y
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
features = ["MedInc", "AveOccup", "HouseAge", "AveRooms"]
# 定义要进行偏依赖分析的特征列表

est = RandomForestRegressor(n_estimators=10)
# 使用随机森林回归器，设定决策树数量为10

est.fit(X, y)
# 使用训练数据集 X 和目标数据 y 来拟合随机森林回归器

# plot_partial_dependence has been removed in version 1.2. From 1.2, use
# PartialDependenceDisplay instead.
# display = plot_partial_dependence(
# 绘制偏依赖图已在1.2版本中移除，从1.2版本开始，请使用PartialDependenceDisplay代替。

display = PartialDependenceDisplay.from_estimator(
    est,
    X,
    features,
    kind="individual",
    subsample=50,
    n_jobs=3,
    grid_resolution=20,
    random_state=0,
)
# 使用随机森林回归器 est 计算个体偏依赖，显示部分依赖性图表
# 参数 kind="individual" 表示计算个体特征的偏依赖性
# subsample=50 表示子样本大小为50
# n_jobs=3 表示使用3个并行作业
# grid_resolution=20 表示网格分辨率为20
# random_state=0 表示随机数生成器的种子值设为0

display.figure_.suptitle(
    "Partial dependence of house value on non-location features\n"
    "for the California housing dataset, with BayesianRidge"
)
# 设置图表的总标题，显示房屋价值对非位置特征的偏依赖性
# 用于加利福尼亚房屋数据集，使用贝叶斯岭回归

display.figure_.subplots_adjust(hspace=0.3)
# 调整图表布局，设置水平间距为0.3

##############################################################################
# New Poisson splitting criterion for DecisionTreeRegressor
# ---------------------------------------------------------
# The integration of Poisson regression estimation continues from version 0.23.
# :class:`~sklearn.tree.DecisionTreeRegressor` now supports a new `'poisson'`
# splitting criterion. Setting `criterion="poisson"` might be a good choice
# if your target is a count or a frequency.
# 决策树回归器 `DecisionTreeRegressor` 现在支持新的 `'poisson'` 分割标准。
# 如果目标是计数或频率，则设置 `criterion="poisson"` 可能是一个不错的选择。

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np

n_samples, n_features = 1000, 20
# 设定样本数为1000，特征数为20

rng = np.random.RandomState(0)
X = rng.randn(n_samples, n_features)
# 用随机数生成器生成服从标准正态分布的数据矩阵 X

# positive integer target correlated with X[:, 5] with many zeros:
y = rng.poisson(lam=np.exp(X[:, 5]) / 2)
# 生成与 X[:, 5] 相关且有许多零的正整数目标数据 y

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
# 将数据集 X 和目标数据 y 分割为训练集和测试集

regressor = DecisionTreeRegressor(criterion="poisson", random_state=0)
# 使用 Poisson 分割标准初始化决策树回归器 regressor

regressor.fit(X_train, y_train)
# 使用训练数据集 X_train 和目标数据 y_train 拟合决策树回归器

##############################################################################
# New documentation improvements
# ------------------------------
#
# New examples and documentation pages have been added, in a continuous effort
# to improve the understanding of machine learning practices:
#
# - a new section about :ref:`common pitfalls and recommended
#   practices <common_pitfalls>`,
# - an example illustrating how to :ref:`statistically compare the performance of
#   models <sphx_glr_auto_examples_model_selection_plot_grid_search_stats.py>`
#   evaluated using :class:`~sklearn.model_selection.GridSearchCV`,
# - an example on how to :ref:`interpret coefficients of linear models
#   <sphx_glr_auto_examples_inspection_plot_linear_model_coefficient_interpretation.py>`,
# - an :ref:`example
#   <sphx_glr_auto_examples_cross_decomposition_plot_pcr_vs_pls.py>`
#   comparing Principal Component Regression and Partial Least Squares.
# 新增示例和文档页面，持续努力改进对机器学习实践的理解：
# - 新增关于 :ref:`common pitfalls and recommended practices <common_pitfalls>` 的章节，
# - 一个示例，展示如何使用 :class:`~sklearn.model_selection.GridSearchCV` 来 :ref:`statistically compare the performance of models <sphx_glr_auto_examples_model_selection_plot_grid_search_stats.py>`，
# - 一个示例，说明如何 :ref:`interpret coefficients of linear models <sphx_glr_auto_examples_inspection_plot_linear_model_coefficient_interpretation.py>`，
# - 一个 :ref:`example <sphx_glr_auto_examples_cross_decomposition_plot_pcr_vs_pls.py>` 比较主成分回归和偏最小二乘法。
```