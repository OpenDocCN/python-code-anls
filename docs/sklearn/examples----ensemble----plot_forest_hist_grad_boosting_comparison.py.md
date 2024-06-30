# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_forest_hist_grad_boosting_comparison.py`

```
"""
===============================================================
Comparing Random Forests and Histogram Gradient Boosting models
===============================================================

In this example we compare the performance of Random Forest (RF) and Histogram
Gradient Boosting (HGBT) models in terms of score and computation time for a
regression dataset, though **all the concepts here presented apply to
classification as well**.

The comparison is made by varying the parameters that control the number of
trees according to each estimator:

- `n_estimators` controls the number of trees in the forest. It's a fixed number.
- `max_iter` is the maximum number of iterations in a gradient boosting
  based model. The number of iterations corresponds to the number of trees for
  regression and binary classification problems. Furthermore, the actual number
  of trees required by the model depends on the stopping criteria.

HGBT uses gradient boosting to iteratively improve the model's performance by
fitting each tree to the negative gradient of the loss function with respect to
the predicted value. RFs, on the other hand, are based on bagging and use a
majority vote to predict the outcome.

See the :ref:`User Guide <ensemble>` for more information on ensemble models or
see :ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py` for an
example showcasing some other features of HGBT models.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Load dataset
# ------------

# 导入加利福尼亚房价数据集
from sklearn.datasets import fetch_california_housing

# 加载数据集并返回特征矩阵 X 和目标向量 y，以DataFrame形式返回
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# 获取样本数和特征数
n_samples, n_features = X.shape

# %%
# HGBT uses a histogram-based algorithm on binned feature values that can
# efficiently handle large datasets (tens of thousands of samples or more) with
# a high number of features (see :ref:`Why_it's_faster`). The scikit-learn
# implementation of RF does not use binning and relies on exact splitting, which
# can be computationally expensive.

# 打印数据集包含的样本数和特征数
print(f"The dataset consists of {n_samples} samples and {n_features} features")

# %%
# Compute score and computation times
# -----------------------------------
#
# Notice that many parts of the implementation of
# :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` are parallelized by
# default.
#
# The implementation of :class:`~sklearn.ensemble.RandomForestRegressor` and
# :class:`~sklearn.ensemble.RandomForestClassifier` can also be run on multiple
# cores by using the `n_jobs` parameter, here set to match the number of
# physical cores on the host machine. See :ref:`parallelism` for more
# information.

# 导入 joblib 库
import joblib

# 获取物理核心数
N_CORES = joblib.cpu_count(only_physical_cores=True)
# 打印物理核心数
print(f"Number of physical cores: {N_CORES}")

# %%
# Unlike RF, HGBT models offer an early-stopping option (see
# :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_early_stopping.py`)
# 导入 pandas 库，用于数据处理和分析
import pandas as pd

# 导入随机森林回归器和直方图梯度提升回归器
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
# 导入网格搜索交叉验证和 K 折交叉验证
from sklearn.model_selection import GridSearchCV, KFold

# 定义随机森林和直方图梯度提升回归器的初始参数
models = {
    "Random Forest": RandomForestRegressor(
        min_samples_leaf=5, random_state=0, n_jobs=N_CORES
    ),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_leaf_nodes=15, random_state=0, early_stopping=False
    ),
}

# 定义随机森林和直方图梯度提升回归器的参数网格
param_grids = {
    "Random Forest": {"n_estimators": [10, 20, 50, 100]},
    "Hist Gradient Boosting": {"max_iter": [10, 20, 50, 100, 300, 500]},
}

# 定义 K 折交叉验证对象
cv = KFold(n_splits=4, shuffle=True, random_state=0)

# 初始化结果列表
results = []

# 遍历每个模型及其对应的参数
for name, model in models.items():
    # 使用网格搜索交叉验证对模型进行参数调优
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        return_train_score=True,
        cv=cv,
    ).fit(X, y)
    # 将交叉验证的结果存入结果字典中
    result = {"model": name, "cv_results": pd.DataFrame(grid_search.cv_results_)}
    results.append(result)

# %%
# .. Note::
#   为了避免添加不必要的新树，该算法在每次添加树时使用一个样本外集合来计算模型的泛化性能。
#   因此，如果泛化性能在超过 `n_iter_no_change` 次迭代中没有改善，则停止添加树。
#   
#   其他模型的参数已经调优，但为了简化示例，这里没有显示调优的过程。

# 导入绘图所需的 plotly 库的模块
import plotly.colors as colors
import plotly.express as px
from plotly.subplots import make_subplots

# 创建子图对象，包含两个子图，共享 y 轴，分别显示训练时间与分数，预测时间与分数
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=["Train time vs score", "Predict time vs score"],
)

# 获取模型名称列表和颜色列表
model_names = [result["model"] for result in results]
colors_list = colors.qualitative.Plotly * (
    len(model_names) // len(colors.qualitative.Plotly) + 1
)

# 遍历结果列表，生成散点图并添加到子图中
for idx, result in enumerate(results):
    # 获取当前模型的交叉验证结果
    cv_results = result["cv_results"].round(3)
    model_name = result["model"]
    param_name = list(param_grids[model_name].keys())[0]
    cv_results[param_name] = cv_results["param_" + param_name]
    cv_results["model"] = model_name

    # 创建散点图对象
    scatter_fig = px.scatter(
        cv_results,
        x="mean_fit_time",
        y="mean_test_score",
        error_x="std_fit_time",
        error_y="std_test_score",
        hover_data=param_name,
        color="model",
    )
    # 创建折线图，用于展示交叉验证结果中的平均拟合时间与平均测试分数的关系
    line_fig = px.line(
        cv_results,
        x="mean_fit_time",   # x轴数据为平均拟合时间
        y="mean_test_score", # y轴数据为平均测试分数
    )
    
    # 从散点图对象中提取第一个数据点的轨迹
    scatter_trace = scatter_fig["data"][0]
    # 从折线图对象中提取第一个数据点的轨迹
    line_trace = line_fig["data"][0]
    # 更新散点图轨迹的标记属性，设置颜色为给定的颜色列表中的第idx个颜色
    scatter_trace.update(marker=dict(color=colors_list[idx]))
    # 更新折线图轨迹的线条属性，设置颜色为给定的颜色列表中的第idx个颜色
    line_trace.update(line=dict(color=colors_list[idx]))
    # 将散点图轨迹添加到图形中，位置为第1行第1列
    fig.add_trace(scatter_trace, row=1, col=1)
    # 将折线图轨迹添加到图形中，位置为第1行第1列
    fig.add_trace(line_trace, row=1, col=1)
    
    # 创建散点图，用于展示交叉验证结果中的平均评分时间与平均测试分数的关系，并包含误差条和悬停数据
    scatter_fig = px.scatter(
        cv_results,
        x="mean_score_time",     # x轴数据为平均评分时间
        y="mean_test_score",     # y轴数据为平均测试分数
        error_x="std_score_time", # x轴误差为评分时间的标准差
        error_y="std_test_score", # y轴误差为测试分数的标准差
        hover_data=param_name,    # 悬停数据为给定的参数名称列表
    )
    # 创建折线图，用于展示交叉验证结果中的平均评分时间与平均测试分数的关系
    line_fig = px.line(
        cv_results,
        x="mean_score_time",   # x轴数据为平均评分时间
        y="mean_test_score",   # y轴数据为平均测试分数
    )
    
    # 从散点图对象中提取第一个数据点的轨迹
    scatter_trace = scatter_fig["data"][0]
    # 从折线图对象中提取第一个数据点的轨迹
    line_trace = line_fig["data"][0]
    # 更新散点图轨迹的标记属性，设置颜色为给定的颜色列表中的第idx个颜色
    scatter_trace.update(marker=dict(color=colors_list[idx]))
    # 更新折线图轨迹的线条属性，设置颜色为给定的颜色列表中的第idx个颜色
    line_trace.update(line=dict(color=colors_list[idx]))
    # 将散点图轨迹添加到图形中，位置为第1行第2列
    fig.add_trace(scatter_trace, row=1, col=2)
    # 将折线图轨迹添加到图形中，位置为第1行第2列
    fig.add_trace(line_trace, row=1, col=2)
# 更新图表布局，设置 x 轴标题为 "Train time (s) - lower is better"
# 设置 y 轴标题为 "Test R2 score - higher is better"
# 设置第二个 x 轴的标题为 "Predict time (s) - lower is better"
# 设置图例的位置和样式，包括 x 和 y 方向上的位置，轨迹顺序为 "normal"，边框宽度为 1
# 设置图表标题，位于 x 轴的中间，标题文本为 "Speed-score trade-off of tree-based ensembles"
fig.update_layout(
    xaxis=dict(title="Train time (s) - lower is better"),
    yaxis=dict(title="Test R2 score - higher is better"),
    xaxis2=dict(title="Predict time (s) - lower is better"),
    legend=dict(x=0.72, y=0.05, traceorder="normal", borderwidth=1),
    title=dict(x=0.5, text="Speed-score trade-off of tree-based ensembles"),
)

# %%
# 对于 HGBT 和 RF 模型，在集成中增加树的数量会改善它们的表现。
# 然而，评分会达到一个平台期，在此阶段继续增加树的数量只会导致拟合和评分变慢。
# RF 模型会比 HGBDT 模型更早地达到这种平台，并且永远无法达到最大 HGBDT 模型的测试分数。
#
# 注意，上图显示的结果可能在不同运行中略有变化，甚至在其他计算机上运行时会有更显著的差异：建议在本地机器上运行此示例。
#
# 总体来说，通常会观察到基于直方图的梯度提升模型在 "测试分数 vs 训练速度折衷" 中普遍优于随机森林模型
# （HGBDT 曲线应位于 RF 曲线的左上方，不会交叉）。在 "测试分数 vs 预测速度折衷" 中，争议可能更大，
# 但通常也更有利于 HGBDT。始终建议使用超参数调整来检查这两种模型，并比较它们在特定问题上的性能，
# 以确定哪种模型最合适，**但 HGBT 几乎总是提供比 RF 更有利的速度 - 准确度折衷**，无论是使用默认超参数还是包括超参数调整成本。
#
# 不过，有一个例外需要注意：当训练具有大量可能类别的多类分类模型时，HGBDT 每次增强迭代内部适配一个类别的一棵树，
# 而 RF 模型使用的树是自然多类别的，这在某些情况下应该提高 RF 模型的速度 - 准确度折衷。
```