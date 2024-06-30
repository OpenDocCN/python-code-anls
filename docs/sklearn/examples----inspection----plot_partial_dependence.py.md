# `D:\src\scipysrc\scikit-learn\examples\inspection\plot_partial_dependence.py`

```
# %%
# Bike sharing dataset preprocessing
# ----------------------------------
#
# We will use the bike sharing dataset. The goal is to predict the number of bike
# rentals using weather and season data as well as the datetime information.
from sklearn.datasets import fetch_openml

# 从 OpenML 获取自行车共享数据集的版本 2，并将其作为数据框格式加载
bikes = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
# 显式复制数据以避免 pandas 的 "SettingWithCopyWarning"
X, y = bikes.data.copy(), bikes.target

# 我们仅使用数据的子集来加快示例的速度
X = X.iloc[::5, :]
y = y[::5]

# %%
# The feature `"weather"` has a particularity: the category `"heavy_rain"` is a rare
# category.
# 特征 "weather" 有一个特殊性：类别 "heavy_rain" 是一个罕见的类别。
X["weather"].value_counts()

# %%
# Because of this rare category, we collapse it into `"rain"`.
# 因为存在罕见类别，我们将其合并为 "rain"。
X["weather"] = (
    X["weather"]
    .astype(object)  # 将数据类型转换为对象类型
    .replace(to_replace="heavy_rain", value="rain")  # 替换 "heavy_rain" 为 "rain"
    .astype("category")  # 将数据类型再次转换为分类类型
)

# %%
# We now have a closer look at the `"year"` feature:
# 现在我们仔细查看 "year" 特征：
X["year"].value_counts()

# %%
# We see that we have data from two years. We use the first year to train the
# model and the second year to test the model.
# 我们看到数据来自两个不同的年份。我们将第一年的数据用于训练模型，第二年的数据用于测试模型。
mask_training = X["year"] == 0.0
# 删除列"year"，更新数据集X
X = X.drop(columns=["year"])

# 根据训练数据掩码，将数据集X和目标y分割为训练集和测试集
X_train, y_train = X[mask_training], y[mask_training]
X_test, y_test = X[~mask_training], y[~mask_training]

# %%
# 我们可以查看数据集的信息，以了解数据类型的异质性。我们需要相应地预处理不同的列。
X_train.info()

# %%
# 根据前面的信息，我们将考虑“category”列作为名义分类特征。此外，我们还将日期和时间信息视为分类特征。
#
# 手动定义包含数值特征和分类特征的列。
numerical_features = [
    "temp",
    "feel_temp",
    "humidity",
    "windspeed",
]
categorical_features = X_train.columns.drop(numerical_features)

# %%
# 在深入研究不同机器学习流水线的预处理细节之前，我们将尝试获取有关数据集的一些额外直觉，这些直觉将有助于理解模型的统计性能和偏差分析的结果。
#
# 我们通过按季节和年份分组的方式绘制平均自行车租赁数量。
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

days = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
hours = tuple(range(24))
xticklabels = [f"{day}\n{hour}:00" for day, hour in product(days, hours)]
xtick_start, xtick_period = 6, 12

fig, axs = plt.subplots(nrows=2, figsize=(8, 6), sharey=True, sharex=True)
# 计算每小时平均自行车租赁数量，并按年份和季节分组
average_bike_rentals = bikes.frame.groupby(
    ["year", "season", "weekday", "hour"], observed=True
).mean(numeric_only=True)["count"]
# 根据年份分组绘制图表
for ax, (idx, df) in zip(axs, average_bike_rentals.groupby("year")):
    df.groupby("season", observed=True).plot(ax=ax, legend=True)

    # 装饰图表
    ax.set_xticks(
        np.linspace(
            start=xtick_start,
            stop=len(xticklabels),
            num=len(xticklabels) // xtick_period,
        )
    )
    ax.set_xticklabels(xticklabels[xtick_start::xtick_period])
    ax.set_xlabel("")
    ax.set_ylabel("Average number of bike rentals")
    ax.set_title(
        f"Bike rental for {'2010 (train set)' if idx == 0.0 else '2011 (test set)'}"
    )
    ax.set_ylim(0, 1_000)
    ax.set_xlim(0, len(xticklabels))
    ax.legend(loc=2)

# %%
# 训练集和测试集的显著差异之一是测试集中自行车租赁数量较高。因此，我们可能会得到一个低估自行车租赁数量的机器学习模型。我们还观察到春季的自行车租赁数量较低。
# 此外，在工作日，早上6-7点和下午5-6点有一些自行车租赁的高峰。我们可以记住这些不同的见解，并用它们来理解偏差分析的结果。
#
# 机器学习模型的预处理器
# -------------------------
#
# Since we later use two different models, a
# :class:`~sklearn.neural_network.MLPRegressor` and a
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor`, we create two different
# preprocessors, specific for each model.
#
# Preprocessor for the neural network model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We will use a :class:`~sklearn.preprocessing.QuantileTransformer` to scale the
# numerical features and encode the categorical features with a
# :class:`~sklearn.preprocessing.OneHotEncoder`.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

# 创建用于 MLPRegressor 模型的数据预处理器
mlp_preprocessor = ColumnTransformer(
    transformers=[
        ("num", QuantileTransformer(n_quantiles=100), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
mlp_preprocessor

# %%
# Preprocessor for the gradient boosting model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For the gradient boosting model, we leave the numerical features as-is and only
# encode the categorical features using a
# :class:`~sklearn.preprocessing.OrdinalEncoder`.
from sklearn.preprocessing import OrdinalEncoder

# 创建用于 HistGradientBoostingRegressor 模型的数据预处理器
hgbdt_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(), categorical_features),
        ("num", "passthrough", numerical_features),
    ],
    sparse_threshold=1,
    verbose_feature_names_out=False,
).set_output(transform="pandas")
hgbdt_preprocessor

# %%
# 1-way partial dependence with different models
# ----------------------------------------------
#
# In this section, we will compute 1-way partial dependence with two different
# machine-learning models: (i) a multi-layer perceptron and (ii) a
# gradient-boosting model. With these two models, we illustrate how to compute and
# interpret both partial dependence plot (PDP) for both numerical and categorical
# features and individual conditional expectation (ICE).
#
# Multi-layer perceptron
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Let's fit a :class:`~sklearn.neural_network.MLPRegressor` and compute
# single-variable partial dependence plots.
from time import time

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

print("Training MLPRegressor...")
tic = time()
# 创建 MLPRegressor 模型的管道，包括预处理器和具体的 MLPRegressor 模型设置
mlp_model = make_pipeline(
    mlp_preprocessor,
    MLPRegressor(
        hidden_layer_sizes=(30, 15),
        learning_rate_init=0.01,
        early_stopping=True,
        random_state=0,
    ),
)
mlp_model.fit(X_train, y_train)
print(f"done in {time() - tic:.3f}s")
print(f"Test R2 score: {mlp_model.score(X_test, y_test):.2f}")

# %%
# We configured a pipeline using the preprocessor that we created specifically for the
# neural network and tuned the neural network size and learning rate to get a reasonable
# compromise between training time and predictive performance on a test set.
#
# Importantly, this tabular dataset has very different dynamic ranges for its
# 导入需要的库和模块
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import HistGradientBoostingRegressor
from time import time

# 定义一些共用的参数
common_params = {
    "subsample": 50,
    "n_jobs": 2,
    "grid_resolution": 20,
    "random_state": 0,
}

# 打印信息，计算部分依赖图
print("Computing partial dependence plots...")

# 定义特征的相关信息
features_info = {
    # 感兴趣的特征列表
    "features": ["temp", "humidity", "windspeed", "season", "weather", "hour"],
    # partial dependence plot的类型
    "kind": "average",
    # 关于分类特征的信息
    "categorical_features": categorical_features,
}

# 创建子图用于显示部分依赖
tic = time()
_, ax = plt.subplots(ncols=3, nrows=2, figsize=(9, 8), constrained_layout=True)

# 使用PartialDependenceDisplay从估计器（MLP模型）创建部分依赖展示
display = PartialDependenceDisplay.from_estimator(
    mlp_model,
    X_train,
    **features_info,
    ax=ax,
    **common_params,
)
print(f"done in {time() - tic:.3f}s")

# 设置图标题
_ = display.figure_.suptitle(
    (
        "Partial dependence of the number of bike rentals\n"
        "for the bike rental dataset with an MLPRegressor"
    ),
    fontsize=16,
)

# %%
# Gradient boosting
# ~~~~~~~~~~~~~~~~~
#
# 现在我们来拟合一个HistGradientBoostingRegressor，并计算相同特征的部分依赖。
# 我们还将使用为此模型创建的特定预处理器。

# 打印信息，训练HistGradientBoostingRegressor模型
print("Training HistGradientBoostingRegressor...")
tic = time()

# 创建管道，包含预处理器和HistGradientBoostingRegressor模型
hgbdt_model = make_pipeline(
    hgbdt_preprocessor,
    HistGradientBoostingRegressor(
        categorical_features=categorical_features,
        random_state=0,
        max_iter=50,
    ),
)

# 训练模型
hgbdt_model.fit(X_train, y_train)
print(f"done in {time() - tic:.3f}s")

# 打印测试集的R2分数
print(f"Test R2 score: {hgbdt_model.score(X_test, y_test):.2f}")

# %%
# 这里我们使用了梯度提升模型的默认超参数，没有进行任何预处理，因为基于树的模型在数值特征的单调转换上天生具有鲁棒性。
#
# 注意，在这种表格数据集上，梯度提升机的训练速度更快，且更准确，比神经网络更具优势。调整超参数的成本也更低（默认值通常表现良好，而神经网络则不一定）。
#
# 我们将绘制一些数值和分类特征的部分依赖图。
# 打印信息，指示正在计算偏依赖图
print("Computing partial dependence plots...")
# 记录开始计时
tic = time()
# 创建一个具有3列和2行的子图，用于显示偏依赖图，设置图形大小和布局
_, ax = plt.subplots(ncols=3, nrows=2, figsize=(9, 8), constrained_layout=True)
# 从估计器中生成偏依赖展示对象，使用训练数据集和其他特征信息，将结果绘制在指定的子图上
display = PartialDependenceDisplay.from_estimator(
    hgbdt_model,
    X_train,
    **features_info,
    ax=ax,
    **common_params,
)
# 打印完成计算所需时间
print(f"done in {time() - tic:.3f}s")
# 设置图形的总标题
_ = display.figure_.suptitle(
    (
        "Partial dependence of the number of bike rentals\n"
        "for the bike rental dataset with a gradient boosting"
    ),
    fontsize=16,
)

# %%
# 分析图形的含义
# ~~~~~~~~~~~~~~~~~~~~~
#
# 首先分析数值特征的偏依赖图。无论哪种模型，温度的偏依赖图通常显示出随着温度的升高，自行车租赁数量增加的趋势。
# 湿度特征则展示出相反的趋势，即湿度增加时自行车租赁数量减少。风速特征也显示出类似的趋势，即风速增加时自行车租赁数量减少，
# 这种趋势在两种模型中都观察到。我们还观察到:class:`~sklearn.neural_network.MLPRegressor` 模型比
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` 模型具有更平滑的预测效果。
#
# 接下来，我们将分析分类特征的偏依赖图。
#
# 我们观察到春季是季节特征中租赁量最低的时段。在天气特征中，雨天是租赁量最低的情况。关于小时特征，我们观察到在早上7点和下午6点左右有两个高峰。
# 这些发现与我们早期对数据集的观察一致。
#
# 然而，值得注意的是，如果特征之间存在相关性，我们可能会生成一些毫无意义的合成样本。
#
# ICE vs. PDP
# ~~~~~~~~~~~
# PDP 是特征边际效应的平均值。我们对所提供集合的所有样本的响应进行平均。因此，某些效应可能会被隐藏。
# 针对这一点，我们可以绘制每个单独响应。这种表示被称为个体效应图（ICE）。在下面的图中，我们随机选择了50条温度和湿度特征的ICE线。
print("Computing partial dependence plots and individual conditional expectation...")
# 记录开始计时
tic = time()
# 创建一个包含两列的子图，用于显示ICE和PDP的表示，设置图形大小和共享y轴
_, ax = plt.subplots(ncols=2, figsize=(6, 4), sharey=True, constrained_layout=True)
# 定义特征信息，包括特征名称、显示方式和其他参数
features_info = {
    "features": ["temp", "humidity"],
    "kind": "both",
    "centered": True,
}
# 从估计器中生成偏依赖展示对象，使用训练数据集和特征信息，将结果绘制在指定的子图上
display = PartialDependenceDisplay.from_estimator(
    hgbdt_model,
    X_train,
    **features_info,
    ax=ax,
    **common_params,
)
# 打印完成计算所需时间
print(f"done in {time() - tic:.3f}s")
# 设置图形的总标题
_ = display.figure_.suptitle("ICE and PDP representations", fontsize=16)

# %%
# 我们看到温度特征的ICE图为我们提供了额外信息：
# 一些ICE线是平的，而另一些显示出随着温度超过35摄氏度，依赖性下降的趋势。湿度特征也呈现出类似的模式。
# humidity feature: some of the ICEs lines show a sharp decrease when the humidity is
# above 80%.
#
# Not all ICE lines are parallel, this indicates that the model finds
# interactions between features. We can repeat the experiment by constraining the
# gradient boosting model to not use any interactions between features using the
# parameter `interaction_cst`:
from sklearn.base import clone

# 创建一个新的 interaction_cst 列表，其中包含 X_train 特征数量的单特征子列表
interaction_cst = [[i] for i in range(X_train.shape[1])]
# 克隆 hgbdt_model 并设置参数，限制使用特征间的交互性
hgbdt_model_without_interactions = (
    clone(hgbdt_model)
    .set_params(histgradientboostingregressor__interaction_cst=interaction_cst)
    .fit(X_train, y_train)
)
# 打印不考虑特征交互的模型在测试集上的 R2 分数
print(f"Test R2 score: {hgbdt_model_without_interactions.score(X_test, y_test):.2f}")

# %%
_, ax = plt.subplots(ncols=2, figsize=(6, 4), sharey=True, constrained_layout=True)

# 设定 features_info 中的 centered 键为 False
features_info["centered"] = False
# 使用 PartialDependenceDisplay 从估计器 hgbdt_model_without_interactions 创建显示
display = PartialDependenceDisplay.from_estimator(
    hgbdt_model_without_interactions,
    X_train,
    **features_info,
    ax=ax,
    **common_params,
)
# 设置图形的总标题
_ = display.figure_.suptitle("ICE and PDP representations", fontsize=16)

# %%
# 2D interaction plots
# --------------------
#
# PDPs with two features of interest enable us to visualize interactions among them.
# However, ICEs cannot be plotted in an easy manner and thus interpreted. We will show
# the representation of available in
# :meth:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` that is a 2D
# heatmap.
print("Computing partial dependence plots...")
# 设定 features_info 字典，包括 "temp", "humidity" 和 ("temp", "humidity") 作为特征
features_info = {
    "features": ["temp", "humidity", ("temp", "humidity")],
    "kind": "average",
}
# 创建一个包含三个子图的新图形对象
_, ax = plt.subplots(ncols=3, figsize=(10, 4), constrained_layout=True)
# 计时开始
tic = time()
# 使用 PartialDependenceDisplay 从估计器 hgbdt_model 创建显示
display = PartialDependenceDisplay.from_estimator(
    hgbdt_model,
    X_train,
    **features_info,
    ax=ax,
    **common_params,
)
# 打印计算时间
print(f"done in {time() - tic:.3f}s")
# 设置图形的总标题
_ = display.figure_.suptitle(
    "1-way vs 2-way of numerical PDP using gradient boosting", fontsize=16
)

# %%
# The two-way partial dependence plot shows the dependence of the number of bike rentals
# on joint values of temperature and humidity.
# We clearly see an interaction between the two features. For a temperature higher than
# 20 degrees Celsius, the humidity has a impact on the number of bike rentals
# that seems independent on the temperature.
#
# On the other hand, for temperatures lower than 20 degrees Celsius, both the
# temperature and humidity continuously impact the number of bike rentals.
#
# Furthermore, the slope of the of the impact ridge of the 20 degrees Celsius
# threshold is very dependent on the humidity level: the ridge is steep under
# dry conditions but much smoother under wetter conditions above 70% of humidity.
#
# We now contrast those results with the same plots computed for the model
# constrained to learn a prediction function that does not depend on such
# non-linear feature interactions.
print("Computing partial dependence plots...")
# 设定 features_info 字典，包括 "temp", "humidity" 和 ("temp", "humidity") 作为特征
features_info = {
    "features": ["temp", "humidity", ("temp", "humidity")],
    "kind": "average",


    # 设置键为 "kind" 的值为 "average"
}
_, ax = plt.subplots(ncols=3, figsize=(10, 4), constrained_layout=True)
tic = time()
# 从给定的估计器（模型）创建PartialDependenceDisplay对象，显示1D偏依赖图
# 这些图显示了禁止模拟特征交互的模型中每个特征单独的局部峰值，特别是“humidity”特征。
# 这些峰值可能反映了模型的退化行为，它试图通过过度拟合特定的训练点来补偿禁止的交互。
# 需要注意的是，该模型在测试集上的预测性能显著低于未受约束的原始模型。
display = PartialDependenceDisplay.from_estimator(
    hgbdt_model_without_interactions,
    X_train,
    **features_info,
    ax=ax,
    **common_params,
)
print(f"done in {time() - tic:.3f}s")
_ = display.figure_.suptitle(
    "1-way vs 2-way of numerical PDP using gradient boosting", fontsize=16
)

# %%
# 1D偏依赖图的局部峰值可能受到网格分辨率参数的影响。
#
# 这些局部峰值导致2D偏依赖图呈现出噪声较多的网格化效果。由于湿度特征中高频振荡的存在，
# 很难确定这些特征之间是否存在交互作用。然而，可以清楚地看到，当温度穿过20度的边界时，
# 对这个模型不再可见简单的交互效应。
#
# 对于分类特征之间的偏依赖，将提供一个可以显示为热图的离散表示。例如，季节、天气和目标之间的交互将如下所示：
print("Computing partial dependence plots...")
features_info = {
    "features": ["season", "weather", ("season", "weather")],
    "kind": "average",
    "categorical_features": categorical_features,
}
_, ax = plt.subplots(ncols=3, figsize=(14, 6), constrained_layout=True)
tic = time()
# 从给定的估计器（模型）创建PartialDependenceDisplay对象，显示分类特征的1-way vs 2-way PDP
display = PartialDependenceDisplay.from_estimator(
    hgbdt_model,
    X_train,
    **features_info,
    ax=ax,
    **common_params,
)
print(f"done in {time() - tic:.3f}s")
_ = display.figure_.suptitle(
    "1-way vs 2-way PDP of categorical features using gradient boosting", fontsize=16
)

# %%
# 3D表示
# ~~~~~~~~~~~~~~~~~
#
# 让我们使用三维进行相同的偏依赖图，这次是两个特征的交互。
# 用于在matplotlib < 3.2中进行三维投影的未使用但必需的导入
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np

from sklearn.inspection import partial_dependence

fig = plt.figure(figsize=(5.5, 5))

features = ("temp", "humidity")
# 计算部分依赖，显示两个特征的平均效果，网格分辨率为10
pdp = partial_dependence(
    hgbdt_model, X_train, features=features, kind="average", grid_resolution=10
)
XX, YY = np.meshgrid(pdp["grid_values"][0], pdp["grid_values"][1])
Z = pdp.average[0].T
ax = fig.add_subplot(projection="3d")
fig.add_axes(ax)
# 使用给定的数据生成三维表面图，并将返回的表面对象存储在 surf 变量中
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor="k")

# 设置 x 轴的标签为 features 列表的第一个元素
ax.set_xlabel(features[0])

# 设置 y 轴的标签为 features 列表的第二个元素
ax.set_ylabel(features[1])

# 设置整个图表的标题，描述了基于温度和湿度 GBDT 模型的自行车租赁数量的偏依赖图
fig.suptitle(
    "PD of number of bike rentals on\nthe temperature and humidity GBDT model",
    fontsize=16,
)

# 设置 3D 图的初始视角，仰角为 22 度，方位角为 122 度
ax.view_init(elev=22, azim=122)

# 添加一个颜色条到图表中，与 surf 对象相关联，设置颜色条的位置、缩放和长宽比
clb = plt.colorbar(surf, pad=0.08, shrink=0.6, aspect=10)

# 设置颜色条的标题
clb.ax.set_title("Partial\ndependence")

# 显示整个图表
plt.show()
```