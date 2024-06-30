# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_likelihood_ratios.py`

```
"""
=============================================================
Class Likelihood Ratios to measure classification performance
=============================================================

This example demonstrates the :func:`~sklearn.metrics.class_likelihood_ratios`
function, which computes the positive and negative likelihood ratios (`LR+`,
`LR-`) to assess the predictive power of a binary classifier. As we will see,
these metrics are independent of the proportion between classes in the test set,
which makes them very useful when the available data for a study has a different
class proportion than the target application.

A typical use is a case-control study in medicine, which has nearly balanced
classes while the general population has large class imbalance. In such
application, the pre-test probability of an individual having the target
condition can be chosen to be the prevalence, i.e. the proportion of a
particular population found to be affected by a medical condition. The post-test
probabilities represent then the probability that the condition is truly present
given a positive test result.

In this example we first discuss the link between pre-test and post-test odds
given by the :ref:`class_likelihood_ratios`. Then we evaluate their behavior in
some controlled scenarios. In the last section we plot them as a function of the
prevalence of the positive class.

"""

# Authors:  Arturo Amor <david-arturo.amor-quiroz@inria.fr>
#           Olivier Grisel <olivier.grisel@ensta.org>
# %%
# Pre-test vs. post-test analysis
# ===============================
#
# Suppose we have a population of subjects with physiological measurements `X`
# that can hopefully serve as indirect bio-markers of the disease and actual
# disease indicators `y` (ground truth). Most of the people in the population do
# not carry the disease but a minority (in this case around 10%) does:

# 导入必要的库来生成模拟数据集
from sklearn.datasets import make_classification

# 生成带有类别不平衡的二分类数据集，其中正类的权重为10%
X, y = make_classification(n_samples=10_000, weights=[0.9, 0.1], random_state=0)
print(f"Percentage of people carrying the disease: {100*y.mean():.2f}%")

# %%
# A machine learning model is built to diagnose if a person with some given
# physiological measurements is likely to carry the disease of interest. To
# evaluate the model, we need to assess its performance on a held-out test set:

# 导入数据集分割函数
from sklearn.model_selection import train_test_split

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
# Then we can fit our diagnosis model and compute the positive likelihood
# ratio to evaluate the usefulness of this classifier as a disease diagnosis
# tool:

# 导入逻辑回归模型和类别似然比计算函数
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import class_likelihood_ratios

# 使用逻辑回归拟合模型
estimator = LogisticRegression().fit(X_train, y_train)

# 对测试集进行预测
y_pred = estimator.predict(X_test)

# 计算正类别和负类别的似然比
pos_LR, neg_LR = class_likelihood_ratios(y_test, y_pred)
print(f"LR+: {pos_LR:.3f}")

# %%
# Since the positive class likelihood ratio is much larger than 1.0, it means
# 导入 pandas 库，用于数据处理和分析
import pandas as pd


# 定义评分函数 scoring，用于评估模型在数据集上的表现
def scoring(estimator, X, y):
    # 使用模型预测结果
    y_pred = estimator.predict(X)
    # 计算正类和负类的类别似然比，并关闭警告信息
    pos_lr, neg_lr = class_likelihood_ratios(y, y_pred, raise_warning=False)
    # 返回包含正类和负类似然比的字典
    return {"positive_likelihood_ratio": pos_lr, "negative_likelihood_ratio": neg_lr}


# 定义 extract_score 函数，用于从交叉验证结果中提取类别似然比的平均值和标准差
def extract_score(cv_results):
    # 创建 pandas DataFrame 存储交叉验证结果中的正类和负类似然比
    lr = pd.DataFrame(
        {
            "positive": cv_results["test_positive_likelihood_ratio"],
            "negative": cv_results["test_negative_likelihood_ratio"],
        }
    )
    # 返回正类和负类似然比的平均值和标准差
    return lr.aggregate(["mean", "std"])


# %%
# 首先使用默认超参数验证 LogisticRegression 模型的效果，与前一节相同。

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

# 创建 LogisticRegression 的实例作为评估器
estimator = LogisticRegression()
# 执行交叉验证，并提取评估结果的类别似然比分数
extract_score(cross_validate(estimator, X, y, scoring=scoring, cv=10))

# %%
# 确认模型的有效性：后验几率比前验几率大 12 至 20 倍之间。
#
# 相反，考虑一个虚拟模型，其输出随机预测，与训练集中平均疾病患病率的相似概率：

from sklearn.dummy import DummyClassifier

# 创建 DummyClassifier 的实例，使用分层策略和随机种子1234
estimator = DummyClassifier(strategy="stratified", random_state=1234)
# 执行交叉验证，并提取评估结果的类别似然比分数
extract_score(cross_validate(estimator, X, y, scoring=scoring, cv=10))

# %%
# 在这种情况下，正负类别似然比均接近于1.0，使得该分类器无法作为改进疾病检测的诊断工具。
#
# 另一种虚拟模型选择总是预测最频繁的类别，即“无疾病”。

# 创建 DummyClassifier 的实例，使用最频繁策略
estimator = DummyClassifier(strategy="most_frequent")
# 执行交叉验证，并提取评估结果的类别似然比分数
extract_score(cross_validate(estimator, X, y, scoring=scoring, cv=10))

# %%
# 正预测的缺失意味着没有真正的阳性或假阳性，导致未定义的 `LR+`，绝不应将其解释为无限的 `LR+`
# （分类器完全识别阳性病例）。在这种情况下，`LR-` 的值有助于排除此模型。
#
# 当交叉验证高度不平衡数据且样本稀少时，可能会出现类似情况：
# 一些折叠数据集中没有疾病样本，因此在测试时既没有真阳性也没有假阴性。
# 这在数学上导致无限的 `LR+`，同样不应将其解释为模型完美识别阳性病例。这种情况
#
# 创建一个逻辑回归估计器对象
estimator = LogisticRegression()

# 使用 make_classification 函数生成一个包含300个样本的数据集，其中类别不平衡，正类和负类的比例为9:1
X, y = make_classification(n_samples=300, weights=[0.9, 0.1], random_state=0)

# 对逻辑回归模型进行交叉验证，评估模型性能，使用指定的评分函数和10折交叉验证
extract_score(cross_validate(estimator, X, y, scoring=scoring, cv=10))



# %%
# 预valence不变性
# =====================================
#
# 似然比独立于疾病的患病率，并且可以在不考虑可能存在的类别不平衡的情况下在不同人群之间外推，
# **只要同一个模型应用于所有人群**。请注意，在下面的图表中，**决策边界是恒定的**（请参阅
# :ref:`sphx_glr_auto_examples_svm_plot_separating_hyperplane_unbalanced.py` 了解不平衡类别决策边界的研究）。
#
# 在这里，我们在一个患病率为50%的病例对照研究中训练了一个 :class:`~sklearn.linear_model.LogisticRegression`
# 基础模型。然后对具有不同患病率的人群进行评估。我们使用 :func:`~sklearn.datasets.make_classification`
# 函数来确保数据生成过程在下面的图表中始终相同。标签 `1` 对应于阳性类别 "disease"，而标签 `0` 则表示 "no-disease"。
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from sklearn.inspection import DecisionBoundaryDisplay

# 创建一个默认字典，用于存储不同患病率下的数据
populations = defaultdict(list)

# 公共参数，用于生成分类数据的设置
common_params = {
    "n_samples": 10_000,
    "n_features": 2,
    "n_informative": 2,
    "n_redundant": 0,
    "random_state": 0,
}

# 创建一个权重数组，用于在不同患病率下生成数据
weights = np.linspace(0.1, 0.8, 6)
weights = weights[::-1]

# %%
# 在平衡类别上拟合和评估基础模型
# ------------------------------------------------
#
# 这里我们使用 make_classification 函数生成一个平衡类别的数据集，用于训练和评估基础模型。
# 逻辑回归模型被训练在这些平衡类别上，并且进行了交叉验证来评估其性能。
# 提取了正类和负类的似然比及其标准差。

# 生成平衡类别的数据集
X, y = make_classification(**common_params, weights=[0.5, 0.5])

# 创建并拟合逻辑回归模型
estimator = LogisticRegression().fit(X, y)

# 对逻辑回归模型进行交叉验证，评估模型性能，使用指定的评分函数和10折交叉验证
lr_base = extract_score(cross_validate(estimator, X, y, scoring=scoring, cv=10))
pos_lr_base, pos_lr_base_std = lr_base["positive"].values
neg_lr_base, neg_lr_base_std = lr_base["negative"].values

# %%
# 现在我们将展示每个患病率水平的决策边界。请注意，我们只绘制原始数据的子集，以更好地评估线性模型的决策边界。
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))

for ax, (n, weight) in zip(axs.ravel(), enumerate(weights)):
    # 在给定权重下生成分类数据
    X, y = make_classification(
        **common_params,
        weights=[weight, 1 - weight],
    )
    
    # 计算当前数据集的患病率
    prevalence = y.mean()
    
    # 将当前数据集的患病率添加到 populations 字典中
    populations["prevalence"].append(prevalence)
    populations["X"].append(X)
    populations["y"].append(y)

    # 为了绘图，从数据集中随机抽取500个样本进行绘制
    rng = np.random.RandomState(1)
    plot_indices = rng.choice(np.arange(X.shape[0]), size=500, replace=True)
    X_plot, y_plot = X[plot_indices], y[plot_indices]

    # 使用固定的基础模型决策边界绘制不同患病率下的决策边界
    # 使用给定的estimator创建一个DecisionBoundaryDisplay对象，用于显示决策边界
    disp = DecisionBoundaryDisplay.from_estimator(
        estimator,          # 给定的机器学习模型或者估算器
        X_plot,             # 绘制决策边界所用的特征数据
        response_method="predict",  # 使用"predict"方法来获取预测值
        alpha=0.5,          # 设置绘制的透明度为0.5
        ax=ax,              # 将图形绘制在给定的坐标轴(ax)上
    )
    # 在绘制的图形上添加散点图，显示特征数据的分布
    scatter = disp.ax_.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, edgecolor="k")
    # 设置图形的标题，显示类别标签的平均值
    disp.ax_.set_title(f"prevalence = {y_plot.mean():.2f}")
    # 在图形上添加图例，用散点图中的类别颜色表示类别
    disp.ax_.legend(*scatter.legend_elements())
# %%
# We define a function for bootstrapping.
def scoring_on_bootstrap(estimator, X, y, rng, n_bootstrap=100):
    # Initialize a defaultdict to store results for each prevalence
    results_for_prevalence = defaultdict(list)
    
    # Perform bootstrapping n_bootstrap times
    for _ in range(n_bootstrap):
        # Generate random bootstrap indices with replacement
        bootstrap_indices = rng.choice(
            np.arange(X.shape[0]), size=X.shape[0], replace=True
        )
        
        # Score the estimator using the bootstrap samples and store results
        for key, value in scoring(estimator, X[bootstrap_indices], y[bootstrap_indices]).items():
            results_for_prevalence[key].append(value)
    
    # Return results as a DataFrame
    return pd.DataFrame(results_for_prevalence)


# %%
# We score the base model for each prevalence using bootstrapping.

# Initialize a defaultdict to store aggregated metrics
results = defaultdict(list)
n_bootstrap = 100
rng = np.random.default_rng(seed=0)

# Iterate over each prevalence, X, and y in the populations dataset
for prevalence, X, y in zip(populations["prevalence"], populations["X"], populations["y"]):
    # Score the estimator on bootstrapped samples
    results_for_prevalence = scoring_on_bootstrap(estimator, X, y, rng, n_bootstrap=n_bootstrap)
    
    # Append prevalence value
    results["prevalence"].append(prevalence)
    
    # Aggregate mean and standard deviation of metrics across bootstraps
    results["metrics"].append(
        results_for_prevalence.aggregate(["mean", "std"]).unstack()
    )

# Convert results to a DataFrame
results = pd.DataFrame(results["metrics"], index=results["prevalence"])
results.index.name = "prevalence"
results

# %%
# In the plots below we observe that the class likelihood ratios re-computed
# with different prevalences are indeed constant within one standard deviation
# of those computed with on balanced classes.

# Create subplots for positive and negative likelihood ratios
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot mean positive likelihood ratio and confidence intervals
results["positive_likelihood_ratio"]["mean"].plot(
    ax=ax1, color="r", label="extrapolation through populations"
)
ax1.axhline(y=pos_lr_base + pos_lr_base_std, color="r", linestyle="--")
ax1.axhline(
    y=pos_lr_base - pos_lr_base_std,
    color="r",
    linestyle="--",
    label="base model confidence band",
)
ax1.fill_between(
    results.index,
    results["positive_likelihood_ratio"]["mean"]
    - results["positive_likelihood_ratio"]["std"],
    results["positive_likelihood_ratio"]["mean"]
    + results["positive_likelihood_ratio"]["std"],
    color="r",
    alpha=0.3,
)
ax1.set(
    title="Positive likelihood ratio",
    ylabel="LR+",
    ylim=[0, 5],
)
ax1.legend(loc="lower right")

# Plot mean negative likelihood ratio and confidence intervals
ax2 = results["negative_likelihood_ratio"]["mean"].plot(
    ax=ax2, color="b", label="extrapolation through populations"
)
ax2.axhline(y=neg_lr_base + neg_lr_base_std, color="b", linestyle="--")
ax2.axhline(
    y=neg_lr_base - neg_lr_base_std,
    color="b",
    linestyle="--",
    label="base model confidence band",
)
ax2.fill_between(
    results.index,
    results["negative_likelihood_ratio"]["mean"]
    - results["negative_likelihood_ratio"]["std"],
    results["negative_likelihood_ratio"]["mean"]
    + results["negative_likelihood_ratio"]["std"],
    color="b",
    alpha=0.3,
)
ax2.set(
    title="Negative likelihood ratio",
    ylabel="LR-",
    ylim=[0, 0.5],
)
ax2.legend(loc="lower right")

# Display the plot
plt.show()
```