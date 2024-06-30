# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_tuned_decision_threshold.py`

```
# %%
# The diabetes dataset
# --------------------
#
# To illustrate the tuning of the decision threshold, we will use the diabetes dataset.
# This dataset is available on OpenML: https://www.openml.org/d/37. We use the
# :func:`~sklearn.datasets.fetch_openml` function to fetch this dataset.
from sklearn.datasets import fetch_openml

# 从 OpenML 中获取糖尿病数据集，返回一个包含数据和目标的 Pandas DataFrame
diabetes = fetch_openml(data_id=37, as_frame=True, parser="pandas")
data, target = diabetes.data, diabetes.target

# %%
# We look at the target to understand the type of problem we are dealing with.
# 输出目标变量的值统计信息，以理解我们正在处理的问题类型
target.value_counts()

# %%
# We can see that we are dealing with a binary classification problem. Since the
# labels are not encoded as 0 and 1, we make it explicit that we consider the class
# labeled "tested_negative" as the negative class (which is also the most frequent)
# and the class labeled "tested_positive" the positive class:
# 根据标签值的统计信息，确定负类和正类的标签
neg_label, pos_label = target.value_counts().index

# %%
# We can also observe that this binary problem is slightly imbalanced where we have
# around twice more samples from the negative class than from the positive class. When
# it comes to evaluation, we should consider this aspect to interpret the results.
# 观察到这个二元分类问题稍微不平衡，负类样本数量大约是正类样本数量的两倍，评估模型时需要考虑这一点
#
# Our vanilla classifier
# ----------------------
#
# We define a basic predictive model composed of a scaler followed by a logistic
# regression classifier.
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 创建一个包含标准化和逻辑回归分类器的流水线模型
model = make_pipeline(StandardScaler(), LogisticRegression())
model

# %%
# We evaluate our model using cross-validation. We use the accuracy and the balanced
# accuracy to report the performance of our model. The balanced accuracy is a metric
# that is less sensitive to class imbalance and will allow us to put the accuracy
# score in perspective.
# 使用交叉验证评估模型，使用准确率和平衡准确率来报告模型的性能，平衡准确率不太敏感于类别不平衡，有助于更好地理解准确率得分
#
# Cross-validation allows us to study the variance of the decision threshold across
# different splits of the data. However, the dataset is rather small and it would be
# detrimental to use more than 5 folds to evaluate the dispersion. Therefore, we use
# a :class:`~sklearn.model_selection.RepeatedStratifiedKFold` where we apply several
# repetitions of 5-fold cross-validation.
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
# 导入所需的模块和函数：RepeatedStratifiedKFold 用于重复分层 K 折交叉验证，
# cross_validate 用于执行交叉验证并返回评估指标的结果。

scoring = ["accuracy", "balanced_accuracy"]
# 定义用于评估模型的指标，包括准确率和平衡精度。

cv_scores = [
    "train_accuracy",
    "test_accuracy",
    "train_balanced_accuracy",
    "test_balanced_accuracy",
]
# 定义用于展示交叉验证结果的指标列表，包括训练集和测试集的准确率和平衡精度。

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
# 创建一个重复分层 K 折交叉验证对象，用于后续的模型评估。
# n_splits=5 表示每次分 5 折交叉验证，n_repeats=10 表示重复 10 次，
# random_state=42 用于确保结果的可重复性。

cv_results_vanilla_model = pd.DataFrame(
    cross_validate(
        model,
        data,
        target,
        scoring=scoring,
        cv=cv,
        return_train_score=True,
        return_estimator=True,
    )
)
# 执行交叉验证评估未调整阈值的基础模型，返回包含各项评估结果的数据帧。

cv_results_vanilla_model[cv_scores].aggregate(["mean", "std"]).T
# 计算基础模型的交叉验证评估指标的均值和标准差，并转置结果以便更好展示。

# %%
# 我们的预测模型成功地捕捉到了数据和目标之间的关系。训练和测试分数相近，
# 表明我们的预测模型没有过拟合。我们还可以观察到，平衡精度低于准确率，
# 这是由之前提到的类别不平衡导致的。
#
# 对于这个分类器，我们使用默认值 0.5 来作为决策阈值，将正类的概率转换为类别预测。
# 然而，这个阈值可能不是最优的。如果我们的目标是最大化平衡精度，我们应选择
# 另一个可以最大化该指标的阈值。
#
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` 元估计器允许
# 根据所选指标调整分类器的决策阈值。
#
# 调整决策阈值
# -----------------------------
#
# 我们创建一个 :class:`~sklearn.model_selection.TunedThresholdClassifierCV` 对象，
# 并配置它以最大化平衡精度。我们使用与之前相同的交叉验证策略来评估模型。

from sklearn.model_selection import TunedThresholdClassifierCV
# 导入调整阈值的元估计器类。

tuned_model = TunedThresholdClassifierCV(estimator=model, scoring="balanced_accuracy")
# 创建一个调整决策阈值的元估计器对象，基于平衡精度评分。

cv_results_tuned_model = pd.DataFrame(
    cross_validate(
        tuned_model,
        data,
        target,
        scoring=scoring,
        cv=cv,
        return_train_score=True,
        return_estimator=True,
    )
)
# 执行交叉验证评估调整过决策阈值的模型，返回包含各项评估结果的数据帧。

cv_results_tuned_model[cv_scores].aggregate(["mean", "std"]).T
# 计算调整过决策阈值模型的交叉验证评估指标的均值和标准差，并转置结果以便更好展示。

# %%
# 与基础模型相比，我们观察到平衡精度得分提高了。当然，这是以准确率得分降低为代价的。
# 这意味着我们的模型现在对正类更敏感，但在负类上犯更多错误。
#
# 然而，重要的是要注意，这个调整过的预测模型在内部与基础模型是相同的：它们具有相同的拟合系数。

import matplotlib.pyplot as plt
# 导入用于绘图的 matplotlib 库。

vanilla_model_coef = pd.DataFrame(
    [est[-1].coef_.ravel() for est in cv_results_vanilla_model["estimator"]],
    columns=diabetes.feature_names,
)
# 提取基础模型的各个估计器（estimator）的系数，并将其转换为数据帧。

tuned_model_coef = pd.DataFrame(
    [est.estimator_[-1].coef_.ravel() for est in cv_results_tuned_model["estimator"]],
    columns=diabetes.feature_names,
)
# 提取调整过决策阈值模型的各个估计器（estimator）的系数，并将其转换为数据帧。
# 创建一个包含两个子图的图形对象，每个子图共享相同的X和Y轴
fig, ax = plt.subplots(ncols=2, figsize=(12, 4), sharex=True, sharey=True)

# 在第一个子图(ax[0])上绘制箱线图，显示未调优模型的系数值分布
vanilla_model_coef.boxplot(ax=ax[0])

# 设置第一个子图(ax[0])的Y轴标签
ax[0].set_ylabel("Coefficient value")

# 设置第一个子图(ax[0])的标题
ax[0].set_title("Vanilla model")

# 在第二个子图(ax[1])上绘制箱线图，显示调优后模型的系数值分布
tuned_model_coef.boxplot(ax=ax[1])

# 设置第二个子图(ax[1])的标题
ax[1].set_title("Tuned model")

# 设置整个图形的标题
_ = fig.suptitle("Coefficients of the predictive models")

# %%
# 仅在交叉验证过程中改变了每个模型的决策阈值。
# 从调优后的模型的交叉验证结果中提取决策阈值，形成一个Series对象
decision_threshold = pd.Series(
    [est.best_threshold_ for est in cv_results_tuned_model["estimator"]],
)

# 绘制决策阈值的核密度估计图，显示其分布情况
ax = decision_threshold.plot.kde()

# 在核密度估计图上绘制平均决策阈值的垂直参考线
ax.axvline(
    decision_threshold.mean(),
    color="k",
    linestyle="--",
    label=f"Mean decision threshold: {decision_threshold.mean():.2f}",
)

# 设置X轴的标签
ax.set_xlabel("Decision threshold")

# 添加图例，位于右上角
ax.legend(loc="upper right")

# 设置图形的标题
_ = ax.set_title(
    "Distribution of the decision threshold \nacross different cross-validation folds"
)

# %%
# 平均而言，决策阈值约为0.32时能最大化平衡准确率，这与默认决策阈值0.5不同。
# 因此，在使用预测模型输出作为决策依据时，调整决策阈值尤为重要。
# 此外，用于调整决策阈值的度量标准应谨慎选择。
# 这里使用了平衡准确率，但对于特定问题，可能需要根据领域知识选择“正确”的度量标准。
# 更多细节可参考标题为：《如何选择合适的模型度量标准》的示例。
```