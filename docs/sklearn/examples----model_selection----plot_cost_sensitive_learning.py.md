# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_cost_sensitive_learning.py`

```
# %%
# Cost-sensitive learning with constant gains and costs
# -----------------------------------------------------
#
# In this first section, we illustrate the use of the
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` in a setting of
# cost-sensitive learning when the gains and costs associated to each entry of the
# confusion matrix are constant. We use the problematic presented in [2]_ using the
# "Statlog" German credit dataset [1]_.
#
# "Statlog" German credit dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We fetch the German credit dataset from OpenML.
import sklearn
from sklearn.datasets import fetch_openml

# 设置输出转换为 pandas 数据结构
sklearn.set_config(transform_output="pandas")

# 使用 fetch_openml 函数获取德国信用数据集，返回为 pandas 格式
german_credit = fetch_openml(data_id=31, as_frame=True, parser="pandas")
X, y = german_credit.data, german_credit.target

# %%
# We check the feature types available in `X`.
X.info()

# %%
# Many features are categorical and usually string-encoded. We need to encode
# these categories when we develop our predictive model. Let's check the targets.
y.value_counts()

# %%
# Another observation is that the dataset is imbalanced. We would need to be careful
# when evaluating our predictive model and use a family of metrics that are adapted
# to this setting.
#
# In addition, we observe that the target is string-encoded. Some metrics
# (e.g. precision and recall) require to provide the label of interest also called
# the "positive label". Here, we define that our goal is to predict whether or not
# a sample is a "bad" credit.
pos_label, neg_label = "bad", "good"

# %%
# To carry our analysis, we split our dataset using a single stratified split.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# %%
# We are ready to design our predictive model and the associated evaluation strategy.
#
# Evaluation metrics
# ^^^^^^^^^^^^^^^^^^
#
# In this section, we define a set of metrics that we use later. To see
# the effect of tuning the cut-off point, we evaluate the predictive model using
# the Receiver Operating Characteristic (ROC) curve and the Precision-Recall curve.
# The values reported on these plots are therefore the true positive rate (TPR),
# also known as the recall or the sensitivity, and the false positive rate (FPR),
# also known as the specificity, for the ROC curve and the precision and recall for
# the Precision-Recall curve.
#
# From these four metrics, scikit-learn does not provide a scorer for the FPR. We
# therefore need to define a small custom function to compute it.
from sklearn.metrics import confusion_matrix


def fpr_score(y, y_pred, neg_label, pos_label):
    cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    tn, fp, _, _ = cm.ravel()
    tnr = tn / (tn + fp)
    return 1 - tnr


# %%
# As previously stated, the "positive label" is not defined as the value "1" and calling
# some of the metrics with this non-standard value raise an error. We need to
# provide the indication of the "positive label" to the metrics.
#
# We therefore need to define a scikit-learn scorer using
# :func:`~sklearn.metrics.make_scorer` where the information is passed. We store all
# the custom scorers in a dictionary. To use them, we need to pass the fitted model,
# the data and the target on which we want to evaluate the predictive model.
from sklearn.metrics import make_scorer, precision_score, recall_score

tpr_score = recall_score  # TPR and recall are the same metric
scoring = {
    "precision": make_scorer(precision_score, pos_label=pos_label),
    "recall": make_scorer(recall_score, pos_label=pos_label),
    "fpr": make_scorer(fpr_score, neg_label=neg_label, pos_label=pos_label),
    "tpr": make_scorer(tpr_score, pos_label=pos_label),
}

# %%
# In addition, the original research [1]_ defines a custom business metric. We
# call a "business metric" any metric function that aims at quantifying how the
# predictions (correct or wrong) might impact the business value of deploying a
# given machine learning model in a specific application context. For our
# credit prediction task, the authors provide a custom cost-matrix which
# encodes that classifying a a "bad" credit as "good" is 5 times more costly on
# average than the opposite: it is less costly for the financing institution to
# not grant a credit to a potential customer that will not default (and
# therefore miss a good customer that would have otherwise both reimbursed the
# credit and payed interests) than to grant a credit to a customer that will
# default.
#
# We define a python function that weight the confusion matrix and return the
# overall cost.
import numpy as np


def credit_gain_score(y, y_pred, neg_label, pos_label):
    # Compute the confusion matrix between true labels (y) and predicted labels (y_pred)
    cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    
    # The rows of the confusion matrix hold the counts of observed classes
    # while the columns hold counts of predicted classes. Recall that here we
    # consider "bad" as the positive class (second row and column).
    
    # Scikit-learn model selection tools expect that we follow a convention
    # that "higher" means "better", hence the following gain matrix assigns
    # negative gains (costs) to the two kinds of prediction errors:
    # - a gain of -1 for each false positive ("good" credit labeled as "bad"),
    # - a gain of -5 for each false negative ("bad" credit labeled as "good"),
    # The true positives and true negatives are assigned null gains in this
    # metric.
    gain_matrix = np.array(
        [
            [0, -1],  # -1 gain for false positives
            [-5, 0],  # -5 gain for false negatives
        ]
    )
    
    # Calculate and return the overall credit gain score by multiplying
    # the confusion matrix with the gain matrix element-wise and summing up
    return np.sum(cm * gain_matrix)


scoring["credit_gain"] = make_scorer(
    credit_gain_score, neg_label=neg_label, pos_label=pos_label
)
# %%
# Vanilla predictive model
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# We use :class:`~sklearn.ensemble.HistGradientBoostingClassifier` as a predictive model
# that natively handles categorical features and missing values.
from sklearn.ensemble import HistGradientBoostingClassifier

# Initialize a HistGradientBoostingClassifier model with categorical features
# inferred from data type and set random state for reproducibility
model = HistGradientBoostingClassifier(
    categorical_features="from_dtype", random_state=0
).fit(X_train, y_train)
model

# %%
# We evaluate the performance of our predictive model using the ROC and Precision-Recall
# curves.
import matplotlib.pyplot as plt

from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

# Create subplots for ROC and Precision-Recall curves
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Plot Precision-Recall curve for the model
PrecisionRecallDisplay.from_estimator(
    model, X_test, y_test, pos_label=pos_label, ax=axs[0], name="GBDT"
)

# Overlay a default cut-off point at a probability of 0.5 on the Precision-Recall curve
axs[0].plot(
    scoring["recall"](model, X_test, y_test),
    scoring["precision"](model, X_test, y_test),
    marker="o",
    markersize=10,
    color="tab:blue",
    label="Default cut-off point at a probability of 0.5",
)
axs[0].set_title("Precision-Recall curve")
axs[0].legend()

设置子图 `axs[0]` 的标题为 "Precision-Recall curve"。


RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test,
    pos_label=pos_label,
    ax=axs[1],
    name="GBDT",
    plot_chance_level=True,
)

使用 `RocCurveDisplay` 类从模型 `model` 中生成 ROC 曲线，并将其绘制在子图 `axs[1]` 上。`pos_label` 参数指定正例标签，`name` 参数设置曲线的名称为 "GBDT"，`plot_chance_level=True` 表示绘制 ROC 曲线上的随机分类水平线。


axs[1].plot(
    scoring["fpr"](model, X_test, y_test),
    scoring["tpr"](model, X_test, y_test),
    marker="o",
    markersize=10,
    color="tab:blue",
    label="Default cut-off point at a probability of 0.5",
)

在子图 `axs[1]` 上绘制 ROC 曲线，使用 `scoring["fpr"]` 和 `scoring["tpr"]` 函数计算模型在测试集 `X_test` 和 `y_test` 上的假正率和真正率。使用圆形标记 ("o")，大小为 10，颜色为 "tab:blue"，标签为 "Default cut-off point at a probability of 0.5"。


axs[1].set_title("ROC curve")
axs[1].legend()

设置子图 `axs[1]` 的标题为 "ROC curve"，并添加图例。


_ = fig.suptitle("Evaluation of the vanilla GBDT model")

设置整个图形 `fig` 的标题为 "Evaluation of the vanilla GBDT model"。


# %%
# We recall that these curves give insights on the statistical performance of the
# predictive model for different cut-off points. For the Precision-Recall curve, the
# reported metrics are the precision and recall and for the ROC curve, the reported
# metrics are the TPR (same as recall) and FPR.
#
# Here, the different cut-off points correspond to different levels of posterior
# probability estimates ranging between 0 and 1. By default, `model.predict` uses a
# cut-off point at a probability estimate of 0.5. The metrics for such a cut-off point
# are reported with the blue dot on the curves: it corresponds to the statistical
# performance of the model when using `model.predict`.
#
# However, we recall that the original aim was to minimize the cost (or maximize the
# gain) as defined by the business metric. We can compute the value of the business
# metric:
print(f"Business defined metric: {scoring['credit_gain'](model, X_test, y_test)}")

打印显示关于统计性能的信息，指出 Precision-Recall 曲线报告的精确率和召回率，以及 ROC 曲线报告的真正率（TPR，与召回率相同）和假正率（FPR）。提到不同的切分点对应不同的后验概率估计水平，`model.predict` 默认使用概率估计为 0.5 的切分点。打印显示使用该切分点的模型统计性能的蓝色点。


# %%
# At this stage we don't know if any other cut-off can lead to a greater gain. To find
# the optimal one, we need to compute the cost-gain using the business metric for all
# possible cut-off points and choose the best. This strategy can be quite tedious to
# implement by hand, but the
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` class is here to help us.
# It automatically computes the cost-gain for all possible cut-off points and optimizes
# for the `scoring`.
#
# .. _cost_sensitive_learning_example:
#
# Tuning the cut-off point
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# We use :class:`~sklearn.model_selection.TunedThresholdClassifierCV` to tune the
# cut-off point. We need to provide the business metric to optimize as well as the
# positive label. Internally, the optimum cut-off point is chosen such that it maximizes
# the business metric via cross-validation. By default a 5-fold stratified
# cross-validation is used.
from sklearn.model_selection import TunedThresholdClassifierCV

tuned_model = TunedThresholdClassifierCV(
    estimator=model,
    scoring=scoring["credit_gain"],
    store_cv_results=True,  # necessary to inspect all results
)
tuned_model.fit(X_train, y_train)
print(f"{tuned_model.best_threshold_=:0.2f}")

引入 `TunedThresholdClassifierCV` 类来调整切分点。此处计算所有可能切分点的成本收益，并选择最佳切分点，最大化 `scoring` 中定义的业务指标。打印输出最佳切分点 `tuned_model.best_threshold_`。


# %%
# We plot the ROC and Precision-Recall curves for the vanilla model and the tuned model.
# Also we plot the cut-off points that would be used by each model. Because, we are

计划绘制 ROC 和 Precision-Recall 曲线，展示原始模型和调整后模型的性能，并标出每个模型使用的切分点。
# 重新使用相同的代码，定义一个生成绘图的函数。
def plot_roc_pr_curves(vanilla_model, tuned_model, *, title):
    # 创建一个包含3个子图的画布，设置尺寸为21x6英寸
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))

    # 设置线型、标记样式、颜色和名称
    linestyles = ("dashed", "dotted")
    markerstyles = ("o", ">")
    colors = ("tab:blue", "tab:orange")
    names = ("Vanilla GBDT", "Tuned GBDT")
    for idx, (est, linestyle, marker, color, name) in enumerate(
        zip((vanilla_model, tuned_model), linestyles, markerstyles, colors, names)
    ):
        # 获取分类器的最佳阈值（如果有的话）
        decision_threshold = getattr(est, "best_threshold_", 0.5)
        
        # 绘制Precision-Recall曲线，并添加到第一个子图
        PrecisionRecallDisplay.from_estimator(
            est,
            X_test,
            y_test,
            pos_label=pos_label,
            linestyle=linestyle,
            color=color,
            ax=axs[0],
            name=name,
        )
        axs[0].plot(
            scoring["recall"](est, X_test, y_test),
            scoring["precision"](est, X_test, y_test),
            marker,
            markersize=10,
            color=color,
            label=f"Cut-off point at probability of {decision_threshold:.2f}",
        )
        
        # 绘制ROC曲线，并添加到第二个子图
        RocCurveDisplay.from_estimator(
            est,
            X_test,
            y_test,
            pos_label=pos_label,
            linestyle=linestyle,
            color=color,
            ax=axs[1],
            name=name,
            plot_chance_level=idx == 1,
        )
        axs[1].plot(
            scoring["fpr"](est, X_test, y_test),
            scoring["tpr"](est, X_test, y_test),
            marker,
            markersize=10,
            color=color,
            label=f"Cut-off point at probability of {decision_threshold:.2f}",
        )

    # 设置第一个子图标题和图例
    axs[0].set_title("Precision-Recall curve")
    axs[0].legend()
    
    # 设置第二个子图标题和图例
    axs[1].set_title("ROC curve")
    axs[1].legend()

    # 绘制第三个子图，显示决策阈值对应的业务指标分数
    axs[2].plot(
        tuned_model.cv_results_["thresholds"],
        tuned_model.cv_results_["scores"],
        color="tab:orange",
    )
    axs[2].plot(
        tuned_model.best_threshold_,
        tuned_model.best_score_,
        "o",
        markersize=10,
        color="tab:orange",
        label="Optimal cut-off point for the business metric",
    )
    axs[2].legend()
    axs[2].set_xlabel("Decision threshold (probability)")
    axs[2].set_ylabel("Objective score (using cost-matrix)")
    axs[2].set_title("Objective score as a function of the decision threshold")
    
    # 设置整个图的标题
    fig.suptitle(title)
# 输出一个描述性信息，展示经过调整的模型在测试集上的业务定义指标得分
print(f"Business defined metric: {scoring['credit_gain'](tuned_model, X_test, y_test)}")

# %%
# 我们观察到，调整决策阈值几乎使我们的业务收益提高了近两倍。
#
# .. _TunedThresholdClassifierCV_no_cv:
#
# 关于模型重新拟合和交叉验证的考虑
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 在上述实验中，我们使用了 :class:`~sklearn.model_selection.TunedThresholdClassifierCV` 的默认设置。
# 具体来说，通过5折分层交叉验证来调整截断点。此外，一旦选择了截断点，基础预测模型将在整个训练数据集上重新拟合。
#
# 可以通过提供 `refit` 和 `cv` 参数来更改这两种策略。
# 例如，可以提供一个已拟合的 `estimator` 并设置 `cv="prefit"`，这样截断点将在拟合时提供的整个数据集上找到。
# 也可以通过设置 `refit=False` 来不重新拟合基础分类器。在这里，我们可以尝试进行这样的实验。
model.fit(X_train, y_train)
tuned_model.set_params(cv="prefit", refit=False).fit(X_train, y_train)
print(f"{tuned_model.best_threshold_=:0.2f}")

# %%
# 然后，我们使用与之前相同的方法评估我们的模型：
title = "Tuned GBDT model without refitting and using the entire dataset"
plot_roc_pr_curves(model, tuned_model, title=title)

# %%
# 我们观察到最优截断点与前一实验中找到的不同。如果我们看右侧的图表，我们会发现业务收益在一大段决策阈值范围内有一个接近于零的大台阶。
# 这种行为是过拟合的症状。因为我们禁用了交叉验证，我们在与模型训练集相同的集合上调整了截断点，这就是观察到过拟合的原因。
#
# 因此，使用此选项时需要谨慎。需要确保提供给 :class:`~sklearn.model_selection.TunedThresholdClassifierCV` 拟合时的数据不同于用于训练基础分类器的数据。
# 有时候，想要在完全新的验证集上调整预测模型，而不需要昂贵的完全重新拟合。
#
# 当交叉验证成本过高时，一种潜在的替代方案是通过将浮点数（范围为 `[0, 1]`）提供给 `cv` 参数来使用单一的训练-测试分割。这将数据分割为训练集和测试集。让我们探索这个选项：
tuned_model.set_params(cv=0.75).fit(X_train, y_train)

# %%
# 标题为“经过调优的GBDT模型，无需重新拟合并使用整个数据集”
plot_roc_pr_curves(model, tuned_model, title=title)

# %%
# 关于截断点，我们观察到最优点与多次重复交叉验证的情况相似。然而，请注意，单一拆分不能考虑拟合/预测过程的变异性，因此我们无法确定截断点是否存在变化。多次重复交叉验证可以平均这种效应。
#
# 另一个观察是关于调优模型的ROC和Precision-Recall曲线。如预期的那样，这些曲线与原始模型的曲线不同，因为我们在拟合期间使用了数据子集训练了底层分类器，并保留了验证集用于调整截断点。
#
# 在收益和成本不恒定时的成本敏感学习
# -------------------------------------------------------------
#
# 正如[2]_中所述，实际问题中收益和成本通常不是恒定的。在本节中，我们使用与[2]_中类似的示例来解决信用卡交易记录中欺诈检测的问题。
#
# 信用卡数据集
# ^^^^^^^^^^^^^^^^^^^^^^^
credit_card = fetch_openml(data_id=1597, as_frame=True, parser="pandas")
credit_card.frame.info()

# %%
# 数据集包含有关信用卡记录的信息，其中一些是欺诈性的，其他是合法的。因此，我们的目标是预测信用卡记录是否欺诈。
columns_to_drop = ["Class"]
data = credit_card.frame.drop(columns=columns_to_drop)
target = credit_card.frame["Class"].astype(int)

# %%
# 首先，我们检查数据集的类别分布。
target.value_counts(normalize=True)

# %%
# 数据集极度不平衡，欺诈交易仅占数据的0.17%。由于我们有兴趣训练一个机器学习模型，我们还应确保在少数类别中有足够的样本来训练模型。
target.value_counts()

# %%
# 我们观察到我们大约有500个样本，这是训练机器学习模型所需样本数的下限。除了目标分布外，我们还检查欺诈交易金额的分布。
fraud = target == 1
amount_fraud = data["Amount"][fraud]
_, ax = plt.subplots()
ax.hist(amount_fraud, bins=30)
ax.set_title("Amount of fraud transaction")
_ = ax.set_xlabel("Amount (€)")

# %%
# 使用业务指标解决问题
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 现在，我们创建依赖于每笔交易金额的业务指标。
# 定义成本矩阵，根据 [2]_ 类似的描述。接受合法交易获得交易金额的2%作为收益。
# 然而，接受欺诈交易将导致损失等于交易金额。如 [2]_ 所述，拒绝合法和欺诈交易的收益和损失
# 并不容易定义。在这里，我们定义拒绝合法交易估计为损失5€，拒绝欺诈交易估计为收益50€。
# 因此，我们定义以下函数来计算给定决策的总收益。
def business_metric(y_true, y_pred, amount):
    # 创建布尔掩码以确定真实为正例且预测为正例的情况
    mask_true_positive = (y_true == 1) & (y_pred == 1)
    # 创建布尔掩码以确定真实为反例且预测为反例的情况
    mask_true_negative = (y_true == 0) & (y_pred == 0)
    # 创建布尔掩码以确定真实为反例且预测为正例的情况
    mask_false_positive = (y_true == 0) & (y_pred == 1)
    # 创建布尔掩码以确定真实为正例且预测为反例的情况
    mask_false_negative = (y_true == 1) & (y_pred == 0)
    
    # 计算拒绝欺诈交易的总收益
    fraudulent_refuse = mask_true_positive.sum() * 50
    # 计算接受欺诈交易的总损失
    fraudulent_accept = -amount[mask_false_negative].sum()
    # 计算拒绝合法交易的总损失
    legitimate_refuse = mask_false_positive.sum() * -5
    # 计算接受合法交易的总收益
    legitimate_accept = (amount[mask_true_negative] * 0.02).sum()
    
    # 返回总收益，包括所有情况下的收益和损失
    return fraudulent_refuse + fraudulent_accept + legitimate_refuse + legitimate_accept



# %%
# 根据这个业务度量标准，我们创建一个 scikit-learn 的评分器，它能够在给定一个
# 拟合的分类器和一个测试集的情况下计算业务度量标准。为此，我们使用
# :func:`~sklearn.metrics.make_scorer` 工厂函数。变量 `amount` 是一个
# 额外的元数据，需要传递给评分器，并且我们需要使用 :ref:`元数据路由 <metadata_routing>` 来
# 考虑这些信息。
sklearn.set_config(enable_metadata_routing=True)
business_scorer = make_scorer(business_metric).set_score_request(amount=True)



# %%
# 因此，在这个阶段，我们观察到交易金额被使用了两次：一次作为训练预测模型的特征，
# 另一次作为元数据用于计算业务度量标准，进而计算我们模型的统计性能。作为特征时，
# 我们只需要在 `data` 中有一个包含每笔交易金额的列。要将此信息作为元数据使用，
# 我们需要有一个外部变量，可以将其传递给评分器或模型，并在内部路由到评分器。
# 所以让我们创建这个变量。
amount = credit_card.frame["Amount"].to_numpy()



# %%
from sklearn.model_selection import train_test_split

# 对数据进行训练集和测试集的划分，同时保持交易目标和金额的对应关系
data_train, data_test, target_train, target_test, amount_train, amount_test = (
    train_test_split(
        data, target, amount, stratify=target, test_size=0.5, random_state=42
    )
)



# %%
# 我们首先评估一些基线策略，作为参考。请记住，“0” 类是合法类，而“1” 类是欺诈类。
from sklearn.dummy import DummyClassifier

# 始终接受策略作为一个基线模型，始终预测为合法类
always_accept_policy = DummyClassifier(strategy="constant", constant=0)
always_accept_policy.fit(data_train, target_train)
# 使用业务评分器计算该策略的收益
benefit = business_scorer(
    always_accept_policy, data_test, target_test, amount=amount_test


    # 将 amount_test 元组的值分别赋给变量 always_accept_policy, data_test, target_test 和 amount
    # 如果 amount_test 元组中包含的元素个数不是四个，将会引发 ValueError 异常
# %%
# 输出具有“始终接受”政策的利益：约为220,000€。
# 我们对预测所有交易为欺诈性的分类器做出相同的评估。
always_reject_policy = DummyClassifier(strategy="constant", constant=1)
always_reject_policy.fit(data_train, target_train)
benefit = business_scorer(
    always_reject_policy, data_test, target_test, amount=amount_test
)
print(f"Benefit of the 'always reject' policy: {benefit:,.2f}€")

# %%
# 这样的政策将导致灾难性损失：约为670,000€。这是预期的，因为绝大多数交易是合法的，
# 但政策会以非常高的成本拒绝它们。
#
# 一个根据每个交易基础上的接受/拒绝决策进行调整的预测模型应该能够使我们获得大于
# 最佳常数基准政策220,000€的利润。
#
# 我们从默认决策阈值为0.5的逻辑回归模型开始。在这里，我们使用逻辑回归的`C`超参数
# 进行调优，使用适当的评分规则（对数损失）来确保模型的概率预测通过其`predict_proba`
# 方法尽可能准确，不受决策阈值值的影响。
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logistic_regression = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {"logisticregression__C": np.logspace(-6, 6, 13)}
model = GridSearchCV(logistic_regression, param_grid, scoring="neg_log_loss").fit(
    data_train, target_train
)
model

# %%
# 输出逻辑回归模型使用默认阈值的利益：
# {business_scorer(model, data_test, target_test, amount=amount_test):,.2f}€
print(
    "Benefit of logistic regression with default threshold: "
    f"{business_scorer(model, data_test, target_test, amount=amount_test):,.2f}€"
)

# %%
# 业务指标显示，我们的预测模型使用默认决策阈值在利润方面已经超过了基准值，
# 使用它来接受或拒绝交易已经是有利的选择。
#
# 调整决策阈值
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 现在的问题是：我们的模型是否对我们想做的决策类型是最优的？
# 到目前为止，我们尚未优化决策阈值。我们使用
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` 来优化我们的决策，
# 根据我们的业务评分来优化。为了避免嵌套交叉验证，我们将使用在前面网格搜索中找到的
# 最佳估算器。
tuned_model = TunedThresholdClassifierCV(
    estimator=model.best_estimator_,
    scoring=business_scorer,
    thresholds=100,
    n_jobs=2,
)

# %%
# 由于我们的业务评分器需要每笔交易的金额信息，我们需要在`fit`方法中传递这些信息。
# The
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` is in charge of
# automatically dispatching this metadata to the underlying scorer.
tuned_model.fit(data_train, target_train, amount=amount_train)

# %%
# We observe that the tuned decision threshold is far away from the default 0.5:
print(f"Tuned decision threshold: {tuned_model.best_threshold_:.2f}")

# %%
print(
    "Benefit of logistic regression with a tuned threshold: "
    f"{business_scorer(tuned_model, data_test, target_test, amount=amount_test):,.2f}€"
)

# %%
# We observe that tuning the decision threshold increases the expected profit
# when deploying our model - as indicated by the business metric. It is therefore
# valuable, whenever possible, to optimize the decision threshold with respect
# to the business metric.
#
# Manually setting the decision threshold instead of tuning it
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the previous example, we used the
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` to find the optimal
# decision threshold. However, in some cases, we might have some prior knowledge about
# the problem at hand and we might be happy to set the decision threshold manually.
#
# The class :class:`~sklearn.model_selection.FixedThresholdClassifier` allows us to
# manually set the decision threshold. At prediction time, it behave as the previous
# tuned model but no search is performed during the fitting process.
#
# Here, we will reuse the decision threshold found in the previous section to create a
# new model and check that it gives the same results.
from sklearn.model_selection import FixedThresholdClassifier

model_fixed_threshold = FixedThresholdClassifier(
    estimator=model, threshold=tuned_model.best_threshold_, prefit=True
).fit(data_train, target_train)

# %%
business_score = business_scorer(
    model_fixed_threshold, data_test, target_test, amount=amount_test
)
print(f"Benefit of logistic regression with a tuned threshold:  {business_score:,.2f}€")

# %%
# We observe that we obtained the exact same results but the fitting process
# was much faster since we did not perform any hyper-parameter search.
#
# Finally, the estimate of the (average) business metric itself can be unreliable, in
# particular when the number of data points in the minority class is very small.
# Any business impact estimated by cross-validation of a business metric on
# historical data (offline evaluation) should ideally be confirmed by A/B testing
# on live data (online evaluation). Note however that A/B testing models is
# beyond the scope of the scikit-learn library itself.
```