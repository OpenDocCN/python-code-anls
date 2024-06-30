# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_roc.py`

```
# %%
# Load and prepare data
# =====================
#
# We import the :ref:`iris_dataset` which contains 3 classes, each one
# corresponding to a type of iris plant. One class is linearly separable from
# the other 2; the latter are **not** linearly separable from each other.
#
# Here we binarize the output and add noisy features to make the problem harder.
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载iris数据集，其中包含3种类别，每种代表一种鸢尾植物类型。
iris = load_iris()
target_names = iris.target_names
X, y = iris.data, iris.target
y = iris.target_names[y]

# 创建随机种子并增加噪声特征，使问题更加复杂。
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
n_classes = len(np.unique(y))
X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)
# 将数据集分割为训练集和测试集
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

# %%
# We train a :class:`~sklearn.linear_model.LogisticRegression` model which can
# naturally handle multiclass problems, thanks to the use of the multinomial
# formulation.
from sklearn.linear_model import LogisticRegression

# 实例化LogisticRegression分类器
classifier = LogisticRegression()
# 训练分类器并预测测试集数据的概率
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# %%
# One-vs-Rest multiclass ROC
# ==========================
#
# The One-vs-the-Rest (OvR) multiclass strategy, also known as one-vs-all,
# consists in computing a ROC curve per each of the `n_classes`. In each step, a
# 在多类别分类中，One-vs-Rest (OvR) 策略也称为一对全，会计算每个类别的ROC曲线。
# 导入所需的库和模块
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np

# 使用 LabelBinarizer 对训练集的目标变量进行拟合，实现标签的二进制编码
label_binarizer = LabelBinarizer().fit(y_train)

# 对测试集的目标变量进行一对多编码（One-vs-Rest）
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape  # 输出结果的形状为 (n_samples, n_classes)

# %%
# 我们也可以轻松地检查特定类别的编码：

label_binarizer.transform(["virginica"])

# %%
# ROC 曲线展示特定类别
# --------------------
#
# 在下面的图中，我们展示了当将鸢尾花视为"virginica"类（`class_id=2`）或"非virginica"类时，
# 得到的ROC曲线。

class_of_interest = "virginica"
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
class_id

# %%
# 使用 RocCurveDisplay.from_predictions 创建 ROC 曲线显示对象，显示 "virginica vs the rest" 的 ROC 曲线。

display = RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    y_score[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    plot_chance_level=True,
)
_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)",
)

# %%
# 使用微平均的 OvR 绘制 ROC 曲线
# ------------------------------
#
# 微平均将所有类别的贡献汇总起来，使用 numpy.ravel 计算平均指标，具体计算方式如下：
#
# TPR = sum(TP_c) / sum(TP_c + FN_c)
#
# FPR = sum(FP_c) / sum(FP_c + TN_c)
#
# 我们可以简单演示 numpy.ravel 的效果：

print(f"y_score:\n{y_score[0:2,:]}")
print()
print(f"y_score.ravel():\n{y_score[0:2,:].ravel()}")

# %%
# 在多类分类设置中，当类别严重不平衡时，微平均优于宏平均。在这种情况下，可以考虑使用加权宏平均，这里未作演示。

# 使用 RocCurveDisplay.from_predictions 创建 ROC 曲线显示对象，显示 "micro-average OvR" 的 ROC 曲线。

display = RocCurveDisplay.from_predictions(
    y_onehot_test.ravel(),
    y_score.ravel(),
    name="micro-average OvR",
    color="darkorange",
    plot_chance_level=True,
)
_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
)
    title="Micro-averaged One-vs-Rest\nReceiver Operating Characteristic",


# 定义变量 title，并赋值为指定的字符串，包含换行符
# %%
# 如果主要关注的不是绘图而是ROC-AUC分数本身，可以使用:class:`~sklearn.metrics.roc_auc_score`来重现图表中显示的值。
from sklearn.metrics import roc_auc_score

# 计算多类别问题的Micro-averaged One-vs-Rest ROC AUC分数
micro_roc_auc_ovr = roc_auc_score(
    y_test,       # 真实标签
    y_score,      # 预测得分
    multi_class="ovr",  # 使用One-vs-Rest策略
    average="micro",    # 使用micro平均
)

# 打印Micro-averaged One-vs-Rest ROC AUC分数
print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")

# %%
# 这等同于使用:class:`~sklearn.metrics.roc_curve`计算ROC曲线，然后使用:class:`~sklearn.metrics.auc`计算展开的真实类别和预测类别的曲线下面积。
from sklearn.metrics import auc, roc_curve

# 存储所有平均策略的假正率(fpr)、真正率(tpr)和ROC AUC
fpr, tpr, roc_auc = dict(), dict(), dict()

# 计算Micro-average ROC曲线和ROC曲线下面积
fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 打印Micro-averaged One-vs-Rest ROC AUC分数
print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

# %%
# .. note:: 默认情况下，ROC曲线的计算会通过线性插值和McClish校正在最大假阳性率处添加一个单独的点。
#
# 使用OvR宏平均的ROC曲线
# -------------------------------------
#
# 获得宏平均需要独立计算每个类别的指标，然后对它们取平均，因此在先验上平等地对待所有类别。
# 首先聚合每个类别的真阳性率和假阳性率：

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr_grid = np.linspace(0.0, 1.0, 1000)

# 在这些点上对所有ROC曲线进行插值
mean_tpr = np.zeros_like(fpr_grid)

for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # 线性插值

# 平均化并计算AUC
mean_tpr /= n_classes

fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 打印Macro-averaged One-vs-Rest ROC AUC分数
print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

# %%
# 这个计算等同于直接调用
macro_roc_auc_ovr = roc_auc_score(
    y_test,       # 真实标签
    y_score,      # 预测得分
    multi_class="ovr",  # 使用One-vs-Rest策略
    average="macro",    # 使用macro平均
)

# 打印Macro-averaged One-vs-Rest ROC AUC分数
print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")

# %%
# 将所有OvR ROC曲线绘制在一起
# --------------------------------
from itertools import cycle

fig, ax = plt.subplots(figsize=(6, 6))

plt.plot(
    fpr["micro"],                   # micro-average的假阳性率
    tpr["micro"],                   # micro-average的真阳性率
    label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",  # 图例标签
    color="deeppink",               # 曲线颜色
    linestyle=":",                  # 线型
    linewidth=4,                    # 线宽
)

plt.plot(
    fpr["macro"],                   # macro-average的假阳性率
    tpr["macro"],                   # macro-average的真阳性率
    label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",  # 图例标签
    color="navy",                   # 曲线颜色
    linestyle=":",                  # 线型
    linewidth=4,                    # 线宽
)
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
# 使用循环生成一个无限序列，包含颜色名称，用于绘制 ROC 曲线的不同类别的颜色

for class_id, color in zip(range(n_classes), colors):
    # 遍历每个类别 ID 和对应的颜色
    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"ROC curve for {target_names[class_id]}",
        color=color,
        ax=ax,
        plot_chance_level=(class_id == 2),
    )
    # 绘制 ROC 曲线，使用预测值和类别分数
    # 设置曲线名称为对应类别的 ROC 曲线
    # 使用指定颜色绘制曲线
    # 在绘制的同时，考虑是否绘制随机水平线（当类别 ID 为 2 时）

_ = ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
)
# 设置坐标轴标签和标题，展示多类别的 ROC 曲线

# %%
# One-vs-One multiclass ROC
# =========================
#
# The One-vs-One (OvO) multiclass strategy consists in fitting one classifier
# per class pair. Since it requires to train `n_classes` * (`n_classes` - 1) / 2
# classifiers, this method is usually slower than One-vs-Rest due to its
# O(`n_classes` ^2) complexity.
#
# In this section, we demonstrate the macro-averaged AUC using the OvO scheme
# for the 3 possible combinations in the :ref:`iris_dataset`: "setosa" vs
# "versicolor", "versicolor" vs "virginica" and  "virginica" vs "setosa". Notice
# that micro-averaging is not defined for the OvO scheme.
#
# ROC curve using the OvO macro-average
# -------------------------------------
#
# In the OvO scheme, the first step is to identify all possible unique
# combinations of pairs. The computation of scores is done by treating one of
# the elements in a given pair as the positive class and the other element as
# the negative class, then re-computing the score by inversing the roles and
# taking the mean of both scores.
#
# 根据 OvO 方案展示宏平均 AUC
# -------------------------------------
#
# OvO 方案中的 ROC 曲线绘制过程，首先需要确定所有可能的唯一类别对组合。

from itertools import combinations
# 导入 itertools 中的 combinations 函数，用于生成类别对的组合

pair_list = list(combinations(np.unique(y), 2))
# 生成所有可能的类别对组合列表，并转换为列表形式
print(pair_list)

# %%
pair_scores = []
mean_tpr = dict()

for ix, (label_a, label_b) in enumerate(pair_list):
    # 遍历类别对的索引和对应的类别标签
    
    a_mask = y_test == label_a
    b_mask = y_test == label_b
    ab_mask = np.logical_or(a_mask, b_mask)
    # 创建布尔掩码，标识出属于类别 A 或类别 B 的样本

    a_true = a_mask[ab_mask]
    b_true = b_mask[ab_mask]
    # 从测试集中提取出符合类别 A 和类别 B 的真实标签

    idx_a = np.flatnonzero(label_binarizer.classes_ == label_a)[0]
    idx_b = np.flatnonzero(label_binarizer.classes_ == label_b)[0]
    # 确定类别 A 和类别 B 在二进制标签编码中的索引位置

    fpr_a, tpr_a, _ = roc_curve(a_true, y_score[ab_mask, idx_a])
    fpr_b, tpr_b, _ = roc_curve(b_true, y_score[ab_mask, idx_b])
    # 计算类别 A 和类别 B 的 ROC 曲线的假阳率和真阳率

    mean_tpr[ix] = np.zeros_like(fpr_grid)
    mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
    mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
    mean_tpr[ix] /= 2
    # 计算两个类别的平均真阳率

    mean_score = auc(fpr_grid, mean_tpr[ix])
    pair_scores.append(mean_score)
    # 计算平均 AUC 分数并添加到列表中

    fig, ax = plt.subplots(figsize=(6, 6))
    # 创建一个新的图形和轴对象，用于绘制 ROC 曲线

    plt.plot(
        fpr_grid,
        mean_tpr[ix],
        label=f"Mean {label_a} vs {label_b} (AUC = {mean_score :.2f})",
        linestyle=":",
        linewidth=4,
    )
    # 绘制平均 ROC 曲线，显示平均 AUC 分数
    # 设置线条风格为虚线，线宽为4

    RocCurveDisplay.from_predictions(
        a_true,
        y_score[ab_mask, idx_a],
        ax=ax,
        name=f"{label_a} as positive class",
    )
    # 使用 ROC 曲线显示类别 A 作为正类的曲线
    RocCurveDisplay.from_predictions(
        b_true,  # 实际标签数组，表示真实的类别
        y_score[ab_mask, idx_b],  # 预测分数数组，根据布尔掩码和索引选择特定类别的预测分数
        ax=ax,  # 绘图的坐标轴对象
        name=f"{label_b} as positive class",  # 曲线名称，以"{label_b} as positive class"格式命名
        plot_chance_level=True,  # 是否绘制偶然水平线
    )
    ax.set(
        xlabel="False Positive Rate",  # X 轴标签，表示假阳率
        ylabel="True Positive Rate",  # Y 轴标签，表示真阳率
        title=f"{target_names[idx_a]} vs {label_b} ROC curves",  # 图表标题，展示不同类别之间的 ROC 曲线
    )
# 打印多类别分类问题中使用的 One-vs-One ROC AUC 的宏平均分数，保留两位小数输出
print(f"Macro-averaged One-vs-One ROC AUC score:\n{np.average(pair_scores):.2f}")

# %%
# 也可以断言我们手动计算的宏平均与 `average="macro"` 选项实现的
# :class:`~sklearn.metrics.roc_auc_score` 函数等价。
# 使用 `roc_auc_score` 函数计算多类别分类的 One-vs-One ROC AUC 分数
macro_roc_auc_ovo = roc_auc_score(
    y_test,
    y_score,
    multi_class="ovo",
    average="macro",
)

# 打印计算得到的 One-vs-One ROC AUC 的宏平均分数，保留两位小数输出
print(f"Macro-averaged One-vs-One ROC AUC score:\n{macro_roc_auc_ovo:.2f}")

# %%
# 将所有的 OvO ROC 曲线绘制在一起
# --------------------------------
# 初始化 OvO 真正率数组
ovo_tpr = np.zeros_like(fpr_grid)

# 创建一个 6x6 大小的图表
fig, ax = plt.subplots(figsize=(6, 6))
# 遍历每一对类别标签，并绘制平均真正率
for ix, (label_a, label_b) in enumerate(pair_list):
    ovo_tpr += mean_tpr[ix]
    ax.plot(
        fpr_grid,
        mean_tpr[ix],
        label=f"Mean {label_a} vs {label_b} (AUC = {pair_scores[ix]:.2f})",
    )

# 计算 OvO 的平均真正率
ovo_tpr /= sum(1 for pair in enumerate(pair_list))

# 绘制 OvO 的平均 ROC 曲线
ax.plot(
    fpr_grid,
    ovo_tpr,
    label=f"One-vs-One macro-average (AUC = {macro_roc_auc_ovo:.2f})",
    linestyle=":",
    linewidth=4,
)
# 绘制对角线表示随机分类的效果
ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
# 设置图表的标签和标题等属性
_ = ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Extension of Receiver Operating Characteristic\nto One-vs-One multiclass",
    aspect="equal",
    xlim=(-0.01, 1.01),
    ylim=(-0.01, 1.01),
)

# %%
# 我们确认 "versicolor" 和 "virginica" 类别无法被线性分类器很好地识别。
# 注意 "virginica" 对其他类别的 One-vs-Rest ROC-AUC 分数为 0.77，
# 介于 "versicolor" vs "virginica" (0.64) 和 "setosa" vs "virginica" (0.90) 的 OvO ROC-AUC 分数之间。
# 实际上，OvO 策略在处理一对类别的混淆时提供了额外的信息，
# 但当类别数量较多时会增加计算成本。
#
# 如果用户主要关心正确识别特定类别或类别子集，推荐使用 OvO 策略。
# 而评估分类器的整体性能可以通过给定的平均策略进行总结。
#
# Micro-averaged OvR ROC 受到更频繁类别的支配，因为统计数据被合并。
# 宏平均方法更好地反映了较少类别的统计信息，因此在所有类别的性能同等重要时更为适用。
```