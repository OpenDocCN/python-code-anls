# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_roc_crossval.py`

```
# %%
# Classification and ROC analysis
# -------------------------------
#
# Here we run a :class:`~sklearn.svm.SVC` classifier with cross-validation and
# plot the ROC curves fold-wise. Notice that the baseline to define the chance
# level (dashed ROC curve) is a classifier that would always predict the most
# frequent class.

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold

# Number of splits for cross-validation
n_splits = 6
# Create StratifiedKFold object for cross-validation
cv = StratifiedKFold(n_splits=n_splits)
# Define the classifier: Support Vector Classifier with linear kernel,
# enabling probability estimates, and fixed random state for reproducibility
classifier = svm.SVC(kernel="linear", probability=True, random_state=random_state)

# Lists to store true positive rates (TPRs) and area under the ROC curve (AUC) values
tprs = []
aucs = []
# 创建一个包含100个均匀分布的假设的假阳性率数组
mean_fpr = np.linspace(0, 1, 100)

# 创建一个新的图形和轴对象，设置图形大小为6x6英寸
fig, ax = plt.subplots(figsize=(6, 6))

# 使用交叉验证(cv)的每一折(fold)来训练和评估分类器，并绘制ROC曲线
for fold, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])  # 在训练集上拟合分类器
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax,
        plot_chance_level=(fold == n_splits - 1),
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # 插值得到均匀的真阳性率数组
    interp_tpr[0] = 0.0  # 将插值后的第一个真阳性率设为0.0
    tprs.append(interp_tpr)  # 将插值后的真阳性率数组添加到列表中
    aucs.append(viz.roc_auc)  # 将当前折的AUC值添加到AUC列表中

# 计算所有折的平均真阳性率
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0  # 将平均真阳性率的最后一个值设为1.0
mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均ROC曲线下的AUC值
std_auc = np.std(aucs)  # 计算AUC值的标准差

# 绘制平均ROC曲线
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

# 计算真阳性率的标准差和上下界
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)  # 计算上界
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)  # 计算下界

# 使用灰色填充标准差区域
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

# 设置图形的标签和标题
ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label '{target_names[1]}')",
)

# 在右下角添加图例
ax.legend(loc="lower right")

# 显示图形
plt.show()
```