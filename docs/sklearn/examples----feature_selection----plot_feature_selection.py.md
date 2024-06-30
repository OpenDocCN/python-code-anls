# `D:\src\scipysrc\scikit-learn\examples\feature_selection\plot_feature_selection.py`

```
# %%
# Generate sample data
# --------------------
#
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# The iris dataset
X, y = load_iris(return_X_y=True)

# Some noisy data not correlated
E = np.random.RandomState(42).uniform(0, 0.1, size=(X.shape[0], 20))

# Add the noisy data to the informative features
X = np.hstack((X, E))

# Split dataset to select feature and evaluate the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# %%
# Univariate feature selection
# ----------------------------
#
# Univariate feature selection with F-test for feature scoring.
# We use the default selection function to select
# the four most significant features.
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=4)  # 使用F检验进行单变量特征选择，选择4个最显著的特征
selector.fit(X_train, y_train)
scores = -np.log10(selector.pvalues_)  # 计算特征的p值的负对数作为特征重要性得分
scores /= scores.max()  # 归一化得分，使得最大得分为1

# %%
import matplotlib.pyplot as plt

X_indices = np.arange(X.shape[-1])
plt.figure(1)
plt.clf()
plt.bar(X_indices - 0.05, scores, width=0.2)
plt.title("Feature univariate score")  # 设置图表标题为“特征单变量得分”
plt.xlabel("Feature number")  # 设置x轴标签为“特征编号”
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")  # 设置y轴标签为“单变量得分（$-Log(p_{value})$）”
plt.show()

# %%
# In the total set of features, only the 4 of the original features are significant.
# We can see that they have the highest score with univariate feature
# selection.

# %%
# Compare with SVMs
# -----------------
#
# Without univariate feature selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

clf = make_pipeline(MinMaxScaler(), LinearSVC())  # 创建包括MinMaxScaler和LinearSVC的管道
clf.fit(X_train, y_train)
print(
    "Classification accuracy without selecting features: {:.3f}".format(
        clf.score(X_test, y_test)
    )
)

svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
svm_weights /= svm_weights.sum()

# %%
# After univariate feature selection
clf_selected = make_pipeline(SelectKBest(f_classif, k=4), MinMaxScaler(), LinearSVC())
clf_selected.fit(X_train, y_train)
print(
    "Classification accuracy after univariate feature selection: {:.3f}".format(
        clf_selected.score(X_test, y_test)
    )
)

svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
svm_weights_selected /= svm_weights_selected.sum()

# %%
plt.bar(
    X_indices - 0.45, scores, width=0.2, label=r"Univariate score ($-Log(p_{value})$)"



# 在图表上绘制一个条形图，具体设置包括：
# - X_indices: 横坐标位置，偏移量为0.45
# - scores: 条形图的高度或数值
# - width: 条形的宽度，设置为0.2
# - label: 条形图的标签，显示为"Univariate score ($-Log(p_{value})$)"
# 使用 Matplotlib 绘制条形图，显示 SVM 模型的特征权重
plt.bar(X_indices - 0.25, svm_weights, width=0.2, label="SVM weight")

# 继续在同一图上绘制经过特征选择后的 SVM 特征权重条形图
plt.bar(
    X_indices[selector.get_support()] - 0.05,
    svm_weights_selected,
    width=0.2,
    label="SVM weights after selection",
)

# 设置图表标题
plt.title("Comparing feature selection")
# 设置 x 轴标签
plt.xlabel("Feature number")
# 设置 y 轴刻度为空，不显示刻度值
plt.yticks(())
# 调整坐标轴范围，使得图表更紧凑
plt.axis("tight")
# 设置图例位置在右上角
plt.legend(loc="upper right")
# 显示图表
plt.show()

# %%
# 在未进行单变量特征选择的情况下，SVM 模型会给前四个原始显著特征分配较大的权重，
# 但也会选择许多无信息的特征。在应用单变量特征选择之后再训练 SVM 模型，
# 会增加对显著特征的权重，从而提升分类性能。
```