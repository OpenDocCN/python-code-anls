# `D:\src\scipysrc\scikit-learn\examples\feature_selection\plot_feature_selection_pipeline.py`

```
# %%
# 我们将从生成一个二元分类数据集开始。随后，我们将该数据集分为两个子集。

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_features=20,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
# 特征选择时常见的错误是在整个数据集上搜索区分性特征的子集，而不是仅在训练集上进行。使用 scikit-learn 的 :func:`~sklearn.pipeline.Pipeline`
# 可以防止这样的错误发生。
#
# 在这里，我们将演示如何构建一个流水线，其中第一步是特征选择。
#
# 在对训练数据调用 `fit` 方法时，将选择一部分特征，并将这些选定特征的索引存储下来。特征选择器随后会减少特征数量，并将这个子集传递给分类器进行训练。

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# 创建一个 ANOVA 特征选择器
anova_filter = SelectKBest(f_classif, k=3)
# 创建一个线性支持向量分类器
clf = LinearSVC()
# 创建包含特征选择器和分类器的流水线
anova_svm = make_pipeline(anova_filter, clf)
# 在训练集上拟合流水线
anova_svm.fit(X_train, y_train)

# %%
# 训练完成后，我们可以对新的未见过的样本进行预测。在这种情况下，特征选择器将根据训练过程中存储的信息仅选择最具区分性的特征。然后，数据将传递给分类器进行预测。

from sklearn.metrics import classification_report

# 使用训练好的模型对测试集进行预测
y_pred = anova_svm.predict(X_test)
# 打印分类报告，展示最终的评估指标
print(classification_report(y_test, y_pred))

# %%
# 注意，您可以检查流水线中的每个步骤。例如，我们可能对分类器的参数感兴趣。由于我们选择了三个特征，我们期望有三个系数。

anova_svm[-1].coef_

# %%
# 然而，我们不知道哪些特征是从原始数据集中选择出来的。我们可以通过几种方式来进行处理。在这里，我们将反转这些系数的转换，以获取有关原始空间的信息。

anova_svm[:-1].inverse_transform(anova_svm[-1].coef_)

# %%
# 我们可以看到，具有非零系数的特征是第一步选择的特征。
```