# `D:\src\scipysrc\scikit-learn\examples\multiclass\plot_multiclass_overview.py`

```
"""
===============================================
Overview of multiclass training meta-estimators
===============================================

In this example, we discuss the problem of classification when the target
variable is composed of more than two classes. This is called multiclass
classification.

In scikit-learn, all estimators support multiclass classification out of the
box: the most sensible strategy was implemented for the end-user. The
:mod:`sklearn.multiclass` module implements various strategies that one can use
for experimenting or developing third-party estimators that only support binary
classification.

:mod:`sklearn.multiclass` includes OvO/OvR strategies used to train a
multiclass classifier by fitting a set of binary classifiers (the
:class:`~sklearn.multiclass.OneVsOneClassifier` and
:class:`~sklearn.multiclass.OneVsRestClassifier` meta-estimators). This example
will review them.
"""

# %%
# The Yeast UCI dataset
# ---------------------
#
# In this example, we use a UCI dataset [1]_, generally referred as the Yeast
# dataset. We use the :func:`sklearn.datasets.fetch_openml` function to load
# the dataset from OpenML.
from sklearn.datasets import fetch_openml

# Fetch the Yeast dataset from OpenML, storing features in X and target labels in y
X, y = fetch_openml(data_id=181, as_frame=True, return_X_y=True)

# %%
# To know the type of data science problem we are dealing with, we can check
# the target for which we want to build a predictive model.

# Display the counts of each class in the target variable y, sorted by index
y.value_counts().sort_index()

# %%
# We see that the target is discrete and composed of 10 classes. We therefore
# deal with a multiclass classification problem.
#
# Strategies comparison
# ---------------------
#
# In the following experiment, we use a
# :class:`~sklearn.tree.DecisionTreeClassifier` and a
# :class:`~sklearn.model_selection.RepeatedStratifiedKFold` cross-validation
# with 3 splits and 5 repetitions.
#
# We compare the following strategies:
#
# * :class:~sklearn.tree.DecisionTreeClassifier can handle multiclass
#   classification without needing any special adjustments. It works by breaking
#   down the training data into smaller subsets and focusing on the most common
#   class in each subset. By repeating this process, the model can accurately
#   classify input data into multiple different classes.
# * :class:`~sklearn.multiclass.OneVsOneClassifier` trains a set of binary
#   classifiers where each classifier is trained to distinguish between
#   two classes.
# * :class:`~sklearn.multiclass.OneVsRestClassifier`: trains a set of binary
#   classifiers where each classifier is trained to distinguish between
#   one class and the rest of the classes.
# * :class:`~sklearn.multiclass.OutputCodeClassifier`: trains a set of binary
#   classifiers where each classifier is trained to distinguish between
#   a set of classes from the rest of the classes. The set of classes is
#   defined by a codebook, which is randomly generated in scikit-learn. This
#   method exposes a parameter `code_size` to control the size of the codebook.
# 导入 pandas 库，并使用别名 pd
import pandas as pd

# 导入交叉验证工具 RepeatedStratifiedKFold 和 cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
# 导入多类别分类器 OneVsOneClassifier、OneVsRestClassifier 和 OutputCodeClassifier
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
# 导入决策树分类器 DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# 定义重复分层 K 折交叉验证对象 cv
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=0)

# 初始化决策树分类器对象 tree
tree = DecisionTreeClassifier(random_state=0)
# 初始化 One-vs-One 多类别分类器对象 ovo_tree
ovo_tree = OneVsOneClassifier(tree)
# 初始化 One-vs-Rest 多类别分类器对象 ovr_tree
ovr_tree = OneVsRestClassifier(tree)
# 初始化输出码多类别分类器对象 ecoc
ecoc = OutputCodeClassifier(tree, code_size=2)

# 使用交叉验证计算决策树分类器的性能 cv_results_tree
cv_results_tree = cross_validate(tree, X, y, cv=cv, n_jobs=2)
# 使用交叉验证计算 One-vs-One 分类器的性能 cv_results_ovo
cv_results_ovo = cross_validate(ovo_tree, X, y, cv=cv, n_jobs=2)
# 使用交叉验证计算 One-vs-Rest 分类器的性能 cv_results_ovr
cv_results_ovr = cross_validate(ovr_tree, X, y, cv=cv, n_jobs=2)
# 使用交叉验证计算输出码分类器的性能 cv_results_ecoc
cv_results_ecoc = cross_validate(ecoc, X, y, cv=cv, n_jobs=2)

# %%
# 现在可以比较不同策略的统计性能。
# 绘制不同策略的得分分布图。
from matplotlib import pyplot as plt

# 将交叉验证结果存储到 DataFrame 中
scores = pd.DataFrame(
    {
        "DecisionTreeClassifier": cv_results_tree["test_score"],
        "OneVsOneClassifier": cv_results_ovo["test_score"],
        "OneVsRestClassifier": cv_results_ovr["test_score"],
        "OutputCodeClassifier": cv_results_ecoc["test_score"],
    }
)
# 使用核密度估计方法绘制得分分布图，显示图例
ax = scores.plot.kde(legend=True)
# 设置 x 轴标签为 "Accuracy score"
ax.set_xlabel("Accuracy score")
# 设置 x 轴范围
ax.set_xlim([0, 0.7])
# 设置图标题
_ = ax.set_title(
    "Density of the accuracy scores for the different multiclass strategies"
)

# %%
# 一开始可以看出决策树分类器的内置策略效果相当不错。
# One-vs-One 和纠错输出码策略效果更好。
# 然而，One-vs-Rest 策略的效果不如其他策略。
#
# 实际上，这些结果重现了文献中的某些报道 [2]_。
# 然而，事情并不像看起来那么简单。
#
# 超参数搜索的重要性
# ----------------------------------------
#
# 在 [3]_ 中后来还表明，如果首先优化基分类器的超参数，
# 多类别策略可能显示出类似的分数。
#
# 在这里，我们尝试通过至少优化基决策树的深度来复现这样的结果。
from sklearn.model_selection import GridSearchCV

# 定义网格搜索的参数字典，优化基决策树的最大深度
param_grid = {"max_depth": [3, 5, 8]}
# 使用 GridSearchCV 对象优化决策树分类器
tree_optimized = GridSearchCV(tree, param_grid=param_grid, cv=3)
# 使用优化后的决策树分类器重新初始化 One-vs-One 多类别分类器对象 ovo_tree
ovo_tree = OneVsOneClassifier(tree_optimized)
# 使用优化后的决策树分类器重新初始化 One-vs-Rest 多类别分类器对象 ovr_tree
ovr_tree = OneVsRestClassifier(tree_optimized)
# 使用优化后的决策树分类器初始化输出码多类别分类器对象 ecoc
ecoc = OutputCodeClassifier(tree_optimized, code_size=2)

# 使用交叉验证计算优化后的决策树分类器的性能 cv_results_tree
cv_results_tree = cross_validate(tree_optimized, X, y, cv=cv, n_jobs=2)
# 使用交叉验证计算优化后的 One-vs-One 分类器的性能 cv_results_ovo
cv_results_ovo = cross_validate(ovo_tree, X, y, cv=cv, n_jobs=2)
# 使用交叉验证计算优化后的 One-vs-Rest 分类器的性能 cv_results_ovr
cv_results_ovr = cross_validate(ovr_tree, X, y, cv=cv, n_jobs=2)
# 使用交叉验证计算优化后的输出码分类器的性能 cv_results_ecoc
cv_results_ecoc = cross_validate(ecoc, X, y, cv=cv, n_jobs=2)

# 将优化后的结果存储到 DataFrame 中
scores = pd.DataFrame(
    {
        # 使用 DecisionTreeClassifier 的测试分数作为值，关联到对应的分类器名
        "DecisionTreeClassifier": cv_results_tree["test_score"],
        # 使用 OneVsOneClassifier 的测试分数作为值，关联到对应的分类器名
        "OneVsOneClassifier": cv_results_ovo["test_score"],
        # 使用 OneVsRestClassifier 的测试分数作为值，关联到对应的分类器名
        "OneVsRestClassifier": cv_results_ovr["test_score"],
        # 使用 OutputCodeClassifier 的测试分数作为值，关联到对应的分类器名
        "OutputCodeClassifier": cv_results_ecoc["test_score"],
    }
)
# 创建一个密度图表格，并显示准确率分数的核密度估计
ax = scores.plot.kde(legend=True)
# 设置 x 轴标签为 "Accuracy score"
ax.set_xlabel("Accuracy score")
# 设置 x 轴的范围为 [0, 0.7]
ax.set_xlim([0, 0.7])
# 设置图表的标题为 "Density of the accuracy scores for the different multiclass strategies"
_ = ax.set_title(
    "Density of the accuracy scores for the different multiclass strategies"
)

plt.show()

# %%
# 我们可以看到一旦优化了超参数，所有的多类策略表现相似，如 [3]_ 所述。
#
# 结论
# ----
#
# 我们可以从这些结果中得到一些直觉。
#
# 首先，当超参数未经优化时，一对一和错误修正输出编码的表现优于决策树，原因在于它们集成了更多的分类器。集成提高了泛化性能。这与 bagging 分类器通常在未优化超参数的情况下表现更好的原因有些相似，如果不注意优化超参数的话。
#
# 然后，我们看到优化超参数的重要性。事实上，即使使用集成技术有助于减少这种影响，但在开发预测模型时，应该定期探索优化超参数。
#
# 最后，重要的是要记住，scikit-learn 中的估计器是采用特定策略来处理多类分类问题的。因此，对于这些估计器，通常不需要使用不同的策略。这些策略主要对于仅支持二元分类的第三方估计器有用。无论如何，我们也展示了优化超参数的重要性。
#
# 参考文献
# ------
#
#   .. [1] https://archive.ics.uci.edu/ml/datasets/Yeast
#
#   .. [2] `"Reducing multiclass to binary: A unifying approach for margin classifiers."
#      Allwein, Erin L., Robert E. Schapire, and Yoram Singer.
#      Journal of machine learning research 1
#      Dec (2000): 113-141.
#      <https://www.jmlr.org/papers/volume1/allwein00a/allwein00a.pdf>`_.
#
#   .. [3] `"In defense of one-vs-all classification."
#      Journal of Machine Learning Research 5
#      Jan (2004): 101-141.
#      <https://www.jmlr.org/papers/volume5/rifkin04a/rifkin04a.pdf>`_.
```