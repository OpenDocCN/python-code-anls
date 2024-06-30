# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_forest_importances.py`

```
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# %%
# 数据生成和模型拟合
# ---------------------------------
# 生成一个带有3个信息特征的合成数据集。我们明确不打乱数据集，以确保信息特征
# 对应于 X 的前三列。此外，我们将数据集分割为训练集和测试集。
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1000,        # 样本数
    n_features=10,         # 总特征数
    n_informative=3,       # 信息特征数
    n_redundant=0,         # 冗余特征数
    n_repeated=0,          # 重复特征数
    n_classes=2,           # 分类类别数
    random_state=0,        # 随机种子
    shuffle=False,         # 不打乱数据
)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# %%
# 将随机森林分类器拟合，以计算特征重要性。
from sklearn.ensemble import RandomForestClassifier

feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

# %%
# 基于信息增益平均减少的特征重要性
# -----------------------------------------------------
# 特征重要性由已拟合属性 `feature_importances_` 提供，并且它们是在每棵树中
# 对不纯度减少累积的平均值和标准差计算得出的。
#
# .. warning::
#     基于不纯度的特征重要性对于 **高基数** 特征（具有许多唯一值）可能具有误导性。
#     可参考下文的 :ref:`permutation_importance` 作为替代方法。
import time

import numpy as np

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

# %%
# 绘制基于不纯度的特征重要性图。
import pandas as pd

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# %%
# 我们观察到，如预期的那样，前三个特征被认为是重要的。
#
# 基于特征排列的特征重要性
# -----------------------------------------------
# 特征排列的特征重要性克服了基于不纯度的方法的一些限制。
# 导入 permutation_importance 函数，用于计算特征重要性，不会偏向高基数特征，并可在测试集上计算。
from sklearn.inspection import permutation_importance

# 记录开始时间
start_time = time.time()
# 计算特征重要性，使用 forest 模型在 X_test 和 y_test 上，重复10次，随机种子为42，使用2个并行工作线程
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
# 计算经过的时间
elapsed_time = time.time() - start_time
# 打印计算重要性所用的时间
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

# 创建一个 Series 对象，包含特征重要性的均值，使用特征名作为索引
forest_importances = pd.Series(result.importances_mean, index=feature_names)

# %%
# 计算完整置换重要性的计算成本更高。特征被重复洗牌 n 次，并重新拟合模型以估计其重要性。
# 详细信息请参见 :ref:`permutation_importance`。现在我们可以绘制特征重要性排序。

# 创建一个图形和轴对象
fig, ax = plt.subplots()
# 使用条形图绘制特征重要性，yerr 表示标准差，ax=ax 表示将图绘制在指定的轴上
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# 设置图的标题
ax.set_title("Feature importances using permutation on full model")
# 设置 y 轴的标签
ax.set_ylabel("Mean accuracy decrease")
# 调整图的布局
fig.tight_layout()
# 显示图形
plt.show()

# %%
# 使用两种方法检测到的主要特征相同。尽管相对重要性会有所变化。如图所示，MDI 不太可能完全忽略一个特征。
```