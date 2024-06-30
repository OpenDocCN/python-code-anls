# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_successive_halving_iterations.py`

```
"""
Successive Halving Iterations
=============================

This example illustrates how a successive halving search
(:class:`~sklearn.model_selection.HalvingGridSearchCV` and
:class:`~sklearn.model_selection.HalvingRandomSearchCV`)
iteratively chooses the best parameter combination out of
multiple candidates.

"""

import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算
import pandas as pd  # 导入 pandas 库，用于数据处理
from scipy.stats import randint  # 从 scipy.stats 导入 randint 函数，用于生成随机整数

from sklearn import datasets  # 导入 sklearn 的 datasets 模块，用于数据集生成
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.experimental import enable_halving_search_cv  # 启用 halving search cv 实验性模块
from sklearn.model_selection import HalvingRandomSearchCV  # 导入 HalvingRandomSearchCV 类，用于带递归缩减的随机搜索

# %%
# We first define the parameter space and train a
# :class:`~sklearn.model_selection.HalvingRandomSearchCV` instance.

rng = np.random.RandomState(0)  # 创建随机数生成器对象 rng

X, y = datasets.make_classification(n_samples=400, n_features=12, random_state=rng)  # 生成分类数据集 X 和 y

clf = RandomForestClassifier(n_estimators=20, random_state=rng)  # 创建随机森林分类器 clf

param_dist = {
    "max_depth": [3, None],  # 最大深度的候选值
    "max_features": randint(1, 6),  # 最大特征数的候选值范围
    "min_samples_split": randint(2, 11),  # 最小样本分割数的候选值范围
    "bootstrap": [True, False],  # 是否启用 bootstrap 的候选值
    "criterion": ["gini", "entropy"],  # 分裂标准的候选值
}

rsh = HalvingRandomSearchCV(
    estimator=clf, param_distributions=param_dist, factor=2, random_state=rng  # 创建 HalvingRandomSearchCV 实例 rsh
)
rsh.fit(X, y)  # 在数据集 X, y 上拟合模型

# %%
# We can now use the `cv_results_` attribute of the search estimator to inspect
# and plot the evolution of the search.

results = pd.DataFrame(rsh.cv_results_)  # 创建包含搜索结果的 DataFrame
results["params_str"] = results.params.apply(str)  # 将参数转换为字符串格式
results.drop_duplicates(subset=("params_str", "iter"), inplace=True)  # 删除重复的参数配置
mean_scores = results.pivot(
    index="iter", columns="params_str", values="mean_test_score"  # 生成按迭代次数和参数配置的平均测试分数的透视表
)
ax = mean_scores.plot(legend=False, alpha=0.6)  # 绘制平均测试分数的折线图

labels = [
    f"iter={i}\nn_samples={rsh.n_resources_[i]}\nn_candidates={rsh.n_candidates_[i]}"
    for i in range(rsh.n_iterations_)
]  # 设置图表的标签信息

ax.set_xticks(range(rsh.n_iterations_))  # 设置 x 轴刻度
ax.set_xticklabels(labels, rotation=45, multialignment="left")  # 设置 x 轴标签
ax.set_title("Scores of candidates over iterations")  # 设置图表标题
ax.set_ylabel("mean test score", fontsize=15)  # 设置 y 轴标签
ax.set_xlabel("iterations", fontsize=15)  # 设置 x 轴标签
plt.tight_layout()  # 调整布局
plt.show()  # 显示图表

# %%
# Number of candidates and amount of resource at each iteration
# -------------------------------------------------------------
#
# At the first iteration, a small amount of resources is used. The resource
# here is the number of samples that the estimators are trained on. All
# candidates are evaluated.
#
# At the second iteration, only the best half of the candidates is evaluated.
# The number of allocated resources is doubled: candidates are evaluated on
# twice as many samples.
#
# This process is repeated until the last iteration, where only 2 candidates
# are left. The best candidate is the candidate that has the best score at the
# last iteration.
```