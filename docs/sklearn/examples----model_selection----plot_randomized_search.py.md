# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_randomized_search.py`

```
"""
=========================================================================
Comparing randomized search and grid search for hyperparameter estimation
=========================================================================

Compare randomized search and grid search for optimizing hyperparameters of a
linear SVM with SGD training.
All parameters that influence the learning are searched simultaneously
(except for the number of estimators, which poses a time / quality tradeoff).

The randomized search and the grid search explore exactly the same space of
parameters. The result in parameter settings is quite similar, while the run
time for randomized search is drastically lower.

The performance is may slightly worse for the randomized search, and is likely
due to a noise effect and would not carry over to a held-out test set.

Note that in practice, one would not search over this many different parameters
simultaneously using grid search, but pick only the ones deemed most important.

"""

from time import time  # 导入时间模块中的time函数

import numpy as np  # 导入NumPy库并使用别名np
import scipy.stats as stats  # 导入SciPy库中的stats模块，并使用别名stats

from sklearn.datasets import load_digits  # 从sklearn库中导入load_digits函数
from sklearn.linear_model import SGDClassifier  # 从sklearn库中导入SGDClassifier类
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # 从sklearn库中导入GridSearchCV和RandomizedSearchCV类

# get some data
X, y = load_digits(return_X_y=True, n_class=3)  # 调用load_digits函数加载手写数字数据集，返回特征X和标签y，仅包括3个类别的数据

# build a classifier
clf = SGDClassifier(loss="hinge", penalty="elasticnet", fit_intercept=True)  # 初始化SGDClassifier分类器，设置损失函数为'hinge'，正则化类型为'elasticnet'，拟合截距

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):  # 循环输出前n_top个最佳得分的模型信息
        candidates = np.flatnonzero(results["rank_test_score"] == i)  # 找到测试得分排名为i的模型索引
        for candidate in candidates:
            print("Model with rank: {0}".format(i))  # 输出模型排名
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],  # 输出平均验证分数
                    results["std_test_score"][candidate],  # 输出验证分数标准差
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))  # 输出模型的参数设置
            print("")

# specify parameters and distributions to sample from
param_dist = {
    "average": [True, False],  # 平均参数，布尔型True或False
    "l1_ratio": stats.uniform(0, 1),  # L1比率参数，均匀分布在[0, 1]之间
    "alpha": stats.loguniform(1e-2, 1e0),  # 正则化系数alpha，对数均匀分布在[0.01, 1]之间
}

# run randomized search
n_iter_search = 15  # 随机搜索的迭代次数
random_search = RandomizedSearchCV(
    clf, param_distributions=param_dist, n_iter=n_iter_search  # 创建RandomizedSearchCV对象，设置分类器、参数分布和迭代次数
)

start = time()  # 记录开始时间
random_search.fit(X, y)  # 运行随机搜索
print(
    "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
    % ((time() - start), n_iter_search)  # 输出随机搜索的耗时和参数设置数量
)
report(random_search.cv_results_)  # 输出随机搜索的结果报告

# use a full grid over all parameters
param_grid = {
    "average": [True, False],  # 平均参数，布尔型True或False
    "l1_ratio": np.linspace(0, 1, num=10),  # L1比率参数，等间隔取值在[0, 1]之间的10个数
    "alpha": np.power(10, np.arange(-2, 1, dtype=float)),  # 正则化系数alpha，取值为[0.01, 0.1, 1.0]
}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)  # 创建GridSearchCV对象，设置分类器和参数网格

start = time()  # 记录开始时间
grid_search.fit(X, y)  # 运行网格搜索
print(
    "GridSearchCV took %.2f seconds for %d candidate parameter settings."
    % (time() - start, len(grid_search.cv_results_["params"]))  # 输出网格搜索的耗时和参数设置数量
)
# 调用函数来输出网格搜索的交叉验证结果报告
report(grid_search.cv_results_)
```