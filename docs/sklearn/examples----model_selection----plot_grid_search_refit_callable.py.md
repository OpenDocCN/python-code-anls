# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_grid_search_refit_callable.py`

```
"""
==================================================
Balance model complexity and cross-validated score
==================================================

This example balances model complexity and cross-validated score by
finding a decent accuracy within 1 standard deviation of the best accuracy
score while minimizing the number of PCA components [1].

The figure shows the trade-off between cross-validated score and the number
of PCA components. The balanced case is when n_components=10 and accuracy=0.88,
which falls into the range within 1 standard deviation of the best accuracy
score.

[1] Hastie, T., Tibshirani, R.,, Friedman, J. (2001). Model Assessment and
Selection. The Elements of Statistical Learning (pp. 219-260). New York,
NY, USA: Springer New York Inc..

"""

# Author: Wenhao Zhang <wenhaoz@ucla.edu>

import matplotlib.pyplot as plt  # 导入绘图库matplotlib
import numpy as np  # 导入数值计算库numpy

from sklearn.datasets import load_digits  # 导入手写数字数据集加载函数
from sklearn.decomposition import PCA  # 导入PCA降维方法
from sklearn.model_selection import GridSearchCV  # 导入网格搜索交叉验证模块
from sklearn.pipeline import Pipeline  # 导入管道模块
from sklearn.svm import LinearSVC  # 导入线性支持向量机模型


def lower_bound(cv_results):
    """
    Calculate the lower bound within 1 standard deviation
    of the best `mean_test_scores`.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`

    Returns
    -------
    float
        Lower bound within 1 standard deviation of the
        best `mean_test_score`.
    """
    best_score_idx = np.argmax(cv_results["mean_test_score"])  # 获取最佳测试分数的索引

    return (
        cv_results["mean_test_score"][best_score_idx]  # 返回最佳测试分数
        - cv_results["std_test_score"][best_score_idx]  # 减去最佳测试分数的标准差
    )


def best_low_complexity(cv_results):
    """
    Balance model complexity with cross-validated score.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`.

    Return
    ------
    int
        Index of a model that has the fewest PCA components
        while has its test score within 1 standard deviation of the best
        `mean_test_score`.
    """
    threshold = lower_bound(cv_results)  # 获取低于最佳测试分数一个标准差的下限
    candidate_idx = np.flatnonzero(cv_results["mean_test_score"] >= threshold)  # 找出测试分数高于下限的索引
    best_idx = candidate_idx[
        cv_results["param_reduce_dim__n_components"][candidate_idx].argmin()
    ]  # 从符合条件的索引中选取PCA组件最少的模型索引
    return best_idx


pipe = Pipeline(
    [
        ("reduce_dim", PCA(random_state=42)),  # PCA降维步骤，使用随机种子42
        ("classify", LinearSVC(random_state=42, C=0.01)),  # 线性SVM分类器，使用随机种子42和C=0.01
    ]
)

param_grid = {"reduce_dim__n_components": [6, 8, 10, 12, 14]}  # 不同PCA组件数量的参数网格

grid = GridSearchCV(
    pipe,
    cv=10,  # 10折交叉验证
    n_jobs=1,  # 使用单线程
    param_grid=param_grid,  # 参数网格为PCA组件数量
    scoring="accuracy",  # 评分标准为准确率
    refit=best_low_complexity,  # 使用最佳复杂度的模型进行再拟合
)
X, y = load_digits(return_X_y=True)  # 加载手写数字数据集
grid.fit(X, y)  # 在数据上进行网格搜索交叉验证

n_components = grid.cv_results_["param_reduce_dim__n_components"]  # 获取PCA组件数量
test_scores = grid.cv_results_["mean_test_score"]  # 获取平均测试分数

plt.figure()  # 创建新的绘图窗口
plt.bar(n_components, test_scores, width=1.3, color="b")  # 绘制柱状图，显示PCA组件数量与测试分数

lower = lower_bound(grid.cv_results_)  # 计算最低分数的下限
# 添加水平虚线到图表中，表示测试分数的最大值，线型为虚线，颜色为黄色，标签为“Best score”
plt.axhline(np.max(test_scores), linestyle="--", color="y", label="Best score")

# 添加水平虚线到图表中，表示下限值，线型为虚线，颜色为灰色，标签为“Best score - 1 std”
plt.axhline(lower, linestyle="--", color=".5", label="Best score - 1 std")

# 设置图表标题为“Balance model complexity and cross-validated score”
plt.title("Balance model complexity and cross-validated score")

# 设置 x 轴标签为“Number of PCA components used”
plt.xlabel("Number of PCA components used")

# 设置 y 轴标签为“Digit classification accuracy”
plt.ylabel("Digit classification accuracy")

# 设置 x 轴刻度为 n_components 列表的值
plt.xticks(n_components.tolist())

# 设置 y 轴范围为 (0, 1.0)
plt.ylim((0, 1.0))

# 设置图例位置为左上角
plt.legend(loc="upper left")

# 获取网格搜索结果中的最佳索引
best_index_ = grid.best_index_

# 打印输出最佳索引的信息
print("The best_index_ is %d" % best_index_)

# 打印输出所选取的主成分数目
print("The n_components selected is %d" % n_components[best_index_])

# 打印输出对应的准确率分数
print(
    "The corresponding accuracy score is %.2f"
    % grid.cv_results_["mean_test_score"][best_index_]
)

# 显示图表
plt.show()
```