# `D:\src\scipysrc\scikit-learn\examples\exercises\plot_cv_diabetes.py`

```
"""
=========================================================
Cross-validation on diabetes Dataset Exercise
=========================================================

A tutorial exercise which uses cross-validation with linear models.

This exercise is used in the :ref:`cv_estimators_tut` part of the
:ref:`model_selection_tut` section of the :ref:`stat_learn_tut_index`.

"""

# %%
# Load dataset and apply GridSearchCV
# -----------------------------------
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
import numpy as np  # 导入numpy库用于数值计算

from sklearn import datasets  # 导入sklearn中的数据集模块
from sklearn.linear_model import Lasso  # 导入Lasso回归模型
from sklearn.model_selection import GridSearchCV  # 导入GridSearchCV模块，用于参数调优

X, y = datasets.load_diabetes(return_X_y=True)  # 加载糖尿病数据集
X = X[:150]  # 只使用前150个样本
y = y[:150]

lasso = Lasso(random_state=0, max_iter=10000)  # 初始化Lasso回归模型
alphas = np.logspace(-4, -0.5, 30)  # 设定要调优的alpha值范围

tuned_parameters = [{"alpha": alphas}]  # 构建参数字典，用于GridSearchCV
n_folds = 5  # 设定交叉验证的折数

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)  # 构建GridSearchCV对象
clf.fit(X, y)  # 在数据集上进行参数搜索
scores = clf.cv_results_["mean_test_score"]  # 获取交叉验证的平均测试分数
scores_std = clf.cv_results_["std_test_score"]  # 获取交叉验证的测试分数标准差

# %%
# Plot error lines showing +/- std. errors of the scores
# ------------------------------------------------------

plt.figure().set_size_inches(8, 6)  # 创建一个图像对象，并设定大小
plt.semilogx(alphas, scores)  # 绘制alpha与CV分数的关系图

std_error = scores_std / np.sqrt(n_folds)  # 计算CV分数的标准误差

plt.semilogx(alphas, scores + std_error, "b--")  # 绘制上方的误差线
plt.semilogx(alphas, scores - std_error, "b--")  # 绘制下方的误差线

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)  # 填充误差区域

plt.ylabel("CV score +/- std error")  # 设定y轴标签
plt.xlabel("alpha")  # 设定x轴标签
plt.axhline(np.max(scores), linestyle="--", color=".5")  # 绘制最大CV分数的水平线
plt.xlim([alphas[0], alphas[-1]])  # 设定x轴范围

# %%
# Bonus: how much can you trust the selection of alpha?
# -----------------------------------------------------

# To answer this question we use the LassoCV object that sets its alpha
# parameter automatically from the data by internal cross-validation (i.e. it
# performs cross-validation on the training data it receives).
# We use external cross-validation to see how much the automatically obtained
# alphas differ across different cross-validation folds.

from sklearn.linear_model import LassoCV  # 导入LassoCV模块，用于自动选择alpha值
from sklearn.model_selection import KFold  # 导入KFold模块，用于交叉验证

lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=10000)  # 初始化LassoCV对象
k_fold = KFold(3)  # 创建3折交叉验证对象

print("Answer to the bonus question:", "how much can you trust the selection of alpha?")
print()
print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k, (train, test) in enumerate(k_fold.split(X, y)):  # 遍历每一折交叉验证
    lasso_cv.fit(X[train], y[train])  # 在训练集上拟合模型
    print(
        "[fold {0}] alpha: {1:.5f}, score: {2:.5f}".format(
            k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])
        )
    )  # 打印每一折的最佳alpha值和对应的分数
print()
print("Answer: Not very much since we obtained different alphas for different")
print("subsets of the data and moreover, the scores for these alphas differ")
print("quite substantially.")

plt.show()  # 展示绘制的图像
```