# `D:\src\scipysrc\scikit-learn\examples\feature_selection\plot_select_from_model_diabetes.py`

```
# %%
# Loading the data
# ----------------
#
# We first load the diabetes dataset which is available from within
# scikit-learn, and print its description:
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
print(diabetes.DESCR)



# %%
# Feature importance from coefficients
# ------------------------------------
#
# To get an idea of the importance of the features, we are going to use the
# :class:`~sklearn.linear_model.RidgeCV` estimator. The features with the
# highest absolute `coef_` value are considered the most important.
# We can observe the coefficients directly without needing to scale them (or
# scale the data) because from the description above, we know that the features
# were already standardized.
# For a more complete example on the interpretations of the coefficients of
# linear models, you may refer to
# :ref:`sphx_glr_auto_examples_inspection_plot_linear_model_coefficient_interpretation.py`.  # noqa: E501
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
importance = np.abs(ridge.coef_)
feature_names = np.array(diabetes.feature_names)
plt.bar(height=importance, x=feature_names)
plt.title("Feature importances via coefficients")
plt.show()



# %%
# Selecting features based on importance
# --------------------------------------
#
# Now we want to select the two features which are the most important according
# to the coefficients. The :class:`~sklearn.feature_selection.SelectFromModel`
# is meant just for that. :class:`~sklearn.feature_selection.SelectFromModel`
# accepts a `threshold` parameter and will select the features whose importance
# (defined by the coefficients) are above this threshold.
#
# Since we want to select only 2 features, we will set this threshold slightly
# above the coefficient of third most important feature.
from time import time

from sklearn.feature_selection import SelectFromModel

threshold = np.sort(importance)[-3] + 0.01

tic = time()
sfm = SelectFromModel(ridge, threshold=threshold).fit(X, y)
toc = time()
print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
print(f"Done in {toc - tic:.3f}s")



# %%
# Selecting features with Sequential Feature Selection
# ----------------------------------------------------
#
# Another way of selecting features is to use
# :class:`~sklearn.feature_selection.SequentialFeatureSelector`
# (SFS). SFS is a greedy procedure where, at each iteration, we choose the best
# new feature to add to our selected features based a cross-validation score.
# That is, we start with 0 features and choose the best single feature with the
# highest score. The procedure is repeated until we reach the desired number of
# selected features.
#
# We can also go in the reverse direction (backward SFS), *i.e.* start with all
# the features and greedily choose features to remove one by one. We illustrate
# both approaches here.

# 导入顺序特征选择器类 SequentialFeatureSelector 从 sklearn.feature_selection
from sklearn.feature_selection import SequentialFeatureSelector

# 开始计时，记录正向顺序选择的时间
tic_fwd = time()
# 创建正向顺序特征选择器对象 sfs_forward，选择 2 个特征，方向为 "forward"
sfs_forward = SequentialFeatureSelector(
    ridge, n_features_to_select=2, direction="forward"
).fit(X, y)
# 停止计时
toc_fwd = time()

# 开始计时，记录反向顺序选择的时间
tic_bwd = time()
# 创建反向顺序特征选择器对象 sfs_backward，选择 2 个特征，方向为 "backward"
sfs_backward = SequentialFeatureSelector(
    ridge, n_features_to_select=2, direction="backward"
).fit(X, y)
# 停止计时
toc_bwd = time()

# 打印正向顺序选择器选择的特征名称
print(
    "Features selected by forward sequential selection: "
    f"{feature_names[sfs_forward.get_support()]}"
)
# 打印正向顺序选择所用时间
print(f"Done in {toc_fwd - tic_fwd:.3f}s")
# 打印反向顺序选择器选择的特征名称
print(
    "Features selected by backward sequential selection: "
    f"{feature_names[sfs_backward.get_support()]}"
)
# 打印反向顺序选择所用时间
print(f"Done in {toc_bwd - tic_bwd:.3f}s")

# %%
# Interestingly, forward and backward selection have selected the same set of
# features. In general, this isn't the case and the two methods would lead to
# different results.
#
# We also note that the features selected by SFS differ from those selected by
# feature importance: SFS selects `bmi` instead of `s1`. This does sound
# reasonable though, since `bmi` corresponds to the third most important
# feature according to the coefficients. It is quite remarkable considering
# that SFS makes no use of the coefficients at all.
#
# To finish with, we should note that
# :class:`~sklearn.feature_selection.SelectFromModel` is significantly faster
# than SFS. Indeed, :class:`~sklearn.feature_selection.SelectFromModel` only
# needs to fit a model once, while SFS needs to cross-validate many different
# models for each of the iterations. SFS however works with any model, while
# :class:`~sklearn.feature_selection.SelectFromModel` requires the underlying
# estimator to expose a `coef_` attribute or a `feature_importances_`
# attribute. The forward SFS is faster than the backward SFS because it only
# needs to perform `n_features_to_select = 2` iterations, while the backward
# SFS needs to perform `n_features - n_features_to_select = 8` iterations.
#
# Using negative tolerance values
# -------------------------------
#
# :class:`~sklearn.feature_selection.SequentialFeatureSelector` can be used
# to remove features present in the dataset and return a
# smaller subset of the original features with `direction="backward"`
# and a negative value of `tol`.
#
# 导入所需的库和模块
import numpy as np  # 导入 NumPy 库
from sklearn.datasets import load_breast_cancer  # 从 sklearn 中导入乳腺癌数据集

# 载入乳腺癌数据集，包括 30 个特征和 569 个样本
breast_cancer_data = load_breast_cancer()
X, y = breast_cancer_data.data, breast_cancer_data.target  # 提取特征和目标
feature_names = np.array(breast_cancer_data.feature_names)  # 提取特征名称
print(breast_cancer_data.DESCR)  # 打印数据集的描述信息

# %%
# 使用 LogisticRegression 和 SequentialFeatureSelector 进行特征选择
from sklearn.linear_model import LogisticRegression  # 导入 Logistic 回归模型
from sklearn.metrics import roc_auc_score  # 导入 ROC AUC 评估指标
from sklearn.pipeline import make_pipeline  # 导入管道创建函数
from sklearn.preprocessing import StandardScaler  # 导入标准化处理模块

# 循环遍历不同的 tol 值进行特征选择
for tol in [-1e-2, -1e-3, -1e-4]:
    start = time()  # 记录开始时间
    feature_selector = SequentialFeatureSelector(
        LogisticRegression(),  # 使用 Logistic 回归作为基础模型
        n_features_to_select="auto",  # 自动选择特征数
        direction="backward",  # 向后选择特征
        scoring="roc_auc",  # 使用 ROC AUC 作为评分指标
        tol=tol,  # 设置收敛阈值
        n_jobs=2,  # 并行工作数
    )
    model = make_pipeline(StandardScaler(), feature_selector, LogisticRegression())
    model.fit(X, y)  # 拟合模型
    end = time()  # 记录结束时间
    print(f"\ntol: {tol}")
    print(f"Features selected: {feature_names[model[1].get_support()]}")  # 打印选择的特征名称
    print(f"ROC AUC score: {roc_auc_score(y, model.predict_proba(X)[:, 1]):.3f}")  # 打印 ROC AUC 分数
    print(f"Done in {end - start:.3f}s")  # 打印完成所需时间

# %%
# 观察到随着 tol 值接近零，所选择的特征数量有增加的趋势。
# 随着 tol 值接近零，特征选择所需的时间也会减少。
```