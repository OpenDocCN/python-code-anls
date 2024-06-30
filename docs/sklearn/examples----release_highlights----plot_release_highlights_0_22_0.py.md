# `D:\src\scipysrc\scikit-learn\examples\release_highlights\plot_release_highlights_0_22_0.py`

```
# %%
# New plotting API
# ----------------
#
# A new plotting API is introduced for creating visualizations, providing flexibility
# in adjusting plot visuals without recomputation. It supports combining multiple plots
# in a single figure. The example demonstrates `plot_roc_curve`, alongside other
# utilities such as `plot_partial_dependence`, `plot_precision_recall_curve`, and
# `plot_confusion_matrix`. For more details, refer to the :ref:`User Guide <visualizations>`.

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库用于绘图

from sklearn.datasets import make_classification  # 导入 make_classification 函数生成分类数据
from sklearn.ensemble import RandomForestClassifier  # 导入 RandomForestClassifier 随机森林分类器

# from sklearn.metrics import plot_roc_curve  # 已在版本 1.2 中移除 plot_roc_curve 函数
from sklearn.metrics import RocCurveDisplay  # 导入 RocCurveDisplay 类用于绘制 ROC 曲线显示
from sklearn.model_selection import train_test_split  # 导入 train_test_split 函数用于数据集划分
from sklearn.svm import SVC  # 导入 SVC 支持向量机分类器

X, y = make_classification(random_state=0)  # 生成随机分类数据集 X 和 y
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # 将数据集划分为训练集和测试集

svc = SVC(random_state=42)  # 创建 SVC 模型对象
svc.fit(X_train, y_train)  # 训练 SVC 模型
rfc = RandomForestClassifier(random_state=42)  # 创建 RandomForestClassifier 模型对象
rfc.fit(X_train, y_train)  # 训练 RandomForestClassifier 模型

# plot_roc_curve has been removed in version 1.2. From 1.2, use RocCurveDisplay instead.
# svc_disp = plot_roc_curve(svc, X_test, y_test)
# rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=svc_disp.ax_)
svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)  # 使用 svc 模型绘制 ROC 曲线显示
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=svc_disp.ax_)  # 使用 rfc 模型绘制 ROC 曲线显示，并指定 ax 参数

rfc_disp.figure_.suptitle("ROC curve comparison")  # 设置 ROC 曲线比较图的标题

plt.show()  # 显示图形

# %%
# Stacking Classifier and Regressor
# ---------------------------------
# :class:`~ensemble.StackingClassifier` and
# :class:`~ensemble.StackingRegressor`
# allow you to have a stack of estimators with a final classifier or
# a regressor.
# Stacked generalization consists in stacking the output of individual
# estimators and use a classifier to compute the final prediction. Stacking
# allows to use the strength of each individual estimator by using their output
# as input of a final estimator.
# Base estimators are fitted on the full ``X`` while
# the final estimator is trained using cross-validated predictions of the
# base estimators using ``cross_val_predict``.
#
# Read more in the :ref:`User Guide <stacking>`.

from sklearn.datasets import load_iris  # 导入 load_iris 函数用于加载鸢尾花数据集
from sklearn.ensemble import StackingClassifier  # 导入 StackingClassifier 类用于堆叠分类器
from sklearn.linear_model import LogisticRegression  # 导入 LogisticRegression 类用于逻辑回归分类
# 导入所需的模块和类
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.pipeline import make_pipeline  # 导入创建管道的函数
from sklearn.preprocessing import StandardScaler  # 导入标准化器
from sklearn.svm import LinearSVC  # 导入线性支持向量分类器

# 加载鸢尾花数据集的特征和目标变量
X, y = load_iris(return_X_y=True)

# 定义多个基础估计器
estimators = [
    ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),  # 随机森林分类器
    ("svr", make_pipeline(StandardScaler(), LinearSVC(dual="auto", random_state=42))),  # 标准化 + 线性支持向量分类器
]

# 创建集成分类器，使用Stacking方法
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 训练集成分类器，并计算测试集上的得分
clf.fit(X_train, y_train).score(X_test, y_test)

# %%
# 基于置换的特征重要性
# ------------------------------------
#
# :func:`inspection.permutation_importance` 函数可以用于估计每个特征的重要性，
# 适用于任何已拟合的估计器：

import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库

from sklearn.datasets import make_classification  # 导入生成分类数据集的函数
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.inspection import permutation_importance  # 导入置换重要性函数

# 生成一个具有5个特征的分类数据集
X, y = make_classification(random_state=0, n_features=5, n_informative=3)

# 创建特征名称数组
feature_names = np.array([f"x_{i}" for i in range(X.shape[1])])

# 使用随机森林拟合数据
rf = RandomForestClassifier(random_state=0).fit(X, y)

# 计算置换重要性
result = permutation_importance(rf, X, y, n_repeats=10, random_state=0, n_jobs=2)

# 创建箱线图展示特征重要性
fig, ax = plt.subplots()
sorted_idx = result.importances_mean.argsort()
ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=feature_names[sorted_idx]
)
ax.set_title("每个特征的置换重要性")
ax.set_ylabel("特征")
fig.tight_layout()
plt.show()

# %%
# 梯度提升中对缺失值的本地支持
# -------------------------------------------------------
#
# :class:`ensemble.HistGradientBoostingClassifier` 和 :class:`ensemble.HistGradientBoostingRegressor`
# 现在原生支持缺失值（NaN）。这意味着在训练或预测时无需填充数据。

from sklearn.ensemble import HistGradientBoostingClassifier  # 导入直方图梯度提升分类器

# 创建包含NaN值的数据集
X = np.array([0, 1, 2, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]

# 使用直方图梯度提升分类器拟合数据
gbdt = HistGradientBoostingClassifier(min_samples_leaf=1).fit(X, y)

# 打印预测结果
print(gbdt.predict(X))

# %%
# 预计算稀疏最近邻图
# ------------------------------------------
# 大多数基于最近邻图的估计器现在接受预计算的稀疏图作为输入，
# 可以在多个估计器拟合中重复使用同一图。在管道中使用这一特性时，
# 可以使用 `memory` 参数和两个新的转换器，
# :class:`neighbors.KNeighborsTransformer` 和
# :class:`neighbors.RadiusNeighborsTransformer`。预计算
# 也可以由自定义估计器执行，以使用替代的实现，例如近似最近邻方法。
# 更多详细信息请参阅 :ref:`用户指南 <neighbors_transformer>`。

from tempfile import TemporaryDirectory  # 导入临时目录类

from sklearn.manifold import Isomap  # 导入等距映射算法
# 导入所需的模块和函数
from sklearn.neighbors import KNeighborsTransformer  # 导入K近邻转换器
from sklearn.pipeline import make_pipeline  # 导入创建管道的函数

# 创建一个示例数据集 X 和目标向量 y
X, y = make_classification(random_state=0)

# 使用临时目录作为内存缓存，创建包含两个转换器的管道估计器
with TemporaryDirectory(prefix="sklearn_cache_") as tmpdir:
    estimator = make_pipeline(
        KNeighborsTransformer(n_neighbors=10, mode="distance"),  # 使用K近邻转换器进行距离模式的转换
        Isomap(n_neighbors=10, metric="precomputed"),  # 使用Isomap进行预计算的近邻转换
        memory=tmpdir,  # 设定内存缓存的临时目录
    )
    estimator.fit(X)  # 对数据集 X 进行拟合

    # 可以通过设置参数来减少近邻数，而无需重新计算图形。
    estimator.set_params(isomap__n_neighbors=5)  # 设置Isomap的近邻数为5
    estimator.fit(X)  # 重新对数据集 X 进行拟合

# %%
# 基于KNN的插补
# ------------------------------------
# 现在我们支持使用k近邻算法进行插补，以完成缺失值的填充。
#
# 每个样本的缺失值将使用训练集中找到的``n_neighbors``个最近邻的均值进行插补。
# 如果两个样本在没有缺失值的特征上很接近，则它们被视为接近。
# 默认情况下，使用支持缺失值的欧几里得距离度量，
# :func:`~sklearn.metrics.pairwise.nan_euclidean_distances` 来寻找最近的邻居。
#
# 在 :ref:`User Guide <knnimpute>` 中了解更多信息。

from sklearn.impute import KNNImputer  # 导入KNN插补器

X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2)  # 创建一个使用2个近邻的KNN插补器
print(imputer.fit_transform(X))  # 输出插补后的数据

# %%
# 树的剪枝
# ------------
#
# 现在可以在构建树之后对大多数基于树的估计器进行剪枝。剪枝基于最小成本复杂性。
# 在 :ref:`User Guide <minimal_cost_complexity_pruning>` 中详细阅读详情。

X, y = make_classification(random_state=0)

rf = RandomForestClassifier(random_state=0, ccp_alpha=0).fit(X, y)
print(
    "平均节点数（未剪枝） {:.1f}".format(
        np.mean([e.tree_.node_count for e in rf.estimators_])
    )
)

rf = RandomForestClassifier(random_state=0, ccp_alpha=0.05).fit(X, y)
print(
    "平均节点数（剪枝后） {:.1f}".format(
        np.mean([e.tree_.node_count for e in rf.estimators_])
    )
)

# %%
# 从OpenML获取数据框架
# -------------------------------
# :func:`datasets.fetch_openml` 现在可以返回 pandas 数据框架，因此能够正确处理包含异构数据的数据集：

from sklearn.datasets import fetch_openml  # 导入fetch_openml函数

titanic = fetch_openml("titanic", version=1, as_frame=True, parser="pandas")
print(titanic.data.head()[["pclass", "embarked"]])

# %%
# 检查估计器的scikit-learn兼容性
# ---------------------------------------------------
# 开发者可以使用 :func:`~utils.estimator_checks.check_estimator` 检查其scikit-learn兼容估计器的兼容性。
# 例如，``check_estimator(LinearSVC())`` 通过测试。
#
# 现在我们提供了一个 ``pytest`` 特定的装饰器，允许 ``pytest`` 独立运行所有检查并报告失败的检查。
#
# ..note::
#   此条目在版本0.24中略有更新，通过类
# 导入 LogisticRegression 和 DecisionTreeRegressor 类，用于测试模型兼容性
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
# 导入 parametrize_with_checks 函数，用于测试估计器的一致性
from sklearn.utils.estimator_checks import parametrize_with_checks

# 使用 parametrize_with_checks 函数对 LogisticRegression 和 DecisionTreeRegressor 实例化对象进行测试
@parametrize_with_checks([LogisticRegression(), DecisionTreeRegressor()])
# 定义测试函数 test_sklearn_compatible_estimator，接受一个估计器和一个检查函数作为参数
def test_sklearn_compatible_estimator(estimator, check):
    # 调用检查函数 check 对传入的估计器进行测试
    check(estimator)


# %%
# ROC AUC 现在支持多类分类
# ----------------------------------------------
# :func:`~sklearn.metrics.roc_auc_score` 函数现在也支持多类分类。
# 目前支持两种平均策略：一对一算法计算成对 ROC AUC 分数的平均值，
# 一对多算法计算每个类别与所有其他类别的 ROC AUC 分数的平均值。
# 在这两种情况下，多类 ROC AUC 分数是根据模型对样本属于特定类别的概率估计计算得出的。
# OvO 和 OvR 算法支持均匀加权（``average='macro'``）和按流行度加权（``average='weighted'``）两种平均方式。
#
# 更多详细信息，请参阅 :ref:`User Guide <roc_metrics>`.


# 导入 make_classification 和 roc_auc_score 函数
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
# 导入 SVC 模型
from sklearn.svm import SVC

# 使用 make_classification 创建数据集 X, y
X, y = make_classification(n_classes=4, n_informative=16)
# 实例化 SVC 模型，选择 "ovo" 决策函数形式，并启用概率估计
clf = SVC(decision_function_shape="ovo", probability=True).fit(X, y)
# 计算并打印多类别 "ovo" ROC AUC 分数
print(roc_auc_score(y, clf.predict_proba(X), multi_class="ovo"))
```