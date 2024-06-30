# `D:\src\scipysrc\scikit-learn\examples\release_highlights\plot_release_highlights_1_3_0.py`

```
# %%
# Metadata Routing
# ----------------
# 引入了一种新的元数据路由方式，例如“sample_weight”，会影响到像:class:`pipeline.Pipeline`和
# :class:`model_selection.GridSearchCV`这样的元估计器如何路由元数据。尽管此功能的基础设施已包含在此版本中，
# 但该功能仍在进行中，并非所有元估计器都支持此新功能。有关更多信息，请参阅:ref:`Metadata Routing User Guide
# <metadata_routing>`。请注意，此功能仍在开发中，并且对大多数元估计器尚未实现。

# 第三方开发人员已经可以开始将此功能集成到其元估计器中。有关更多详细信息，请参阅:ref:`metadata routing developer guide
# <sphx_glr_auto_examples_miscellaneous_plot_metadata_routing.py>`。

# %%
# HDBSCAN: hierarchical density-based clustering
# ----------------------------------------------
# 最初托管在scikit-learn-contrib存储库中的:class:`cluster.HDBSCAN`已经被纳入scikit-learn中。它从原始实现中缺少
# 几个功能，这些功能将在未来的版本中添加。通过同时使用多个epsilon值执行修改版的:class:`cluster.DBSCAN`，
# :class:`cluster.HDBSCAN`可以发现密度变化的聚类，因此比:class:`cluster.DBSCAN`更能抵抗参数选择的影响。
# 更多细节请参阅:ref:`User Guide <hdbscan>`。

import numpy as np  # 导入numpy库
from sklearn.cluster import HDBSCAN  # 从sklearn库导入HDBSCAN聚类器
from sklearn.datasets import load_digits  # 从sklearn库导入load_digits数据集
from sklearn.metrics import v_measure_score  # 从sklearn库导入v_measure_score评估指标

X, true_labels = load_digits(return_X_y=True)  # 加载手写数字数据集，返回特征X和真实标签true_labels
print(f"number of digits: {len(np.unique(true_labels))}")  # 打印数据集中不同标签的数量

hdbscan = HDBSCAN(min_cluster_size=15).fit(X)  # 使用HDBSCAN聚类器拟合数据X，设定最小聚类大小为15
non_noisy_labels = hdbscan.labels_[hdbscan.labels_ != -1]  # 选择非噪声标签
print(f"number of clusters found: {len(np.unique(non_noisy_labels))}")  # 打印找到的聚类数目

print(v_measure_score(true_labels[hdbscan.labels_ != -1], non_noisy_labels))  # 计算非噪声标签的V-measure评分

# %%
# TargetEncoder: a new category encoding strategy
# -----------------------------------------------
# 适用于具有高基数的分类特征，:class:`preprocessing.TargetEncoder`根据属于该类别的观测的目标值的平均估计的收缩值来编码类别。
# 更多细节请参考 :ref:`User Guide <target_encoder>`。
# 导入 numpy 库，并引入 TargetEncoder 类
import numpy as np
from sklearn.preprocessing import TargetEncoder

# 创建一个包含多个重复类别的数组 X 和对应的目标变量 y
X = np.array([["cat"] * 30 + ["dog"] * 20 + ["snake"] * 38], dtype=object).T
y = [90.3] * 30 + [20.4] * 20 + [21.2] * 38

# 初始化 TargetEncoder 对象
enc = TargetEncoder(random_state=0)

# 使用 TargetEncoder 对象对 X 进行拟合转换
X_trans = enc.fit_transform(X, y)

# 获取编码后的类别与其对应的平均目标值的字典
enc.encodings_

# %%
# 决策树中对缺失值的支持
# ----------------------------------------
# :class:`tree.DecisionTreeClassifier` 和 :class:`tree.DecisionTreeRegressor` 现在支持缺失值。
# 对于非缺失数据的每个可能阈值，分裂器将评估带有所有缺失值的分裂情况，
# 这些值可能分配给左侧节点或右侧节点。
# 更多详细信息请参考 :ref:`User Guide <tree_missing_value_support>` 或查看
# :ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py` 中的使用案例，
# 了解在 :class:`~ensemble.HistGradientBoostingRegressor` 中此功能的应用。
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 创建一个带有缺失值的数组 X 和对应的目标变量 y
X = np.array([0, 1, 6, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]

# 使用 DecisionTreeClassifier 对象拟合数据
tree = DecisionTreeClassifier(random_state=0).fit(X, y)

# 对训练数据进行预测
tree.predict(X)

# %%
# 新的显示类 :class:`~model_selection.ValidationCurveDisplay`
# ------------------------------------------------------------
# 现在可以使用 :class:`model_selection.ValidationCurveDisplay` 来绘制
# :func:`model_selection.validation_curve` 的结果。
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ValidationCurveDisplay

# 创建一个分类数据集 X 和对应的目标变量 y
X, y = make_classification(1000, 10, random_state=0)

# 使用 ValidationCurveDisplay 类绘制验证曲线
_ = ValidationCurveDisplay.from_estimator(
    LogisticRegression(),
    X,
    y,
    param_name="C",
    param_range=np.geomspace(1e-5, 1e3, num=9),
    score_type="both",
    score_name="Accuracy",
)

# %%
# 梯度提升中的 Gamma 损失函数
# --------------------------------
# :class:`ensemble.HistGradientBoostingRegressor` 支持通过 `loss="gamma"` 使用 Gamma 偏差损失函数。
# 此损失函数适用于具有右偏分布的严格正目标建模。
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_low_rank_matrix
from sklearn.ensemble import HistGradientBoostingRegressor

# 创建一个低秩矩阵 X 和对应的目标变量 y
n_samples, n_features = 500, 10
rng = np.random.RandomState(0)
X = make_low_rank_matrix(n_samples, n_features, random_state=rng)
coef = rng.uniform(low=-10, high=20, size=n_features)
y = rng.gamma(shape=2, scale=np.exp(X @ coef) / 2)

# 初始化 HistGradientBoostingRegressor 对象，并使用 Gamma 损失函数
gbdt = HistGradientBoostingRegressor(loss="gamma")

# 使用交叉验证评估模型性能
cross_val_score(gbdt, X, y).mean()

# %%
# 在 :class:`~preprocessing.OrdinalEncoder` 中对稀有类别的分组
# ------------------------------------------------------------------------
# 类似于 :class:`preprocessing.OneHotEncoder`，现在 :class:`preprocessing.OrdinalEncoder` 类
# 也支持聚合稀有类别。
# 导入 `OrdinalEncoder` 类，用于将分类数据转换为有序整数编码
from sklearn.preprocessing import OrdinalEncoder
# 导入 `numpy` 库，并用 `np` 作为别名
import numpy as np

# 创建一个包含不同动物名称的二维 `numpy` 数组 `X`
X = np.array(
    [["dog"] * 5 + ["cat"] * 20 + ["rabbit"] * 10 + ["snake"] * 3], dtype=object
).T

# 创建 `OrdinalEncoder` 对象 `enc`，并使用 `fit` 方法拟合数据 `X`
# 设置 `min_frequency=6` 参数，以便仅对出现频率至少为 6 次的分类进行编码
enc = OrdinalEncoder(min_frequency=6).fit(X)

# 访问 `OrdinalEncoder` 对象 `enc` 的 `infrequent_categories_` 属性，
# 返回少见分类的列表，即出现频率低于 `min_frequency` 阈值的分类
enc.infrequent_categories_
```