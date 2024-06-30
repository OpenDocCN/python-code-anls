# `D:\src\scipysrc\scikit-learn\examples\release_highlights\plot_release_highlights_0_23_0.py`

```
# ruff: noqa
"""
========================================
Release Highlights for scikit-learn 0.23
========================================

.. currentmodule:: sklearn

We are pleased to announce the release of scikit-learn 0.23! Many bug fixes
and improvements were added, as well as some new key features. We detail
below a few of the major features of this release. **For an exhaustive list of
all the changes**, please refer to the :ref:`release notes <release_notes_0_23>`.

To install the latest version (with pip)::

    pip install --upgrade scikit-learn

or with conda::

    conda install -c conda-forge scikit-learn

"""

##############################################################################
# Generalized Linear Models, and Poisson loss for gradient boosting
# -----------------------------------------------------------------
# Long-awaited Generalized Linear Models with non-normal loss functions are now
# available. In particular, three new regressors were implemented:
# :class:`~sklearn.linear_model.PoissonRegressor`,
# :class:`~sklearn.linear_model.GammaRegressor`, and
# :class:`~sklearn.linear_model.TweedieRegressor`. The Poisson regressor can be
# used to model positive integer counts, or relative frequencies. Read more in
# the :ref:`User Guide <Generalized_linear_regression>`. Additionally,
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` supports a new
# 'poisson' loss as well.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

n_samples, n_features = 1000, 20
rng = np.random.RandomState(0)
X = rng.randn(n_samples, n_features)
# positive integer target correlated with X[:, 5] with many zeros:
y = rng.poisson(lam=np.exp(X[:, 5]) / 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
glm = PoissonRegressor()
gbdt = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.01)
glm.fit(X_train, y_train)  # 训练 PoissonRegressor 模型
gbdt.fit(X_train, y_train)  # 训练 HistGradientBoostingRegressor 模型
print(glm.score(X_test, y_test))  # 打印 PoissonRegressor 在测试集上的评分
print(gbdt.score(X_test, y_test))  # 打印 HistGradientBoostingRegressor 在测试集上的评分

##############################################################################
# Rich visual representation of estimators
# -----------------------------------------
# Estimators can now be visualized in notebooks by enabling the
# `display='diagram'` option. This is particularly useful to summarise the
# structure of pipelines and other composite estimators, with interactivity to
# provide detail.  Click on the example image below to expand Pipeline
# elements.  See :ref:`visualizing_composite_estimators` for how you can use
# this feature.

from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression

set_config(display="diagram")
num_proc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

cat_proc = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = make_column_transformer(
    # 定义对数值特征的处理流程，包括缺失值填充和标准化
    (num_proc, ("feat1", "feat3")),
    # 定义对类别特征的处理流程，包括缺失值填充和独热编码
    (cat_proc, ("feat0", "feat2"))
)

clf = make_pipeline(preprocessor, LogisticRegression())
clf

##############################################################################
# Scalability and stability improvements to KMeans
# ------------------------------------------------
# The :class:`~sklearn.cluster.KMeans` estimator was entirely re-worked, and it
# is now significantly faster and more stable. In addition, the Elkan algorithm
# is now compatible with sparse matrices. The estimator uses OpenMP based
# parallelism instead of relying on joblib, so the `n_jobs` parameter has no
# effect anymore. For more details on how to control the number of threads,
# please refer to our :ref:`parallelism` notes.
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import completeness_score

rng = np.random.RandomState(0)
# 生成用于聚类的虚拟数据
X, y = make_blobs(random_state=rng)
# 将数据转换为稀疏矩阵格式
X = scipy.sparse.csr_matrix(X)
# 将数据分为训练集和测试集
X_train, X_test, _, y_test = train_test_split(X, y, random_state=rng)
# 使用自动选择初始聚类中心的方式拟合 KMeans 模型
kmeans = KMeans(n_init="auto").fit(X_train)
# 输出聚类模型预测结果的完整性得分
print(completeness_score(kmeans.predict(X_test), y_test))

##############################################################################
# Improvements to the histogram-based Gradient Boosting estimators
# ----------------------------------------------------------------
# Various improvements were made to
# :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor`. On top of the
# Poisson loss mentioned above, these estimators now support :ref:`sample
# weights <sw_hgbdt>`. Also, an automatic early-stopping criterion was added:
# early-stopping is enabled by default when the number of samples exceeds 10k.
# Finally, users can now define :ref:`monotonic constraints
# <monotonic_cst_gbdt>` to constrain the predictions based on the variations of
# specific features. In the following example, we construct a target that is
# generally positively correlated with the first feature, with some noise.
# Applying monotonic constraints allows the prediction to capture the global
# effect of the first feature, instead of fitting the noise. For a usecase
# example, see :ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py`.
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import HistGradientBoostingRegressor

n_samples = 500
rng = np.random.RandomState(0)
# 生成用于回归的虚拟数据
X = rng.randn(n_samples, 2)
# 生成服从正态分布的噪声数据，均值为0，标准差为0.01，样本数为n_samples
noise = rng.normal(loc=0.0, scale=0.01, size=n_samples)

# 构造回归目标变量y，其中包括线性项和正弦项，同时减去噪声
y = 5 * X[:, 0] + np.sin(10 * np.pi * X[:, 0]) - noise

# 使用默认参数训练无约束的梯度增强回归树模型
gbdt_no_cst = HistGradientBoostingRegressor().fit(X, y)

# 使用约束参数 [1, 0] 训练约束的梯度增强回归树模型
gbdt_cst = HistGradientBoostingRegressor(monotonic_cst=[1, 0]).fit(X, y)

# plot_partial_dependence 在版本1.2中已移除，建议使用 PartialDependenceDisplay
# 根据无约束模型 gbdt_no_cst 创建偏依赖展示对象 disp
disp = PartialDependenceDisplay.from_estimator(
    gbdt_no_cst,
    X,
    features=[0],
    feature_names=["feature 0"],
    line_kw={"linewidth": 4, "label": "unconstrained", "color": "tab:blue"},
)

# plot_partial_dependence 使用约束模型 gbdt_cst 创建偏依赖展示对象，绘制在 disp.axes_ 上
PartialDependenceDisplay.from_estimator(
    gbdt_cst,
    X,
    features=[0],
    line_kw={"linewidth": 4, "label": "constrained", "color": "tab:orange"},
    ax=disp.axes_,
)

# 在 disp.axes_[0, 0] 上绘制样本点，透明度为0.5，zorder设为-1，标签为"samples"，颜色为"tab:green"
disp.axes_[0, 0].plot(
    X[:, 0], y, "o", alpha=0.5, zorder=-1, label="samples", color="tab:green"
)

# 设置 disp.axes_[0, 0] 的纵坐标范围为 [-3, 3]
disp.axes_[0, 0].set_ylim(-3, 3)

# 设置 disp.axes_[0, 0] 的横坐标范围为 [-1, 1]
disp.axes_[0, 0].set_xlim(-1, 1)

# 添加图例
plt.legend()

# 显示图形
plt.show()

##############################################################################
# Lasso 和 ElasticNet 的样本权重支持
# ----------------------------------
# 现在线性回归器 :class:`~sklearn.linear_model.Lasso` 和 :class:`~sklearn.linear_model.ElasticNet`
# 支持样本权重。

# 导入所需的库和模块
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
import numpy as np

# 定义样本数和特征数
n_samples, n_features = 1000, 20

# 使用随机种子生成数据
rng = np.random.RandomState(0)
X, y = make_regression(n_samples, n_features, random_state=rng)

# 生成随机的样本权重
sample_weight = rng.rand(n_samples)

# 划分训练集和测试集，并使用样本权重
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weight, random_state=rng
)

# 创建 Lasso 模型
reg = Lasso()

# 使用训练集拟合模型，传入样本权重
reg.fit(X_train, y_train, sample_weight=sw_train)

# 打印模型在测试集上的得分，传入测试集的样本权重
print(reg.score(X_test, y_test, sw_test))
```