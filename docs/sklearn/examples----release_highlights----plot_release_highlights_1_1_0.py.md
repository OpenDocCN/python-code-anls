# `D:\src\scipysrc\scikit-learn\examples\release_highlights\plot_release_highlights_1_1_0.py`

```
# ruff: noqa
"""
=======================================
Release Highlights for scikit-learn 1.1
=======================================

.. currentmodule:: sklearn

We are pleased to announce the release of scikit-learn 1.1! Many bug fixes
and improvements were added, as well as some new key features. We detail
below a few of the major features of this release. **For an exhaustive list of
all the changes**, please refer to the :ref:`release notes <release_notes_1_1>`.

To install the latest version (with pip)::

    pip install --upgrade scikit-learn

or with conda::

    conda install -c conda-forge scikit-learn

"""

# %%
# .. _quantile_support_hgbdt:
#
# Quantile loss in :class:`~ensemble.HistGradientBoostingRegressor`
# -----------------------------------------------------------------
# :class:`~ensemble.HistGradientBoostingRegressor` can model quantiles with
# `loss="quantile"` and the new parameter `quantile`.
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt

# Simple regression function for X * cos(X)
rng = np.random.RandomState(42)
X_1d = np.linspace(0, 10, num=2000)
X = X_1d.reshape(-1, 1)
y = X_1d * np.cos(X_1d) + rng.normal(scale=X_1d / 3)

# Define quantiles for prediction
quantiles = [0.95, 0.5, 0.05]
parameters = dict(loss="quantile", max_bins=32, max_iter=50)

# Create HistGradientBoostingRegressor instances for each quantile
hist_quantiles = {
    f"quantile={quantile:.2f}": HistGradientBoostingRegressor(
        **parameters, quantile=quantile
    ).fit(X, y)
    for quantile in quantiles
}

# Plotting
fig, ax = plt.subplots()
ax.plot(X_1d, y, "o", alpha=0.5, markersize=1)
for quantile, hist in hist_quantiles.items():
    ax.plot(X_1d, hist.predict(X), label=quantile)
_ = ax.legend(loc="lower left")

# %%
# For a usecase example, see
# :ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py`

# %%
# `get_feature_names_out` Available in all Transformers
# -----------------------------------------------------
# :term:`get_feature_names_out` is now available in all Transformers. This enables
# :class:`~pipeline.Pipeline` to construct the output feature names for more complex
# pipelines:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

# Fetch dataset
X, y = fetch_openml(
    "titanic", version=1, as_frame=True, return_X_y=True, parser="pandas"
)

# Define numeric and categorical features
numeric_features = ["age", "fare"]
numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
categorical_features = ["embarked", "pclass"]

# Preprocessor to handle both numeric and categorical features
preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_features),  # Numeric feature processing
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),  # Categorical feature processing
            categorical_features,
        ),
    ],
    verbose_feature_names_out=False,  # Disable verbose feature names output
)
# 创建一个机器学习流水线，包括数据预处理、特征选择和逻辑回归模型
log_reg = make_pipeline(preprocessor, SelectKBest(k=7), LogisticRegression())
# 使用流水线训练模型，X 是特征数据，y 是目标变量
log_reg.fit(X, y)


# %%
# 在这里，我们对流水线进行切片，保留除最后一步外的所有步骤。这个流水线切片的输出
# 特征名将作为逻辑回归模型的输入特征。这些特征名直接对应于逻辑回归模型的系数：
import pandas as pd

# 获取流水线除最后一步的输出特征名
log_reg_input_features = log_reg[:-1].get_feature_names_out()
# 绘制逻辑回归模型的系数条形图
pd.Series(log_reg[-1].coef_.ravel(), index=log_reg_input_features).plot.bar()
plt.tight_layout()


# %%
# 将:class:`~preprocessing.OneHotEncoder`中的不常见类别进行分组
# -----------------------------------------------------------------------
# :class:`~preprocessing.OneHotEncoder` 支持将不常见的类别聚合为每个特征的单个输出。
# 启用聚合不常见类别的参数有 `min_frequency` 和 `max_categories`。
# 详细信息请参阅 :ref:`用户指南 <encoder_infrequent_categories>`。
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 创建包含不同数量不常见类别的特征数据 X
X = np.array(
    [["dog"] * 5 + ["cat"] * 20 + ["rabbit"] * 10 + ["snake"] * 3], dtype=object
).T
# 使用 OneHotEncoder 进行拟合，设置最小频率为 6，稀疏输出为 False
enc = OneHotEncoder(min_frequency=6, sparse_output=False).fit(X)
# 查看被编码器识别为不常见类别的类别
enc.infrequent_categories_

# %%
# 因为 dog 和 snake 是不常见类别，它们在转换时被分组在一起：
encoded = enc.transform(np.array([["dog"], ["snake"], ["cat"], ["rabbit"]]))
# 创建一个 DataFrame 来展示编码后的特征
pd.DataFrame(encoded, columns=enc.get_feature_names_out())

# %%
# 性能改进
# ------------------------
# 对于稠密的 float64 数据集，成对距离的计算已被重构，以更好地利用非阻塞线程并行性。
# 例如，:meth:`neighbors.NearestNeighbors.kneighbors` 和
# :meth:`neighbors.NearestNeighbors.radius_neighbors` 的速度分别可以提高 ×20 和 ×5。
# 简而言之，以下函数和估算器现在受益于改进的性能：
#
# - :func:`metrics.pairwise_distances_argmin`
# - :func:`metrics.pairwise_distances_argmin_min`
# - :class:`cluster.AffinityPropagation`
# - :class:`cluster.Birch`
# - :class:`cluster.MeanShift`
# - :class:`cluster.OPTICS`
# - :class:`cluster.SpectralClustering`
# - :func:`feature_selection.mutual_info_regression`
# - :class:`neighbors.KNeighborsClassifier`
# - :class:`neighbors.KNeighborsRegressor`
# - :class:`neighbors.RadiusNeighborsClassifier`
# - :class:`neighbors.RadiusNeighborsRegressor`
# - :class:`neighbors.LocalOutlierFactor`
# - :class:`neighbors.NearestNeighbors`
# - :class:`manifold.Isomap`
# - :class:`manifold.LocallyLinearEmbedding`
# - :class:`manifold.TSNE`
# - :func:`manifold.trustworthiness`
# - :class:`semi_supervised.LabelPropagation`
# - :class:`semi_supervised.LabelSpreading`
#
# 欲了解更多关于这项工作的技术细节，请阅读 `这系列博文 <https://blog.scikit-learn.org/technical/performances/>`_。
#
# %%
# :class:`~decomposition.MiniBatchNMF`: an online version of NMF
# --------------------------------------------------------------
# The new class :class:`~decomposition.MiniBatchNMF` implements a faster but
# less accurate version of non-negative matrix factorization
# (:class:`~decomposition.NMF`). :class:`~decomposition.MiniBatchNMF` divides the
# data into mini-batches and optimizes the NMF model in an online manner by
# cycling over the mini-batches, making it better suited for large datasets. In
# particular, it implements `partial_fit`, which can be used for online
# learning when the data is not readily available from the start, or when the
# data does not fit into memory.
import numpy as np
from sklearn.decomposition import MiniBatchNMF

rng = np.random.RandomState(0)
n_samples, n_features, n_components = 10, 10, 5
true_W = rng.uniform(size=(n_samples, n_components))
true_H = rng.uniform(size=(n_components, n_features))
X = true_W @ true_H

# Create an instance of MiniBatchNMF with specified number of components and random state
nmf = MiniBatchNMF(n_components=n_components, random_state=0)

# Perform partial fit of the model on data X for 10 iterations
for _ in range(10):
    nmf.partial_fit(X)

# Extract the transformed features W and the components H from the fitted model
W = nmf.transform(X)
H = nmf.components_

# Reconstruct the original matrix X using the transformed features W and components H
X_reconstructed = W @ H

# Calculate and print the relative reconstruction error of the matrix X
print(
    f"relative reconstruction error: ",
    f"{np.sum((X - X_reconstructed) ** 2) / np.sum(X**2):.5f}",
)

# %%
# :class:`~cluster.BisectingKMeans`: divide and cluster
# -----------------------------------------------------
# The new class :class:`~cluster.BisectingKMeans` is a variant of
# :class:`~cluster.KMeans`, using divisive hierarchical clustering. Instead of
# creating all centroids at once, centroids are picked progressively based on a
# previous clustering: a cluster is split into two new clusters repeatedly
# until the target number of clusters is reached, giving a hierarchical
# structure to the clustering.
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, BisectingKMeans
import matplotlib.pyplot as plt

# Generate synthetic data with two clusters
X, _ = make_blobs(n_samples=1000, centers=2, random_state=0)

# Fit KMeans and BisectingKMeans clustering models to the data
km = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X)
bisect_km = BisectingKMeans(n_clusters=5, random_state=0).fit(X)

# Plotting setup: create subplots for KMeans and BisectingKMeans visualizations
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Scatter plot of data points colored by KMeans cluster labels
ax[0].scatter(X[:, 0], X[:, 1], s=10, c=km.labels_)
# Plot cluster centers for KMeans in red
ax[0].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=20, c="r")
ax[0].set_title("KMeans")

# Scatter plot of data points colored by BisectingKMeans cluster labels
ax[1].scatter(X[:, 0], X[:, 1], s=10, c=bisect_km.labels_)
# Plot cluster centers for BisectingKMeans in red
ax[1].scatter(
    bisect_km.cluster_centers_[:, 0], bisect_km.cluster_centers_[:, 1], s=20, c="r"
)
ax[1].set_title("BisectingKMeans")

# Show the plot
_ = plt.show()
```