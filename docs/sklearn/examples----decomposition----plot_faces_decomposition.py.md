# `D:\src\scipysrc\scikit-learn\examples\decomposition\plot_faces_decomposition.py`

```
# %%
# Decomposition
# -------------
#
# Initialise different estimators for decomposition and fit each
# of them on all images and plot some results. Each estimator extracts
# 6 components as vectors :math:`h \in \mathbb{R}^{4096}`.
# We just displayed these vectors in human-friendly visualisation as 64x64 pixel images.
#
# Read more in the :ref:`User Guide <decompositions>`.

# %%
# Eigenfaces - PCA using randomized SVD
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Linear dimensionality reduction using Singular Value Decomposition (SVD) of the data
# to project it to a lower dimensional space.
#
#
# .. note::
#    This section applies Principal Component Analysis (PCA) on the Olivetti faces dataset.
#    It uses randomized SVD for efficient computation of the principal components.
#     The Eigenfaces estimator, via the :py:mod:`sklearn.decomposition.PCA`,
#     also provides a scalar `noise_variance_` (the mean of pixelwise variance)
#     that cannot be displayed as an image.

# %%
# 创建 PCA 估计器对象，使用随机化 SVD 求解器和白化处理
pca_estimator = decomposition.PCA(
    n_components=n_components, svd_solver="randomized", whiten=True
)
# 将中心化后的人脸数据拟合到 PCA 估计器上
pca_estimator.fit(faces_centered)
# 绘制 PCA 成分作为图库显示
plot_gallery(
    "Eigenfaces - PCA using randomized SVD", pca_estimator.components_[:n_components]
)

# %%
# Non-negative components - NMF
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 估计非负原始数据作为两个非负矩阵的乘积。

# %%
# 创建 NMF 估计器对象，设置成分数量和容忍度
nmf_estimator = decomposition.NMF(n_components=n_components, tol=5e-3)
# 将原始人脸数据拟合到 NMF 估计器上
nmf_estimator.fit(faces)  # original non- negative dataset
# 绘制 NMF 成分作为图库显示
plot_gallery("Non-negative components - NMF", nmf_estimator.components_[:n_components])

# %%
# Independent components - FastICA
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 独立分量分析将多变量向量分离成最大程度独立的加法子分量。

# %%
# 创建 FastICA 估计器对象，设置成分数量、最大迭代次数和白化方法
ica_estimator = decomposition.FastICA(
    n_components=n_components, max_iter=400, whiten="arbitrary-variance", tol=15e-5
)
# 将中心化后的人脸数据拟合到 FastICA 估计器上
ica_estimator.fit(faces_centered)
# 绘制 FastICA 成分作为图库显示
plot_gallery(
    "Independent components - FastICA", ica_estimator.components_[:n_components]
)

# %%
# Sparse components - MiniBatchSparsePCA
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 迷你批量稀疏PCA (:class:`~sklearn.decomposition.MiniBatchSparsePCA`)
# 提取出最佳重构数据的稀疏成分集合。这种变体比类似的
# :class:`~sklearn.decomposition.SparsePCA` 更快但精度较低。

# %%
# 创建 MiniBatchSparsePCA 估计器对象，设置成分数量、alpha、最大迭代次数、批量大小和随机种子
batch_pca_estimator = decomposition.MiniBatchSparsePCA(
    n_components=n_components, alpha=0.1, max_iter=100, batch_size=3, random_state=rng
)
# 将中心化后的人脸数据拟合到 MiniBatchSparsePCA 估计器上
batch_pca_estimator.fit(faces_centered)
# 绘制 MiniBatchSparsePCA 成分作为图库显示
plot_gallery(
    "Sparse components - MiniBatchSparsePCA",
    batch_pca_estimator.components_[:n_components],
)

# %%
# Dictionary learning
# ^^^^^^^^^^^^^^^^^^^
#
# 默认情况下，:class:`~sklearn.decomposition.MiniBatchDictionaryLearning`
# 将数据划分为小批量，并通过在指定迭代次数内循环这些小批量进行在线优化。

# %%
# 创建 MiniBatchDictionaryLearning 估计器对象，设置成分数量、alpha、最大迭代次数、批量大小和随机种子
batch_dict_estimator = decomposition.MiniBatchDictionaryLearning(
    n_components=n_components, alpha=0.1, max_iter=50, batch_size=3, random_state=rng
)
# 将中心化后的人脸数据拟合到 MiniBatchDictionaryLearning 估计器上
batch_dict_estimator.fit(faces_centered)
# 绘制 MiniBatchDictionaryLearning 成分作为图库显示
plot_gallery("Dictionary learning", batch_dict_estimator.components_[:n_components])

# %%
# Cluster centers - MiniBatchKMeans
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`sklearn.cluster.MiniBatchKMeans` 在线学习，具有
# :meth:`~sklearn.cluster.MiniBatchKMeans.partial_fit` 方法。这使得
# 将一些耗时的算法与 :class:`~sklearn.cluster.MiniBatchKMeans` 结合可能会更加有效。

# %%
# 创建 MiniBatchKMeans 聚类器对象，设置聚类数量、容忍度和批量大小
kmeans_estimator = cluster.MiniBatchKMeans(
    n_clusters=n_components,
    tol=1e-3,
    batch_size=20,
    max_iter=50,
    random_state=rng,



    # 设置最大迭代次数为50次
    max_iter=50,
    # 设置随机数生成器的状态为rng，用于控制随机性
    random_state=rng,
# %%
# Factor Analysis components - FA
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`~sklearn.decomposition.FactorAnalysis` is similar to
# :class:`~sklearn.decomposition.PCA` but has the advantage of modelling the
# variance in every direction of the input space independently (heteroscedastic
# noise). Read more in the :ref:`User Guide <FA>`.

# %%
# 创建一个 Factor Analysis (FA) 的估计器对象，指定要提取的成分数量和最大迭代次数
fa_estimator = decomposition.FactorAnalysis(n_components=n_components, max_iter=20)
# 使用面部数据训练 Factor Analysis 模型
fa_estimator.fit(faces_centered)
# 绘制 Factor Analysis (FA) 提取的成分图像库
plot_gallery("Factor Analysis (FA)", fa_estimator.components_[:n_components])

# --- Pixelwise variance
# 创建一个新的图形窗口，设置大小和背景色，并自动调整布局
plt.figure(figsize=(3.2, 3.6), facecolor="white", tight_layout=True)
# 获取每个像素点的噪声方差
vec = fa_estimator.noise_variance_
# 计算像素方差的最大值，用于图像显示的颜色映射
vmax = max(vec.max(), -vec.min())
# 绘制像素方差的图像，使用灰度颜色映射，插值方式为最近邻，设置最小和最大值
plt.imshow(
    vec.reshape(image_shape),
    cmap=plt.cm.gray,
    interpolation="nearest",
    vmin=-vmax,
    vmax=vmax,
)
# 关闭坐标轴显示
plt.axis("off")
# 设置图像标题
plt.title("Pixelwise variance from \n Factor Analysis (FA)", size=16, wrap=True)
# 添加水平方向的颜色条
plt.colorbar(orientation="horizontal", shrink=0.8, pad=0.03)
# 显示图像
plt.show()

# %%
# Decomposition: Dictionary learning
# ----------------------------------
#
# In the further section, let's consider :ref:`DictionaryLearning` more precisely.
# Dictionary learning is a problem that amounts to finding a sparse representation
# of the input data as a combination of simple elements. These simple elements form
# a dictionary. It is possible to constrain the dictionary and/or coding coefficients
# to be positive to match constraints that may be present in the data.
#
# :class:`~sklearn.decomposition.MiniBatchDictionaryLearning` implements a
# faster, but less accurate version of the dictionary learning algorithm that
# is better suited for large datasets. Read more in the :ref:`User Guide
# <MiniBatchDictionaryLearning>`.

# %%
# 绘制数据集中的人脸样本，使用另一种颜色映射
plot_gallery("Faces from dataset", faces_centered[:n_components], cmap=plt.cm.RdBu)

# %%
# Similar to the previous examples, we change parameters and train
# :class:`~sklearn.decomposition.MiniBatchDictionaryLearning` estimator on all
# images. Generally, the dictionary learning and sparse encoding decompose
# input data into the dictionary and the coding coefficients matrices. :math:`X
# \approx UV`, where :math:`X = [x_1, . . . , x_n]`, :math:`X \in
# \mathbb{R}^{m×n}`, dictionary :math:`U \in \mathbb{R}^{m×k}`, coding
# coefficients :math:`V \in \mathbb{R}^{k×n}`.
#
# Also below are the results when the dictionary and coding
# coefficients are positively constrained.

# %%
# Dictionary learning - positive dictionary
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the following section we enforce positivity when finding the dictionary.

# %%
dict_pos_dict_estimator = decomposition.MiniBatchDictionaryLearning(
    n_components=n_components,  # 设置字典学习的组件数量
    alpha=0.1,  # 控制稀疏性的参数
    max_iter=50,  # 最大迭代次数
    batch_size=3,  # 每次迭代的样本批次大小
    random_state=rng,  # 随机数生成器的种子，用于重现随机结果
    positive_dict=True,  # 将字典限制为非负值
)
dict_pos_dict_estimator.fit(faces_centered)  # 对中心化的面部数据进行字典学习
plot_gallery(
    "Dictionary learning - positive dictionary",  # 绘制图表标题
    dict_pos_dict_estimator.components_[:n_components],  # 使用学到的字典成分绘制图像
    cmap=plt.cm.RdBu,  # 使用的颜色映射
)

# %%
# Dictionary learning - positive code
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Below we constrain the coding coefficients as a positive matrix.
#
dict_pos_code_estimator = decomposition.MiniBatchDictionaryLearning(
    n_components=n_components,  # 设置字典学习的组件数量
    alpha=0.1,  # 控制稀疏性的参数
    max_iter=50,  # 最大迭代次数
    batch_size=3,  # 每次迭代的样本批次大小
    fit_algorithm="cd",  # 选择使用的拟合算法
    random_state=rng,  # 随机数生成器的种子，用于重现随机结果
    positive_code=True,  # 将编码系数限制为非负值
)
dict_pos_code_estimator.fit(faces_centered)  # 对中心化的面部数据进行字典学习
plot_gallery(
    "Dictionary learning - positive code",  # 绘制图表标题
    dict_pos_code_estimator.components_[:n_components],  # 使用学到的字典成分绘制图像
    cmap=plt.cm.RdBu,  # 使用的颜色映射
)

# %%
# Dictionary learning - positive dictionary & code
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Also below are the results if the dictionary values and coding
# coefficients are positively constrained.
#
dict_pos_estimator = decomposition.MiniBatchDictionaryLearning(
    n_components=n_components,  # 设置字典学习的组件数量
    alpha=0.1,  # 控制稀疏性的参数
    max_iter=50,  # 最大迭代次数
    batch_size=3,  # 每次迭代的样本批次大小
    fit_algorithm="cd",  # 选择使用的拟合算法
    random_state=rng,  # 随机数生成器的种子，用于重现随机结果
    positive_dict=True,  # 将字典限制为非负值
    positive_code=True,  # 将编码系数限制为非负值
)
dict_pos_estimator.fit(faces_centered)  # 对中心化的面部数据进行字典学习
plot_gallery(
    "Dictionary learning - positive dictionary & code",  # 绘制图表标题
    dict_pos_estimator.components_[:n_components],  # 使用学到的字典成分绘制图像
    cmap=plt.cm.RdBu,  # 使用的颜色映射
)
```