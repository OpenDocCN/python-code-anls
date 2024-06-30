# `D:\src\scipysrc\scikit-learn\examples\manifold\plot_swissroll.py`

```
# %%
# Swiss Roll
# ---------------------------------------------------
#
# We start by generating the Swiss Roll dataset.

import matplotlib.pyplot as plt

from sklearn import datasets, manifold

# 使用 sklearn 中的 make_swiss_roll 函数生成包含 1500 个样本的瑞士卷数据集，设置随机种子为 0
sr_points, sr_color = datasets.make_swiss_roll(n_samples=1500, random_state=0)

# %%
# Now, let's take a look at our data:

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
# 绘制三维散点图，展示瑞士卷数据集的三个特征维度，颜色由 sr_color 指定，点大小为 50，透明度为 0.8
ax.scatter(
    sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, s=50, alpha=0.8
)
ax.set_title("Swiss Roll in Ambient Space")
ax.view_init(azim=-66, elev=12)
_ = ax.text2D(0.8, 0.05, s="n_samples=1500", transform=ax.transAxes)

# %%
# Computing the LLE and t-SNE embeddings, we find that LLE seems to unroll the
# Swiss Roll pretty effectively. t-SNE on the other hand, is able
# to preserve the general structure of the data, but, poorly represents the
# continuous nature of our original data. Instead, it seems to unnecessarily
# clump sections of points together.

# 计算使用 LLE 方法得到的二维嵌入数据 sr_lle 和估计误差 sr_err
sr_lle, sr_err = manifold.locally_linear_embedding(
    sr_points, n_neighbors=12, n_components=2
)

# 使用 t-SNE 方法得到的二维嵌入数据 sr_tsne，设置困惑度为 40，随机种子为 0
sr_tsne = manifold.TSNE(n_components=2, perplexity=40, random_state=0).fit_transform(
    sr_points
)

# 创建包含两个子图的图像窗口，展示 LLE 和 t-SNE 嵌入的结果
fig, axs = plt.subplots(figsize=(8, 8), nrows=2)
axs[0].scatter(sr_lle[:, 0], sr_lle[:, 1], c=sr_color)
axs[0].set_title("LLE Embedding of Swiss Roll")
axs[1].scatter(sr_tsne[:, 0], sr_tsne[:, 1], c=sr_color)
_ = axs[1].set_title("t-SNE Embedding of Swiss Roll")

# %%
# .. note::
#
#     LLE seems to be stretching the points from the center (purple)
#     of the swiss roll. However, we observe that this is simply a byproduct
#     of how the data was generated. There is a higher density of points near the
#     center of the roll, which ultimately affects how LLE reconstructs the
#     data in a lower dimension.

# %%
# Swiss-Hole
# ---------------------------------------------------
#
# Now let's take a look at how both algorithms deal with us adding a hole to
# the data. First, we generate the Swiss-Hole dataset and plot it:

# 生成包含 1500 个样本的带有洞的瑞士卷数据集，设置随机种子为 0
sh_points, sh_color = datasets.make_swiss_roll(
    n_samples=1500, hole=True, random_state=0
)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
# 绘制三维散点图，展示带有洞的瑞士卷数据集的三个特征维度，颜色由 sh_color 指定，点大小为 50，透明度为 0.8
ax.scatter(
    sh_points[:, 0], sh_points[:, 1], sh_points[:, 2], c=sh_color, s=50, alpha=0.8
)
ax.set_title("Swiss-Hole in Ambient Space")
ax.view_init(azim=-66, elev=12)
_ = ax.text2D(0.8, 0.05, s="n_samples=1500", transform=ax.transAxes)
# Swiss Roll. LLE very capably unrolls the data and even preserves
# the hole. t-SNE, again seems to clump sections of points together, but, we
# note that it preserves the general topology of the original data.

# 使用局部线性嵌入（Locally Linear Embedding, LLE）方法对数据进行降维，保留了数据的展开性并且保留了空洞。
sh_lle, sh_err = manifold.locally_linear_embedding(
    sh_points, n_neighbors=12, n_components=2
)

# 使用 t-SNE 方法对数据进行降维，perplexity 设置为 40，使用随机初始化，随机种子为 0。
sh_tsne = manifold.TSNE(
    n_components=2, perplexity=40, init="random", random_state=0
).fit_transform(sh_points)

# 创建一个大小为 8x8 的图表，包含两行，返回图表和子图对象。
fig, axs = plt.subplots(figsize=(8, 8), nrows=2)

# 在第一个子图上绘制 LLE 方法降维后的数据散点图，使用 sh_color 进行颜色编码。
axs[0].scatter(sh_lle[:, 0], sh_lle[:, 1], c=sh_color)
axs[0].set_title("LLE Embedding of Swiss-Hole")  # 设置第一个子图的标题

# 在第二个子图上绘制 t-SNE 方法降维后的数据散点图，使用 sh_color 进行颜色编码。
axs[1].scatter(sh_tsne[:, 0], sh_tsne[:, 1], c=sh_color)
_ = axs[1].set_title("t-SNE Embedding of Swiss-Hole")  # 设置第二个子图的标题

# %%
#
# Concluding remarks
# ------------------
#
# We note that t-SNE benefits from testing more combinations of parameters.
# Better results could probably have been obtained by better tuning these
# parameters.
#
# We observe that, as seen in the "Manifold learning on
# handwritten digits" example, t-SNE generally performs better than LLE
# on real world data.

# 结论
# ------------------
#
# 注意到 t-SNE 通过测试更多参数组合获益良多。
# 通过更好地调整这些参数，可能可以获得更好的结果。
#
# 我们观察到，正如在“手写数字的流形学习”示例中所见，t-SNE 在真实世界数据上通常比 LLE 表现更好。
```