# `D:\src\scipysrc\scikit-learn\examples\decomposition\plot_ica_vs_pca.py`

```
# %%
# Generate sample data
# --------------------
# 导入所需的库
import numpy as np

# 导入 PCA 和 FastICA 用于降维和独立成分分析
from sklearn.decomposition import PCA, FastICA

# 创建随机数生成器
rng = np.random.RandomState(42)

# 生成非高斯分布的独立源数据
S = rng.standard_t(1.5, size=(20000, 2))
S[:, 0] *= 2.0  # 增加第一个维度的方差

# 混合数据
A = np.array([[1, 1], [0, 2]])  # 混合矩阵

X = np.dot(S, A.T)  # 生成观测数据

# %%
# Plot results
# ------------
# 导入绘图库
import matplotlib.pyplot as plt


# 定义绘制样本数据和向量的函数
def plot_samples(S, axis_list=None):
    # 绘制样本点
    plt.scatter(
        S[:, 0], S[:, 1], s=2, marker="o", zorder=10, color="steelblue", alpha=0.5
    )
    # 绘制向量
    if axis_list is not None:
        for axis, color, label in axis_list:
            axis /= axis.std()  # 标准化向量
            x_axis, y_axis = axis
            plt.quiver(
                (0, 0),
                (0, 0),
                x_axis,
                y_axis,
                zorder=11,
                width=0.01,
                scale=6,
                color=color,
                label=label,
            )

    # 绘制坐标轴
    plt.hlines(0, -3, 3)
    plt.vlines(0, -3, 3)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel("x")
    plt.ylabel("y")


# 创建新的图形
plt.figure()

# 绘制第一个子图：真实的独立源
plt.subplot(2, 2, 1)
plot_samples(S / S.std())
plt.title("True Independent Sources")

# 定义 PCA 和 ICA 的混合矩阵并绘制第二个子图：观测数据
axis_list = [(pca.components_.T, "orange", "PCA"), (ica.mixing_, "red", "ICA")]
plt.subplot(2, 2, 2)
plot_samples(X / np.std(X), axis_list=axis_list)
legend = plt.legend(loc="lower right")
legend.set_zorder(100)
plt.title("Observations")

# 创建第三个子图
plt.subplot(2, 2, 3)
# 绘制 PCA 恢复的信号样本，对每一列除以其标准差以进行标准化
plot_samples(S_pca_ / np.std(S_pca_, axis=0))

# 设置第四个子图，用于展示 ICA 恢复的信号样本
plt.subplot(2, 2, 4)

# 绘制 ICA 恢复的信号样本，对所有信号除以它们的标准差进行标准化
plot_samples(S_ica_ / np.std(S_ica_))

# 调整子图布局参数，以确保图形适当地显示在画布上
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)

# 调整整体布局，以确保所有子图之间有合适的间距
plt.tight_layout()

# 显示绘制的图形
plt.show()
```