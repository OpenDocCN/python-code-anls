# `D:\src\scipysrc\scikit-learn\examples\mixture\plot_gmm_sin.py`

```
# 导入必要的库
import itertools  # 导入 itertools 库，用于迭代工具

import matplotlib as mpl  # 导入 matplotlib 库，并命名为 mpl
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并命名为 plt
import numpy as np  # 导入 numpy 库，并命名为 np
from scipy import linalg  # 从 scipy 库导入 linalg 模块

# 这里定义了一个循环，用于为每个高斯混合模型结果指定不同的颜色
color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])


def plot_results(X, Y, means, covariances, index, title):
    # 创建一个子图，布局为 5 行 1 列，当前绘制的位置是第 1 + index 行
    splot = plt.subplot(5, 1, 1 + index)
    # 遍历均值、协方差和颜色的迭代器，同时获取索引 i
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        # 计算协方差矩阵的特征值和特征向量
        v, w = linalg.eigh(covar)
        # 对特征值进行转换以获取椭圆的长轴和短轴长度
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        # 获取第一个特征向量的单位向量
        u = w[0] / linalg.norm(w[0])
        
        # 如果数据集 Y 中没有索引 i 的任何元素，则跳过绘制
        if not np.any(Y == i):
            continue
        
        # 绘制散点图，显示数据点的分布，颜色由 color 决定
        plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)

        # 绘制椭圆以展示高斯分布的成分
        angle = np.arctan(u[1] / u[0])  # 计算椭圆的旋转角度
        angle = 180.0 * angle / np.pi  # 将角度转换为度数
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)  # 设置椭圆的剪裁框
        ell.set_alpha(0.5)  # 设置椭圆的透明度
        splot.add_artist(ell)  # 将椭圆添加到图中

    # 设置图的 x 轴和 y 轴的范围
    plt.xlim(-6.0, 4.0 * np.pi - 6.0)
    plt.ylim(-5.0, 5.0)
    plt.title(title)  # 设置图的标题
    plt.xticks(())  # 设置 x 轴刻度为空
    plt.yticks(())  # 设置 y 轴刻度为空
# 定义一个函数用于绘制样本数据的散点图
def plot_samples(X, Y, n_components, index, title):
    # 在绘图区域中创建子图，4 + index 表示子图的位置
    plt.subplot(5, 1, 4 + index)
    
    # 对于每个成分及其对应的颜色，绘制散点图
    for i, color in zip(range(n_components), color_iter):
        # 由于 DP 不一定会使用每个成分，我们不应绘制多余的成分
        if not np.any(Y == i):
            continue
        # 绘制散点图
        plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)

    # 设置 x 和 y 轴的范围
    plt.xlim(-6.0, 4.0 * np.pi - 6.0)
    plt.ylim(-5.0, 5.0)
    # 设置子图标题
    plt.title(title)
    # 清空 x 和 y 轴的刻度
    plt.xticks(())
    plt.yticks(())


# 设置参数：样本数
n_samples = 100

# 生成沿正弦曲线的随机样本数据
np.random.seed(0)
X = np.zeros((n_samples, 2))
step = 4.0 * np.pi / n_samples

# 循环生成样本数据
for i in range(X.shape[0]):
    x = i * step - 6.0
    # 添加随机噪声并生成样本数据
    X[i, 0] = x + np.random.normal(0, 0.1)
    X[i, 1] = 3.0 * (np.sin(x) + np.random.normal(0, 0.2))

# 创建一个 10x10 英寸的新图形窗口，并调整子图布局参数
plt.figure(figsize=(10, 10))
plt.subplots_adjust(
    bottom=0.04, top=0.95, hspace=0.2, wspace=0.05, left=0.03, right=0.97
)

# 使用 EM 算法拟合一个高斯混合模型，使用十个成分
gmm = mixture.GaussianMixture(
    n_components=10, covariance_type="full", max_iter=100
).fit(X)
plot_results(
    X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, "Expectation-maximization"
)

# 使用贝叶斯高斯混合模型（DP-GMM）拟合数据，使用一个狄利克雷过程先验
dpgmm = mixture.BayesianGaussianMixture(
    n_components=10,
    covariance_type="full",
    weight_concentration_prior=1e-2,
    weight_concentration_prior_type="dirichlet_process",
    mean_precision_prior=1e-2,
    covariance_prior=1e0 * np.eye(2),
    init_params="random",
    max_iter=100,
    random_state=2,
).fit(X)
plot_results(
    X,
    dpgmm.predict(X),
    dpgmm.means_,
    dpgmm.covariances_,
    1,
    "Bayesian Gaussian mixture models with a Dirichlet process prior "
    r"for $\gamma_0=0.01$.",
)

# 从拟合的 DP-GMM 模型中采样 2000 个样本，并绘制样本数据的散点图
X_s, y_s = dpgmm.sample(n_samples=2000)
plot_samples(
    X_s,
    y_s,
    dpgmm.n_components,
    0,
    "Gaussian mixture with a Dirichlet process prior "
    r"for $\gamma_0=0.01$ sampled with $2000$ samples.",
)

# 使用另一组参数再次拟合 DP-GMM 模型
dpgmm = mixture.BayesianGaussianMixture(
    n_components=10,
    covariance_type="full",
    weight_concentration_prior=1e2,
    weight_concentration_prior_type="dirichlet_process",
    mean_precision_prior=1e-2,
    covariance_prior=1e0 * np.eye(2),
    init_params="kmeans",
    max_iter=100,
    random_state=2,
).fit(X)
plot_results(
    X,
    dpgmm.predict(X),
    dpgmm.means_,
    dpgmm.covariances_,
    2,
    "Bayesian Gaussian mixture models with a Dirichlet process prior "
    r"for $\gamma_0=100$",
)

# 再次从拟合的 DP-GMM 模型中采样 2000 个样本，并绘制样本数据的散点图
X_s, y_s = dpgmm.sample(n_samples=2000)
plot_samples(
    X_s,
    y_s,
    dpgmm.n_components,
    1,
    "Gaussian mixture with a Dirichlet process prior "
    r"for $\gamma_0=100$ sampled with $2000$ samples.",
)

# 显示绘制的图形
plt.show()
```