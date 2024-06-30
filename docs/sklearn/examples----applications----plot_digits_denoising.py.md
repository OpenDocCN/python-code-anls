# `D:\src\scipysrc\scikit-learn\examples\applications\plot_digits_denoising.py`

```
# %%
# In addition, we will use the mean squared error (MSE) to quantitatively
# evaluate the quality of the image reconstruction. Lower MSE indicates
# better reconstruction quality.

import numpy as np  # 导入 NumPy 库，用于数值计算

from sklearn.datasets import fetch_openml  # 导入 fetch_openml 函数，用于从 OpenML 获取数据集
from sklearn.model_selection import train_test_split  # 导入 train_test_split 函数，用于数据集划分
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler 函数，用于数据归一化处理

X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)  # 获取 USPS 数据集的特征和标签
X = MinMaxScaler().fit_transform(X)  # 对特征进行归一化处理

# %%
# The idea will be to learn a PCA basis (with and without a kernel) on
# noisy images and then use these models to reconstruct and denoise these
# images.
#
# Thus, we split our dataset into a training and testing set composed of 1,000
# samples for the training and 100 samples for testing. These images are
# noise-free and we will use them to evaluate the efficiency of the denoising
# approaches. In addition, we create a copy of the original dataset and add a
# Gaussian noise.
#
# The idea of this application, is to show that we can denoise corrupted images
# by learning a PCA basis on some uncorrupted images. We will use both a PCA
# and a kernel-based PCA to solve this problem.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, train_size=1_000, test_size=100
)  # 划分数据集为训练集和测试集，分别包含 1000 个和 100 个样本

rng = np.random.RandomState(0)  # 创建一个随机数生成器对象 rng，用于生成随机数
noise = rng.normal(scale=0.25, size=X_test.shape)  # 生成高斯噪声，大小和 X_test 相同
X_test_noisy = X_test + noise  # 在测试集上添加噪声，得到噪声污染的测试数据集

noise = rng.normal(scale=0.25, size=X_train.shape)  # 生成高斯噪声，大小和 X_train 相同
X_train_noisy = X_train + noise  # 在训练集上添加噪声，得到噪声污染的训练数据集

# %%
# In addition, we will create a helper function to qualitatively assess the
# image reconstruction by plotting the test images.
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图

def plot_digits(X, title):
    """Small helper function to plot 100 digits."""
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))  # 创建子图，10 行 10 列
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")  # 绘制灰度图像
        ax.axis("off")  # 关闭坐标轴
    fig.suptitle(title, fontsize=24)  # 设置图的标题和字体大小

# %%
# In addition, we will use the mean squared error (MSE) to quantitatively
# 评估图像重建效果。
#
# 首先比较无噪声和有噪声图像的差异。我们将在测试集中进行这方面的检查。
plot_digits(X_test, "Uncorrupted test images")  # 绘制未损坏的测试图像

# 绘制有噪声的测试图像，并显示均方误差（MSE）
plot_digits(
    X_test_noisy, f"Noisy test images\nMSE: {np.mean((X_test - X_test_noisy) ** 2):.2f}"
)

# %%
# 学习 PCA 基础
# ---------------------
#
# 现在我们可以使用线性 PCA 和使用径向基函数（RBF）核的核 PCA 来学习我们的 PCA 基础。
from sklearn.decomposition import PCA, KernelPCA

pca = PCA(n_components=32, random_state=42)  # 创建 PCA 对象，指定主成分数量和随机种子
kernel_pca = KernelPCA(
    n_components=400,  # 创建核 PCA 对象，指定主成分数量
    kernel="rbf",  # 使用径向基函数核
    gamma=1e-3,  # RBF 核的参数 gamma
    fit_inverse_transform=True,
    alpha=5e-3,
    random_state=42,
)

pca.fit(X_train_noisy)  # 在噪声训练集上拟合 PCA
_ = kernel_pca.fit(X_train_noisy)  # 在噪声训练集上拟合核 PCA

# %%
# 重建和去噪测试图像
# -----------------------------------
#
# 现在，我们可以转换和重建有噪声的测试集。由于我们使用的成分比原始特征的数量少，
# 我们将得到原始集合的近似。通过去除 PCA 中最少解释方差的成分，我们希望去除噪声。
# 在核 PCA 中也是类似的思路；但是，我们期望获得更好的重建，因为我们使用非线性核来学习
# PCA 基础，并使用核岭回归来学习映射函数。
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(
    kernel_pca.transform(X_test_noisy)
)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test_noisy))

# %%
plot_digits(X_test, "Uncorrupted test images")  # 绘制未损坏的测试图像

# 绘制 PCA 重建图像，并显示均方误差（MSE）
plot_digits(
    X_reconstructed_pca,
    f"PCA reconstruction\nMSE: {np.mean((X_test - X_reconstructed_pca) ** 2):.2f}",
)

# 绘制核 PCA 重建图像，并显示均方误差（MSE）
plot_digits(
    X_reconstructed_kernel_pca,
    (
        "Kernel PCA reconstruction\n"
        f"MSE: {np.mean((X_test - X_reconstructed_kernel_pca) ** 2):.2f}"
    ),
)

# %%
# PCA 的 MSE 比核 PCA 低。然而，定性分析可能不支持 PCA 而不是核 PCA。我们观察到核 PCA
# 能够去除背景噪声并提供更平滑的图像。
#
# 然而，应注意使用核 PCA 进行去噪的结果将取决于参数 `n_components`、`gamma` 和 `alpha`。
```