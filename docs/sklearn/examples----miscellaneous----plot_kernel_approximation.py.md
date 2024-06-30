# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_kernel_approximation.py`

```
# %%
# Python package and dataset imports, load dataset
# ---------------------------------------------------

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Standard scientific Python imports
from time import time  # 导入时间计算函数time

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot用于绘图
import numpy as np  # 导入numpy库，用于数值计算

# Import datasets, classifiers and performance metrics
from sklearn import datasets, pipeline, svm  # 导入数据集、分类器和性能评估模块
from sklearn.decomposition import PCA  # 导入PCA降维模块
from sklearn.kernel_approximation import Nystroem, RBFSampler  # 导入核近似模块

# The digits dataset
digits = datasets.load_digits(n_class=9)  # 加载手写数字数据集，仅包含类别为0-8的数字
    # 创建一个包含两个元组的列表，每个元组包含一个字符串和一个对象
    [
        # 第一个元组，字符串为 "feature_map"，对象是 feature_map_fourier
        ("feature_map", feature_map_fourier),
        # 第二个元组，字符串为 "svm"，对象是一个线性支持向量机分类器 LinearSVC，使用随机状态 42
        ("svm", svm.LinearSVC(random_state=42)),
    ]
`
# 创建一个 Pipeline 对象，包含特征映射和线性 SVM 分类器
nystroem_approx_svm = pipeline.Pipeline(
    [
        ("feature_map", feature_map_nystroem),  # 特征映射步骤
        ("svm", svm.LinearSVC(random_state=42)),  # 线性 SVM 分类器
    ]
)

# 训练并预测使用线性和核心 SVM：

# 计时开始，训练核心 SVM
kernel_svm_time = time()
kernel_svm.fit(data_train, targets_train)
kernel_svm_score = kernel_svm.score(data_test, targets_test)  # 测试核心 SVM 的准确率
kernel_svm_time = time() - kernel_svm_time  # 计算训练时间

# 计时开始，训练线性 SVM
linear_svm_time = time()
linear_svm.fit(data_train, targets_train)
linear_svm_score = linear_svm.score(data_test, targets_test)  # 测试线性 SVM 的准确率
linear_svm_time = time() - linear_svm_time  # 计算训练时间

# 定义样本大小的范围
sample_sizes = 30 * np.arange(1, 10)
fourier_scores = []
nystroem_scores = []
fourier_times = []
nystroem_times = []

# 针对每个样本大小进行循环
for D in sample_sizes:
    # 设置特征映射的组件数目
    fourier_approx_svm.set_params(feature_map__n_components=D)
    nystroem_approx_svm.set_params(feature_map__n_components=D)
    
    # 开始计时，训练 Nystroem SVM
    start = time()
    nystroem_approx_svm.fit(data_train, targets_train)
    nystroem_times.append(time() - start)  # 计算训练时间

    # 开始计时，训练 Fourier SVM
    start = time()
    fourier_approx_svm.fit(data_train, targets_train)
    fourier_times.append(time() - start)  # 计算训练时间

    # 计算 Fourier SVM 和 Nystroem SVM 的准确率
    fourier_score = fourier_approx_svm.score(data_test, targets_test)
    nystroem_score = nystroem_approx_svm.score(data_test, targets_test)
    nystroem_scores.append(nystroem_score)
    fourier_scores.append(fourier_score)

# 绘制结果图表
plt.figure(figsize=(16, 4))
accuracy = plt.subplot(121)
timescale = plt.subplot(122)

# 绘制准确率图表
accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")
timescale.plot(sample_sizes, nystroem_times, "--", label="Nystroem approx. kernel")

accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
timescale.plot(sample_sizes, fourier_times, "--", label="Fourier approx. kernel")

# 绘制线性 SVM 和 RBF SVM 的水平线
accuracy.plot(
    [sample_sizes[0], sample_sizes[-1]],
    [linear_svm_score, linear_svm_score],
    label="linear svm",
)
timescale.plot(
    [sample_sizes[0], sample_sizes[-1]],
    [linear_svm_time, linear_svm_time],
    "--",
    label="linear svm",
)

accuracy.plot(
    [sample_sizes[0], sample_sizes[-1]],
    [kernel_svm_score, kernel_svm_score],
    label="rbf svm",
)
timescale.plot(
    [sample_sizes[0], sample_sizes[-1]],
    [kernel_svm_time, kernel_svm_time],
    "--",
    label="rbf svm",
)

# 绘制数据集维度为 64 的垂直线
accuracy.plot([64, 64], [0.7, 1], label="n_features")

# 设置图表标题和轴标签
accuracy.set_title("Classification accuracy")
timescale.set_title("Training times")
accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
accuracy.set_xticks(())
accuracy.set_ylim(np.min(fourier_scores), 1)
timescale.set_xlabel("Sampling steps = transformed feature dimension")
accuracy.set_ylabel("Classification accuracy")
timescale.set_ylabel("Training time in seconds")
accuracy.legend(loc="best")
timescale.legend(loc="best")
plt.tight_layout()
plt.show()
# visualize the decision surface, projected down to the first
# two principal components of the dataset
# 使用 PCA 将数据集降维到前两个主成分并拟合
pca = PCA(n_components=8, random_state=42).fit(data_train)

# Transform the original data into the PCA space
# 将原始数据转换到 PCA 空间
X = pca.transform(data_train)

# Generate a grid along the directions of the first two principal components
# 生成网格，沿着第一个和第二个主成分的方向
multiples = np.arange(-2, 2, 0.1)
# Steps along the first principal component
# 第一个主成分的步长
first = multiples[:, np.newaxis] * pca.components_[0, :]
# Steps along the second principal component
# 第二个主成分的步长
second = multiples[:, np.newaxis] * pca.components_[1, :]
# Combine to form a grid
# 组合成网格
grid = first[np.newaxis, :, :] + second[:, np.newaxis, :]
# Flatten the grid into a 2D array
# 将网格展平成二维数组
flat_grid = grid.reshape(-1, data.shape[1])

# Titles for the subplots
# 子图的标题
titles = [
    "SVC with rbf kernel",
    "SVC (linear kernel)\n with Fourier rbf feature map\nn_components=100",
    "SVC (linear kernel)\n with Nystroem rbf feature map\nn_components=100",
]

# Create a figure for plotting
# 创建绘图用的图像
plt.figure(figsize=(18, 7.5))
# Update the font size of the plot
# 更新图表的字体大小
plt.rcParams.update({"font.size": 14})

# Predict and plot for each classifier
# 对每个分类器进行预测并绘图
for i, clf in enumerate((kernel_svm, nystroem_approx_svm, fourier_approx_svm)):
    # Plot the decision boundary by predicting on the flattened grid
    # 通过在展平的网格上进行预测来绘制决策边界
    plt.subplot(1, 3, i + 1)
    Z = clf.predict(flat_grid)

    # Put the result into a color plot using contourf
    # 使用 contourf 将结果绘制成彩色图
    Z = Z.reshape(grid.shape[:-1])
    levels = np.arange(10)
    lv_eps = 0.01  # Adjust a mapping from calculated contour levels to color.
    plt.contourf(
        multiples,
        multiples,
        Z,
        levels=levels - lv_eps,
        cmap=plt.cm.tab10,
        vmin=0,
        vmax=10,
        alpha=0.7,
    )
    # Turn off axis labels and ticks
    # 关闭坐标轴标签和刻度
    plt.axis("off")

    # Plot the training points on top of the decision boundary
    # 在决策边界上绘制训练数据点
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=targets_train,
        cmap=plt.cm.tab10,
        edgecolors=(0, 0, 0),
        vmin=0,
        vmax=10,
    )

    # Set title for the subplot
    # 设置子图的标题
    plt.title(titles[i])

# Adjust layout to prevent overlapping of subplots
# 调整布局以防止子图重叠
plt.tight_layout()

# Display the plot
# 显示绘制的图形
plt.show()
```