# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_polynomial_kernel_approximation.py`

```
"""
========================================================================
Benchmark for explicit feature map approximation of polynomial kernels
========================================================================

An example illustrating the approximation of the feature map
of an Homogeneous Polynomial kernel.

.. currentmodule:: sklearn.kernel_approximation

It shows how to use :class:`PolynomialCountSketch` and :class:`Nystroem` to
approximate the feature map of a polynomial kernel for
classification with an SVM on the digits dataset. Results using a linear
SVM in the original space, a linear SVM using the approximate mappings
and a kernelized SVM are compared.

The first plot shows the classification accuracy of Nystroem [2] and
PolynomialCountSketch [1] as the output dimension (n_components) grows.
It also shows the accuracy of a linear SVM and a polynomial kernel SVM
on the same data.

The second plot explores the scalability of PolynomialCountSketch
and Nystroem. For a sufficiently large output dimension,
PolynomialCountSketch should be faster as it is O(n(d+klog k))
while Nystroem is O(n(dk+k^2)). In addition, Nystroem requires
a time-consuming training phase, while training is almost immediate
for PolynomialCountSketch, whose training phase boils down to
initializing some random variables (because is data-independent).

[1] Pham, N., & Pagh, R. (2013, August). Fast and scalable polynomial
kernels via explicit feature maps. In Proceedings of the 19th ACM SIGKDD
international conference on Knowledge discovery and data mining (pp. 239-247)
(https://chbrown.github.io/kdd-2013-usb/kdd/p239.pdf)

[2] Charikar, M., Chen, K., & Farach-Colton, M. (2002, July). Finding frequent
items in data streams. In International Colloquium on Automata, Languages, and
Programming (pp. 693-703). Springer, Berlin, Heidelberg.
(https://people.cs.rutgers.edu/~farach/pubs/FrequentStream.pdf)

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Load data manipulation functions
# Will use this for timing results
from time import time

# Some common libraries
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.kernel_approximation import Nystroem, PolynomialCountSketch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import SVM classifiers and feature map approximation algorithms
from sklearn.svm import SVC, LinearSVC

# Split data in train and test sets
# 加载手写数字数据集并将其分割成训练集和测试集
X, y = load_digits()["data"], load_digits()["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Set the range of n_components for our experiments
# 设置实验中输出维度的范围
out_dims = range(20, 400, 20)

# Evaluate Linear SVM
# 训练并评估线性 SVM 模型
lsvm = LinearSVC().fit(X_train, y_train)
lsvm_score = 100 * lsvm.score(X_test, y_test)

# Evaluate kernelized SVM
# 训练并评估核化 SVM 模型（多项式核）
ksvm = SVC(kernel="poly", degree=2, gamma=1.0).fit(X_train, y_train)
ksvm_score = 100 * ksvm.score(X_test, y_test)
# Evaluate PolynomialCountSketch + LinearSVM
# 初始化一个空列表，用于存储评分结果
ps_svm_scores = []
# 设置运行次数
n_runs = 5

# To compensate for the stochasticity of the method, we make n_runs runs
# 针对输出维度列表中的每个维度 k 进行评分计算
for k in out_dims:
    # 初始化平均分数为 0
    score_avg = 0
    # 进行 n_runs 次评分计算，求平均值
    for _ in range(n_runs):
        # 创建 Pipeline 对象，包括 PolynomialCountSketch 和 LinearSVC 两个步骤
        ps_svm = Pipeline(
            [
                ("PS", PolynomialCountSketch(degree=2, n_components=k)),
                ("SVM", LinearSVC()),
            ]
        )
        # 训练 Pipeline 并计算测试集的准确率，累加得分
        score_avg += ps_svm.fit(X_train, y_train).score(X_test, y_test)
    # 将平均得分乘以 100并除以 n_runs，得到百分比形式的平均准确率，存入列表
    ps_svm_scores.append(100 * score_avg / n_runs)

# Evaluate Nystroem + LinearSVM
# 初始化一个空列表，用于存储评分结果
ny_svm_scores = []
# 设置运行次数
n_runs = 5

# 针对输出维度列表中的每个维度 k 进行评分计算
for k in out_dims:
    # 初始化平均分数为 0
    score_avg = 0
    # 进行 n_runs 次评分计算，求平均值
    for _ in range(n_runs):
        # 创建 Pipeline 对象，包括 Nystroem 和 LinearSVC 两个步骤
        ny_svm = Pipeline(
            [
                (
                    "NY",
                    Nystroem(
                        kernel="poly", gamma=1.0, degree=2, coef0=0, n_components=k
                    ),
                ),
                ("SVM", LinearSVC()),
            ]
        )
        # 训练 Pipeline 并计算测试集的准确率，累加得分
        score_avg += ny_svm.fit(X_train, y_train).score(X_test, y_test)
    # 将平均得分乘以 100并除以 n_runs，得到百分比形式的平均准确率，存入列表
    ny_svm_scores.append(100 * score_avg / n_runs)

# Show results
# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(6, 4))
# 设置图表标题
ax.set_title("Accuracy results")
# 绘制 PolynomialCountSketch + linear SVM 的准确率曲线，设置颜色为橙色
ax.plot(out_dims, ps_svm_scores, label="PolynomialCountSketch + linear SVM", c="orange")
# 绘制 Nystroem + linear SVM 的准确率曲线，设置颜色为蓝色
ax.plot(out_dims, ny_svm_scores, label="Nystroem + linear SVM", c="blue")
# 绘制线性 SVM 的准确率曲线，设置颜色为黑色，线型为虚线
ax.plot(
    [out_dims[0], out_dims[-1]],
    [lsvm_score, lsvm_score],
    label="Linear SVM",
    c="black",
    dashes=[2, 2],
)
# 绘制多项式核 SVM 的准确率曲线，设置颜色为红色，线型为虚线
ax.plot(
    [out_dims[0], out_dims[-1]],
    [ksvm_score, ksvm_score],
    label="Poly-kernel SVM",
    c="red",
    dashes=[2, 2],
)
# 添加图例
ax.legend()
# 设置 x 轴标签
ax.set_xlabel("N_components for PolynomialCountSketch and Nystroem")
# 设置 y 轴标签
ax.set_ylabel("Accuracy (%)")
# 设置 x 轴的范围
ax.set_xlim([out_dims[0], out_dims[-1]])
# 调整布局，使图形紧凑显示
fig.tight_layout()

# Now lets evaluate the scalability of PolynomialCountSketch vs Nystroem
# First we generate some fake data with a lot of samples

# 生成包含大量样本的虚拟数据，维度为 (10000, 100)
fakeData = np.random.randn(10000, 100)
# 生成与虚拟数据对应的随机标签，标签范围在 [0, 10) 之间
fakeDataY = np.random.randint(0, high=10, size=(10000))

# 设置输出维度范围
out_dims = range(500, 6000, 500)

# Evaluate scalability of PolynomialCountSketch as n_components grows
# 初始化空列表，用于存储不同 n_components 下的运行时间
ps_svm_times = []
# 针对输出维度列表中的每个维度 k 进行运行时间评估
for k in out_dims:
    # 创建 PolynomialCountSketch 对象，设置 degree 和 n_components
    ps = PolynomialCountSketch(degree=2, n_components=k)

    # 记录开始时间
    start = time()
    # 对虚拟数据进行拟合和转换，记录运行时间
    ps.fit_transform(fakeData, None)
    # 计算运行时间并添加到列表
    ps_svm_times.append(time() - start)

# Evaluate scalability of Nystroem as n_components grows
# This can take a while due to the inefficient training phase
# 初始化空列表，用于存储不同 n_components 下的运行时间
ny_svm_times = []
# 针对输出维度列表中的每个维度 k 进行运行时间评估
for k in out_dims:
    # 创建 Nystroem 对象，设置 kernel、gamma、degree、coef0 和 n_components
    ny = Nystroem(kernel="poly", gamma=1.0, degree=2, coef0=0, n_components=k)

    # 记录开始时间
    start = time()
    # 对虚拟数据进行拟合和转换，记录运行时间
    ny.fit_transform(fakeData, None)
    # 计算运行时间并添加到列表
    ny_svm_times.append(time() - start)

# Show results
# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(6, 4))
# 设置图表标题
ax.set_title("Scalability results")
# 绘制 PolynomialCountSketch 的运行时间曲线，设置颜色为橙色
ax.plot(out_dims, ps_svm_times, label="PolynomialCountSketch", c="orange")
# 绘制 Nystroem 的运行时间曲线，设置颜色为蓝色
ax.plot(out_dims, ny_svm_times, label="Nystroem", c="blue")
# 添加图例
ax.legend()
# 设置 x 轴标签
ax.set_xlabel("N_components for PolynomialCountSketch and Nystroem")
# 设置 y 轴标签，包含换行符
ax.set_ylabel("fit_transform time \n(s/10.000 samples)")
# 设置图形的 X 轴限制，从 out_dims 列表的第一个元素到最后一个元素
ax.set_xlim([out_dims[0], out_dims[-1]])

# 调整图形布局，使其紧凑显示，以确保子图或标签不重叠
fig.tight_layout()

# 显示绘制好的图形
plt.show()
```