# `D:\src\scipysrc\scikit-learn\examples\gaussian_process\plot_gpr_on_structured_data.py`

```
"""
==========================================================================
Gaussian processes on discrete data structures
==========================================================================

This example illustrates the use of Gaussian processes for regression and
classification tasks on data that are not in fixed-length feature vector form.
This is achieved through the use of kernel functions that operate directly
on discrete structures such as variable-length sequences, trees, and graphs.

Specifically, here the input variables are some gene sequences stored as
variable-length strings consisting of letters 'A', 'T', 'C', and 'G',
while the output variables are floating point numbers and True/False labels
in the regression and classification tasks, respectively.

A kernel between the gene sequences is defined using R-convolution [1]_ by
integrating a binary letter-wise kernel over all pairs of letters among a pair
of strings.

This example will generate three figures.

In the first figure, we visualize the value of the kernel, i.e. the similarity
of the sequences, using a colormap. Brighter color here indicates higher
similarity.

In the second figure, we show some regression result on a dataset of 6
sequences. Here we use the 1st, 2nd, 4th, and 5th sequences as the training set
to make predictions on the 3rd and 6th sequences.

In the third figure, we demonstrate a classification model by training on 6
sequences and make predictions on another 5 sequences. The ground truth here is
simply whether there is at least one 'A' in the sequence. Here the model makes
four correct classifications and fails on one.

.. [1] Haussler, D. (1999). Convolution kernels on discrete structures
       (Vol. 646). Technical report, Department of Computer Science, University
       of California at Santa Cruz.

"""

# %%
# 导入必要的库
import numpy as np

from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import GenericKernelMixin, Hyperparameter, Kernel

# 定义一个自定义的核函数，用于处理变长序列的相似性计算
class SequenceKernel(GenericKernelMixin, Kernel):
    """
    A minimal (but valid) convolutional kernel for sequences of variable
    lengths."""

    def __init__(self, baseline_similarity=0.5, baseline_similarity_bounds=(1e-5, 1)):
        # 初始化函数，设置基础相似度和其取值范围
        self.baseline_similarity = baseline_similarity
        self.baseline_similarity_bounds = baseline_similarity_bounds

    @property
    def hyperparameter_baseline_similarity(self):
        # 返回基础相似度的超参数对象
        return Hyperparameter(
            "baseline_similarity", "numeric", self.baseline_similarity_bounds
        )

    def _f(self, s1, s2):
        """
        kernel value between a pair of sequences
        """
        # 核函数计算，基于 R-convolution 的方法计算两个序列之间的相似度
        return sum(
            [1.0 if c1 == c2 else self.baseline_similarity for c1 in s1 for c2 in s2]
        )
    # 定义一个函数 `_g`，计算两个序列之间的核导数
    def _g(self, s1, s2):
        """
        kernel derivative between a pair of sequences
        """
        # 返回一个列表推导式计算的结果，对于序列 s1 和 s2 中对应位置的字符，相同则为 0.0，不同则为 1.0
        return sum([0.0 if c1 == c2 else 1.0 for c1 in s1 for c2 in s2])

    # 定义一个方法 `__call__`，用于调用对象，接受参数 X 和 Y，可选择是否计算梯度
    def __call__(self, X, Y=None, eval_gradient=False):
        # 如果没有传入 Y，则默认 Y 等于 X
        if Y is None:
            Y = X

        # 如果需要计算梯度
        if eval_gradient:
            # 返回一个元组，第一个元素是 X 和 Y 组成的二维数组，每个元素是调用 `_f` 方法计算的结果
            # 第二个元素是 X 和 Y 组成的三维数组，每个元素是调用 `_g` 方法计算的结果
            return (
                np.array([[self._f(x, y) for y in Y] for x in X]),
                np.array([[[self._g(x, y)] for y in Y] for x in X]),
            )
        else:
            # 如果不需要计算梯度，返回一个二维数组，每个元素是调用 `_f` 方法计算的结果
            return np.array([[self._f(x, y) for y in Y] for x in X])

    # 定义一个方法 `diag`，用于计算对角元素
    def diag(self, X):
        # 返回一个一维数组，每个元素是调用 `_f` 方法计算 X 中每个元素与自身的结果
        return np.array([self._f(x, x) for x in X])

    # 定义一个方法 `is_stationary`，返回 False，表示对象不是静止的
    def is_stationary(self):
        return False

    # 定义一个方法 `clone_with_theta`，用给定的 theta 克隆对象并设置新的 theta 值
    def clone_with_theta(self, theta):
        # 调用 `clone` 方法克隆当前对象
        cloned = clone(self)
        # 设置克隆对象的 theta 属性为给定的 theta 值
        cloned.theta = theta
        return cloned
# 创建一个序列核对象
kernel = SequenceKernel()

# %%
# 序列核下的序列相似性矩阵
# ==========================

# 导入绘图库
import matplotlib.pyplot as plt

# 给定序列数据
X = np.array(["AGCT", "AGC", "AACT", "TAA", "AAA", "GAACA"])

# 计算核函数下的相似性矩阵
K = kernel(X)
# 计算对角线元素
D = kernel.diag(X)

# 绘制图像
plt.figure(figsize=(8, 5))
plt.imshow(np.diag(D**-0.5).dot(K).dot(np.diag(D**-0.5)))
plt.xticks(np.arange(len(X)), X)
plt.yticks(np.arange(len(X)), X)
plt.title("Sequence similarity under the kernel")
plt.show()

# %%
# 回归
# =====

# 定义序列数据和对应的目标值
X = np.array(["AGCT", "AGC", "AACT", "TAA", "AAA", "GAACA"])
Y = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

# 选择训练数据的索引
training_idx = [0, 1, 3, 4]

# 创建高斯过程回归模型
gp = GaussianProcessRegressor(kernel=kernel)
# 在训练数据上拟合模型
gp.fit(X[training_idx], Y[training_idx])

# 绘制预测结果和训练数据
plt.figure(figsize=(8, 5))
plt.bar(np.arange(len(X)), gp.predict(X), color="b", label="prediction")
plt.bar(training_idx, Y[training_idx], width=0.2, color="r", alpha=1, label="training")
plt.xticks(np.arange(len(X)), X)
plt.title("Regression on sequences")
plt.legend()
plt.show()

# %%
# 分类
# =====

# 训练数据集和对应的标签
X_train = np.array(["AGCT", "CGA", "TAAC", "TCG", "CTTT", "TGCT"])
Y_train = np.array([True, True, True, False, False, False])

# 创建高斯过程分类器
gp = GaussianProcessClassifier(kernel)
# 在训练数据上拟合模型
gp.fit(X_train, Y_train)

# 测试数据集和对应的真实标签
X_test = ["AAA", "ATAG", "CTC", "CT", "C"]
Y_test = [True, True, False, False, False]

# 绘制分类结果的散点图
plt.figure(figsize=(8, 5))
plt.scatter(
    np.arange(len(X_train)),
    [1.0 if c else -1.0 for c in Y_train],
    s=100,
    marker="o",
    edgecolor="none",
    facecolor=(1, 0.75, 0),
    label="training",
)
plt.scatter(
    len(X_train) + np.arange(len(X_test)),
    [1.0 if c else -1.0 for c in Y_test],
    s=100,
    marker="o",
    edgecolor="none",
    facecolor="r",
    label="truth",
)
plt.scatter(
    len(X_train) + np.arange(len(X_test)),
    [1.0 if c else -1.0 for c in gp.predict(X_test)],
    s=100,
    marker="x",
    facecolor="b",
    linewidth=2,
    label="prediction",
)
plt.xticks(np.arange(len(X_train) + len(X_test)), np.concatenate((X_train, X_test)))
plt.yticks([-1, 1], [False, True])
plt.title("Classification on sequences")
plt.legend()
plt.show()
```