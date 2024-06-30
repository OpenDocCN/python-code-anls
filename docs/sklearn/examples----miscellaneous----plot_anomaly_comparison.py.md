# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_anomaly_comparison.py`

```
"""
============================================================================
Comparing anomaly detection algorithms for outlier detection on toy datasets
============================================================================

This example shows characteristics of different anomaly detection algorithms
on 2D datasets. Datasets contain one or two modes (regions of high density)
to illustrate the ability of algorithms to cope with multimodal data.

For each dataset, 15% of samples are generated as random uniform noise. This
proportion is the value given to the nu parameter of the OneClassSVM and the
contamination parameter of the other outlier detection algorithms.
Decision boundaries between inliers and outliers are displayed in black
except for Local Outlier Factor (LOF) as it has no predict method to be applied
on new data when it is used for outlier detection.

The :class:`~sklearn.svm.OneClassSVM` is known to be sensitive to outliers and
thus does not perform very well for outlier detection. This estimator is best
suited for novelty detection when the training set is not contaminated by
outliers. That said, outlier detection in high-dimension, or without any
assumptions on the distribution of the inlying data is very challenging, and a
One-class SVM might give useful results in these situations depending on the
value of its hyperparameters.

The :class:`sklearn.linear_model.SGDOneClassSVM` is an implementation of the
One-Class SVM based on stochastic gradient descent (SGD). Combined with kernel
approximation, this estimator can be used to approximate the solution
of a kernelized :class:`sklearn.svm.OneClassSVM`. We note that, although not
identical, the decision boundaries of the
:class:`sklearn.linear_model.SGDOneClassSVM` and the ones of
:class:`sklearn.svm.OneClassSVM` are very similar. The main advantage of using
:class:`sklearn.linear_model.SGDOneClassSVM` is that it scales linearly with
the number of samples.

:class:`sklearn.covariance.EllipticEnvelope` assumes the data is Gaussian and
learns an ellipse. It thus degrades when the data is not unimodal. Notice
however that this estimator is robust to outliers.

:class:`~sklearn.ensemble.IsolationForest` and
:class:`~sklearn.neighbors.LocalOutlierFactor` seem to perform reasonably well
for multi-modal data sets. The advantage of
:class:`~sklearn.neighbors.LocalOutlierFactor` over the other estimators is
shown for the third data set, where the two modes have different densities.
This advantage is explained by the local aspect of LOF, meaning that it only
compares the score of abnormality of one sample with the scores of its
neighbors.

Finally, for the last data set, it is hard to say that one sample is more
abnormal than another sample as they are uniformly distributed in a
hypercube. Except for the :class:`~sklearn.svm.OneClassSVM` which overfits a
little, all estimators present decent solutions for this situation. In such a
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 引入时间模块
import time

# 引入 matplotlib 库并设置负等高线样式为实线
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 引入 sklearn 中的异常检测相关模块和方法
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs, make_moons
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline

# 设置 matplotlib 参数，使得负等高线样式为实线
matplotlib.rcParams["contour.negative_linestyle"] = "solid"

# 设定示例中的数据集样本数量
n_samples = 300
# 异常点的比例
outliers_fraction = 0.15
# 异常点的数量
n_outliers = int(outliers_fraction * n_samples)
# 正常样本的数量
n_inliers = n_samples - n_outliers

# 定义用于比较的异常检测方法列表
anomaly_algorithms = [
    (
        "Robust covariance",
        EllipticEnvelope(contamination=outliers_fraction, random_state=42),
    ),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
    (
        "One-Class SVM (SGD)",
        make_pipeline(
            Nystroem(gamma=0.1, random_state=42, n_components=150),
            SGDOneClassSVM(
                nu=outliers_fraction,
                shuffle=True,
                fit_intercept=True,
                random_state=42,
                tol=1e-6,
            ),
        ),
    ),
    (
        "Isolation Forest",
        IsolationForest(contamination=outliers_fraction, random_state=42),
    ),
    (
        "Local Outlier Factor",
        LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction),
    ),
]

# 定义不同的数据集
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
datasets = [
    make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5], **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, 0.3], **blobs_params)[0],
    4.0
    * (
        make_moons(n_samples=n_samples, noise=0.05, random_state=0)[0]
        - np.array([0.5, 0.25])
    ),
    14.0 * (np.random.RandomState(42).rand(n_samples, 2) - 0.5),
]

# 比较给定设置下的分类器性能
xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))

# 创建绘图窗口
plt.figure(figsize=(len(anomaly_algorithms) * 2 + 4, 12.5))
# 调整子图布局
plt.subplots_adjust(
    ```
    # 设置子图的布局参数，指定左、右、底、顶的边界空白比例，以及子图之间的宽度空白和高度空白比例
    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
# 初始化绘图编号为1
plot_num = 1
# 使用种子42创建随机数生成器
rng = np.random.RandomState(42)

# 遍历数据集列表，每次迭代将索引和数据集X作为元组返回
for i_dataset, X in enumerate(datasets):
    # 添加异常值到数据集X中，异常值范围在[-6, 6]之间，形状为(n_outliers, 2)
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)

    # 遍历异常检测算法列表，每次迭代将算法名和算法本身作为元组返回
    for name, algorithm in anomaly_algorithms:
        # 记录开始时间
        t0 = time.time()
        # 使用当前算法拟合数据集X
        algorithm.fit(X)
        # 记录结束时间
        t1 = time.time()

        # 将当前子图设置为网格布局中的第plot_num个位置
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)

        # 如果是第一个数据集，设置子图标题为当前算法名字，字体大小为18
        if i_dataset == 0:
            plt.title(name, size=18)

        # 根据算法名进行数据拟合和异常标记
        if name == "Local Outlier Factor":
            # 对于局部异常因子算法，使用fit_predict同时进行拟合和预测
            y_pred = algorithm.fit_predict(X)
        else:
            # 对于其他算法，先拟合数据，然后用拟合好的模型预测数据
            y_pred = algorithm.fit(X).predict(X)

        # 绘制等高线和数据点
        if name != "Local Outlier Factor":  # LOF算法不支持predict方法
            # 生成网格点并预测异常值，然后将结果重塑为与xx形状相同的Z
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")

        # 设置颜色数组，根据y_pred的值选择颜色进行数据点的散点图绘制
        colors = np.array(["#377eb8", "#ff7f00"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

        # 设置x和y轴的显示范围
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        # 隐藏x和y轴刻度
        plt.xticks(())
        plt.yticks(())
        
        # 在当前轴的0.99, 0.01位置添加文本，显示算法运行时间（单位为秒），右对齐
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        
        # 更新绘图编号
        plot_num += 1

# 显示所有绘图
plt.show()
```