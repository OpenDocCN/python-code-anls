# `D:\src\scipysrc\scikit-learn\benchmarks\bench_isolation_forest.py`

```
"""
==========================================
IsolationForest benchmark
==========================================
A test of IsolationForest on classical anomaly detection datasets.

The benchmark is run as follows:
1. The dataset is randomly split into a training set and a test set, both
   assumed to contain outliers.
2. Isolation Forest is trained on the training set.
3. The ROC curve is computed on the test set using the knowledge of the labels.

Note that the smtp dataset contains a very small proportion of outliers.
Therefore, depending on the seed of the random number generator, randomly
splitting the data set might lead to a test set containing no outliers. In this
case a warning is raised when computing the ROC curve.
"""

from time import time  # 导入时间模块中的time函数

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，并将其重命名为plt
import numpy as np  # 导入numpy模块，并将其重命名为np

from sklearn.datasets import fetch_covtype, fetch_kddcup99, fetch_openml  # 从sklearn.datasets中导入fetch函数
from sklearn.ensemble import IsolationForest  # 从sklearn.ensemble中导入IsolationForest类
from sklearn.metrics import auc, roc_curve  # 从sklearn.metrics中导入auc函数和roc_curve函数
from sklearn.preprocessing import LabelBinarizer  # 从sklearn.preprocessing中导入LabelBinarizer类
from sklearn.utils import shuffle as sh  # 从sklearn.utils中导入shuffle函数，并将其重命名为sh

print(__doc__)  # 打印脚本开头的文档字符串


def print_outlier_ratio(y):
    """
    Helper function to show the distinct value count of element in the target.
    Useful indicator for the datasets used in bench_isolation_forest.py.
    """
    uniq, cnt = np.unique(y, return_counts=True)  # 计算y中每个元素的出现次数
    print("----- Target count values: ")
    for u, c in zip(uniq, cnt):
        print("------ %s -> %d occurrences" % (str(u), c))  # 打印每个元素及其出现次数
    print("----- Outlier ratio: %.5f" % (np.min(cnt) / len(y)))  # 打印异常值的比例


random_state = 1  # 设置随机数种子

fig_roc, ax_roc = plt.subplots(1, 1, figsize=(8, 5))  # 创建一个8x5大小的图形对象和一个子图对象ax_roc

# Set this to true for plotting score histograms for each dataset:
with_decision_function_histograms = False  # 是否绘制每个数据集的分数直方图，设置为False

# datasets available = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']
datasets = ["http", "smtp", "SA", "SF", "shuttle", "forestcover"]  # 数据集列表

# Loop over all datasets for fitting and scoring the estimator:
for dat in datasets:
    # Loading and vectorizing the data:
    print("====== %s ======" % dat)  # 打印当前数据集的名称
    print("--- Fetching data...")  # 打印正在获取数据

    if dat in ["http", "smtp", "SF", "SA"]:
        dataset = fetch_kddcup99(
            subset=dat, shuffle=True, percent10=True, random_state=random_state
        )  # 从KDD Cup 99数据集中获取指定子集的数据
        X = dataset.data  # 获取特征数据
        y = dataset.target  # 获取目标数据

    if dat == "shuttle":
        dataset = fetch_openml("shuttle", as_frame=False)  # 从OpenML平台获取“shuttle”数据集，不返回DataFrame格式
        X = dataset.data  # 获取特征数据
        y = dataset.target.astype(np.int64)  # 获取目标数据，并转换为int64类型
        X, y = sh(X, y, random_state=random_state)  # 随机打乱数据集的顺序
        # we remove data with label 4
        # normal data are then those of class 1
        s = y != 4  # 选择目标数据中不等于4的数据索引
        X = X[s, :]  # 根据索引选择特征数据
        y = y[s]  # 根据索引选择目标数据
        y = (y != 1).astype(int)  # 将目标数据中非1的数据转换为整数型
        print("----- ")
    # 如果数据集是 "forestcover"，则获取覆盖类型数据集并准备数据
    if dat == "forestcover":
        dataset = fetch_covtype(shuffle=True, random_state=random_state)
        X = dataset.data  # 获取数据集特征
        y = dataset.target  # 获取数据集标签
        # 标准数据是具有属性 2 的数据
        # 异常数据是具有属性 4 的数据
        s = (y == 2) + (y == 4)
        X = X[s, :]  # 仅保留标准或异常数据的特征数据
        y = y[s]  # 仅保留标准或异常数据的标签
        y = (y != 2).astype(int)  # 将标签转换为二进制形式，标准为 0，异常为 1
        print_outlier_ratio(y)  # 打印异常比率信息

    # 输出数据向量化信息
    print("--- Vectorizing data...")

    # 如果数据集是 "SF"，使用标签二值化处理特征列，并调整数据集 X 和 y
    if dat == "SF":
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        X = np.c_[X[:, :1], x1, X[:, 2:]]  # 合并转换后的特征列到原数据集中
        y = (y != b"normal.").astype(int)  # 将标签转换为二进制形式，正常为 0，异常为 1
        print_outlier_ratio(y)  # 打印异常比率信息

    # 如果数据集是 "SA"，对多列特征进行标签二值化处理，并调整数据集 X 和 y
    if dat == "SA":
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        x2 = lb.fit_transform(X[:, 2].astype(str))
        x3 = lb.fit_transform(X[:, 3].astype(str))
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]  # 合并转换后的特征列到原数据集中
        y = (y != b"normal.").astype(int)  # 将标签转换为二进制形式，正常为 0，异常为 1
        print_outlier_ratio(y)  # 打印异常比率信息

    # 如果数据集为 "http" 或 "smtp"，将标签转换为二进制形式，正常为 0，异常为 1
    if dat in ("http", "smtp"):
        y = (y != b"normal.").astype(int)
        print_outlier_ratio(y)  # 打印异常比率信息

    n_samples, n_features = X.shape  # 获取数据集样本数和特征数
    n_samples_train = n_samples // 2  # 计算训练样本数

    X = X.astype(float)  # 将数据集 X 转换为浮点型
    X_train = X[:n_samples_train, :]  # 训练集特征数据
    X_test = X[n_samples_train:, :]  # 测试集特征数据
    y_train = y[:n_samples_train]  # 训练集标签数据
    y_test = y[n_samples_train:]  # 测试集标签数据

    # 输出拟合 IsolationForest 估计器的信息
    print("--- Fitting the IsolationForest estimator...")
    model = IsolationForest(n_jobs=-1, random_state=random_state)  # 创建 IsolationForest 模型
    tstart = time()  # 记录开始时间
    model.fit(X_train)  # 在训练集上拟合模型
    fit_time = time() - tstart  # 计算拟合时间
    tstart = time()  # 记录开始时间

    scoring = -model.decision_function(X_test)  # 计算测试集上的决策函数值，越低越异常

    # 输出准备绘图元素的信息
    print("--- Preparing the plot elements...")
    if with_decision_function_histograms:
        fig, ax = plt.subplots(3, sharex=True, sharey=True)  # 创建包含三个子图的图形
        bins = np.linspace(-0.5, 0.5, 200)  # 设置直方图的区间和数量
        ax[0].hist(scoring, bins, color="black")  # 绘制总体决策函数直方图
        ax[0].set_title("Decision function for %s dataset" % dat)  # 设置标题
        ax[1].hist(scoring[y_test == 0], bins, color="b", label="normal data")  # 绘制正常数据的直方图
        ax[1].legend(loc="lower right")  # 添加图例
        ax[2].hist(scoring[y_test == 1], bins, color="r", label="outliers")  # 绘制异常数据的直方图
        ax[2].legend(loc="lower right")  # 添加图例

    # 显示 ROC 曲线信息
    predict_time = time() - tstart  # 计算预测时间
    fpr, tpr, thresholds = roc_curve(y_test, scoring)  # 计算 ROC 曲线的 FPR、TPR 和阈值
    auc_score = auc(fpr, tpr)  # 计算 AUC 分数
    label = "%s (AUC: %0.3f, train_time= %0.2fs, test_time= %0.2fs)" % (
        dat,
        auc_score,
        fit_time,
        predict_time,
    )
    # 输出 AUC 分数和训练/测试时间
    print(label)
    ax_roc.plot(fpr, tpr, lw=1, label=label)  # 绘制 ROC 曲线
# 设置 ROC 曲线的 x 轴范围，从 -0.05 到 1.05
ax_roc.set_xlim([-0.05, 1.05])
# 设置 ROC 曲线的 y 轴范围，从 -0.05 到 1.05
ax_roc.set_ylim([-0.05, 1.05])
# 设置 x 轴的标签为 "False Positive Rate"，用于 ROC 曲线
ax_roc.set_xlabel("False Positive Rate")
# 设置 y 轴的标签为 "True Positive Rate"，用于 ROC 曲线
ax_roc.set_ylabel("True Positive Rate")
# 设置图表的标题为 "Receiver operating characteristic (ROC) curves"
ax_roc.set_title("Receiver operating characteristic (ROC) curves")
# 在图表的右下角添加图例
ax_roc.legend(loc="lower right")
# 调整图表的布局，使其更紧凑
fig_roc.tight_layout()
# 显示图表
plt.show()
```