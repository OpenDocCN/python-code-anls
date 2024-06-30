# `D:\src\scipysrc\scikit-learn\benchmarks\bench_lof.py`

```
"""
============================
LocalOutlierFactor benchmark
============================

A test of LocalOutlierFactor on classical anomaly detection datasets.

Note that LocalOutlierFactor is not meant to predict on a test set and its
performance is assessed in an outlier detection context:
1. The model is trained on the whole dataset which is assumed to contain
outliers.
2. The ROC curve is computed on the same dataset using the knowledge of the
labels.
In this context there is no need to shuffle the dataset because the model
is trained and tested on the whole dataset. The randomness of this benchmark
is only caused by the random selection of anomalies in the SA dataset.

"""

# 导入所需的库和模块
from time import time  # 导入时间计算模块

import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库

from sklearn.datasets import fetch_covtype, fetch_kddcup99, fetch_openml  # 导入数据集获取函数
from sklearn.metrics import auc, roc_curve  # 导入评估指标函数
from sklearn.neighbors import LocalOutlierFactor  # 导入局部异常因子模型
from sklearn.preprocessing import LabelBinarizer  # 导入标签二值化模块

print(__doc__)  # 打印文件开头的注释信息

random_state = 2  # 控制在SA数据集中随机选择异常数据的随机种子

# 可用的数据集列表
datasets = ["http", "smtp", "SA", "SF", "shuttle", "forestcover"]

plt.figure()  # 创建绘图对象
for dataset_name in datasets:
    # 加载和向量化数据集
    print("loading data")
    if dataset_name in ["http", "smtp", "SA", "SF"]:
        # 对于KDD Cup 99数据集的子集，加载特定的数据子集并设置随机种子
        dataset = fetch_kddcup99(
            subset=dataset_name, percent10=True, random_state=random_state
        )
        X = dataset.data  # 特征数据
        y = dataset.target  # 目标变量

    if dataset_name == "shuttle":
        # 对于OpenML上的shuttle数据集，加载数据并转换成整数类型的目标变量
        dataset = fetch_openml("shuttle", as_frame=False)
        X = dataset.data  # 特征数据
        y = dataset.target.astype(np.int64)  # 目标变量，转换为int64类型
        # 移除标签为4的数据，将标签1视为正常数据，标签其他的转换成0或1
        s = y != 4
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)

    if dataset_name == "forestcover":
        # 对于covtype数据集，加载数据并进行处理
        dataset = fetch_covtype()
        X = dataset.data  # 特征数据
        y = dataset.target  # 目标变量
        # 将属性2视为正常数据，属性4视为异常数据
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)

    print("vectorizing data")  # 向量化数据集

    if dataset_name == "SF":
        # 对于SF数据集，进行标签二值化处理
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != b"normal.").astype(int)

    if dataset_name == "SA":
        # 对于SA数据集，进行多个特征的标签二值化处理
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        x2 = lb.fit_transform(X[:, 2].astype(str))
        x3 = lb.fit_transform(X[:, 3].astype(str))
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != b"normal.").astype(int)

    if dataset_name == "http" or dataset_name == "smtp":
        # 对于http和smtp数据集，将标签转换为二进制
        y = (y != b"normal.").astype(int)

    X = X.astype(float)  # 将特征数据转换为浮点型

    print("LocalOutlierFactor processing...")  # 输出信息：处理局部异常因子模型
    model = LocalOutlierFactor(n_neighbors=20)  # 初始化局部异常因子模型
    tstart = time()  # 记录开始时间
    model.fit(X)  # 拟合模型到数据集
    # 计算模型拟合时间
    fit_time = time() - tstart

    # 获取模型的异常因子（负数表示越接近正常）
    scoring = -model.negative_outlier_factor_

    # 计算ROC曲线的假阳率（FPR）、真阳率（TPR）和阈值（thresholds）
    fpr, tpr, thresholds = roc_curve(y, scoring)

    # 计算ROC曲线下面积（AUC）
    AUC = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.plot(
        fpr,
        tpr,
        lw=1,
        label="ROC for %s (area = %0.3f, train-time: %0.2fs)"
        % (dataset_name, AUC, fit_time),
    )
# 设置 x 轴的范围从 -0.05 到 1.05
plt.xlim([-0.05, 1.05])
# 设置 y 轴的范围从 -0.05 到 1.05
plt.ylim([-0.05, 1.05])
# 设置 x 轴的标签为 "False Positive Rate"
plt.xlabel("False Positive Rate")
# 设置 y 轴的标签为 "True Positive Rate"
plt.ylabel("True Positive Rate")
# 设置图表的标题为 "Receiver operating characteristic"
plt.title("Receiver operating characteristic")
# 在图表的右下角添加图例
plt.legend(loc="lower right")
# 显示绘制好的图表
plt.show()
```