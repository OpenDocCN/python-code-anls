# `D:\src\scipysrc\scikit-learn\benchmarks\bench_online_ocsvm.py`

```
# 导入所需的库和模块
from time import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.datasets import fetch_covtype, fetch_kddcup99  # 导入数据集获取函数
from sklearn.kernel_approximation import Nystroem  # 导入核近似函数
from sklearn.linear_model import SGDOneClassSVM  # 导入SGD单类SVM模型
from sklearn.metrics import auc, roc_curve  # 导入AUC和ROC曲线评估函数
from sklearn.pipeline import make_pipeline  # 导入管道构建函数
from sklearn.preprocessing import LabelBinarizer, StandardScaler  # 导入数据预处理函数
from sklearn.svm import OneClassSVM  # 导入单类SVM模型
from sklearn.utils import shuffle  # 导入数据重排函数

# 设置matplotlib的字体
font = {"weight": "normal", "size": 15}
matplotlib.rc("font", **font)

print(__doc__)  # 打印脚本文件开头的文档字符串

def print_outlier_ratio(y):
    """
    辅助函数：显示目标变量中各元素的唯一值计数。
    对bench_isolation_forest.py中使用的数据集是一个有用的指标。
    """
    uniq, cnt = np.unique(y, return_counts=True)
    print("----- Target count values: ")
    for u, c in zip(uniq, cnt):
        print("------ %s -> %d occurrences" % (str(u), c))
    print("----- Outlier ratio: %.5f" % (np.min(cnt) / len(y)))

# 用于ROC曲线计算的参数设置
n_axis = 1000
x_axis = np.linspace(0, 1, n_axis)

datasets = ["http", "smtp", "SA", "SF", "forestcover"]  # 待处理的数据集名称列表

novelty_detection = False  # 如果为False，训练集中包含异常值

random_states = [42]  # 随机种子列表
nu = 0.05  # SVM中的nu参数

results_libsvm = np.empty((len(datasets), n_axis + 5))  # 用于存储LibSVM结果的数组
results_online = np.empty((len(datasets), n_axis + 5))  # 用于存储在线SGDOneClassSVM结果的数组

for dat, dataset_name in enumerate(datasets):
    print(dataset_name)  # 打印当前处理的数据集名称

    # 加载数据集
    if dataset_name in ["http", "smtp", "SA", "SF"]:
        dataset = fetch_kddcup99(
            subset=dataset_name, shuffle=False, percent10=False, random_state=88
        )
        X = dataset.data  # 获取数据
        y = dataset.target  # 获取目标变量

    if dataset_name == "forestcover":
        dataset = fetch_covtype(shuffle=False)
        X = dataset.data
        y = dataset.target
        # 正常数据的标签是2，异常数据的标签是4
        s = (y == 2) + (y == 4)
        X = X[s, :]  # 选取符合条件的数据样本
        y = y[s]
        y = (y != 2).astype(int)  # 将目标变量转换为二元标签

    # 数据向量化处理
    if dataset_name == "SF":
        # 如果数据集名为"SF"，需要将X的第二列转换为字符串类型，
        # 以便对字符串类别特征应用LabelBinarizer
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        # 使用LabelBinarizer处理后的特征替换原X中的第二列
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        # 将目标变量y转换为整数类型，正常类别"normal."为0，异常类别为1
        y = (y != b"normal.").astype(int)

    if dataset_name == "SA":
        # 如果数据集名为"SA"，需要将X的第1、2、3列分别转换为字符串类型，
        # 以便对字符串类别特征应用LabelBinarizer
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        x2 = lb.fit_transform(X[:, 2].astype(str))
        x3 = lb.fit_transform(X[:, 3].astype(str))
        # 使用LabelBinarizer处理后的特征替换原X中的第1、2、3列
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        # 将目标变量y转换为整数类型，正常类别"normal."为0，异常类别为1
        y = (y != b"normal.").astype(int)

    if dataset_name in ["http", "smtp"]:
        # 如果数据集名为"http"或"smtp"，将目标变量y转换为整数类型，
        # 正常类别"normal."为0，异常类别为1
        y = (y != b"normal.").astype(int)

    # 打印异常样本比例
    print_outlier_ratio(y)

    # 获取数据集样本数和特征数
    n_samples, n_features = np.shape(X)
    
    # 根据数据集名确定训练样本数
    if dataset_name == "SA":
        n_samples_train = n_samples // 20  # LibSVM处理时间过长，取样本数的1/20
    else:
        n_samples_train = n_samples // 2  # 默认取样本数的1/2
    
    # 计算测试样本数
    n_samples_test = n_samples - n_samples_train
    
    # 打印训练样本数和特征数
    print("n_train: ", n_samples_train)
    print("n_features: ", n_features)

    # 初始化用于存储指标的数组
    tpr_libsvm = np.zeros(n_axis)
    tpr_online = np.zeros(n_axis)
    fit_time_libsvm = 0
    fit_time_online = 0
    predict_time_libsvm = 0
    predict_time_online = 0

    # 将特征矩阵X转换为浮点数类型
    X = X.astype(float)

    # 设置OCSVM的默认参数gamma
    gamma = 1 / n_features  # OCSVM默认参数
    # 对于每一个随机状态，依次执行以下操作
    for random_state in random_states:
        # 打印当前随机状态的信息
        print("random state: %s" % random_state)

        # 对特征和标签进行随机重排，使用当前随机状态
        X, y = shuffle(X, y, random_state=random_state)
        
        # 将数据划分为训练集和测试集
        X_train = X[:n_samples_train]
        X_test = X[n_samples_train:]
        y_train = y[:n_samples_train]
        y_test = y[n_samples_train:]

        # 如果是新颖性检测任务，则仅保留训练集中标签为0的样本
        if novelty_detection:
            X_train = X_train[y_train == 0]
            y_train = y_train[y_train == 0]

        # 标准化数据
        std = StandardScaler()

        # 打印信息，指示当前使用的算法为LibSVM的OCSVM
        print("----------- LibSVM OCSVM ------------")
        
        # 初始化LibSVM的One-Class SVM模型
        ocsvm = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
        pipe_libsvm = make_pipeline(std, ocsvm)

        # 开始计时
        tstart = time()
        # 使用LibSVM模型拟合训练集
        pipe_libsvm.fit(X_train)
        fit_time_libsvm += time() - tstart

        # 开始计时
        tstart = time()
        # 计算测试集上的决策函数值，分数越低表示越正常
        scoring = -pipe_libsvm.decision_function(X_test)
        predict_time_libsvm += time() - tstart
        # 计算ROC曲线的假正率和真正率
        fpr_libsvm_, tpr_libsvm_, _ = roc_curve(y_test, scoring)

        # 使用插值函数计算真正率的插值
        f_libsvm = interp1d(fpr_libsvm_, tpr_libsvm_)
        tpr_libsvm += f_libsvm(x_axis)

        # 打印信息，指示当前使用的算法为Online OCSVM
        print("----------- Online OCSVM ------------")
        
        # 使用Nystroem核近似初始化Online OCSVM模型
        nystroem = Nystroem(gamma=gamma, random_state=random_state)
        online_ocsvm = SGDOneClassSVM(nu=nu, random_state=random_state)
        pipe_online = make_pipeline(std, nystroem, online_ocsvm)

        # 开始计时
        tstart = time()
        # 使用Online OCSVM模型拟合训练集
        pipe_online.fit(X_train)
        fit_time_online += time() - tstart

        # 开始计时
        tstart = time()
        # 计算测试集上的决策函数值，分数越低表示越正常
        scoring = -pipe_online.decision_function(X_test)
        predict_time_online += time() - tstart
        # 计算ROC曲线的假正率和真正率
        fpr_online_, tpr_online_, _ = roc_curve(y_test, scoring)

        # 使用插值函数计算真正率的插值
        f_online = interp1d(fpr_online_, tpr_online_)
        tpr_online += f_online(x_axis)

    # 计算LibSVM模型的平均真正率和运行时间
    tpr_libsvm /= len(random_states)
    tpr_libsvm[0] = 0.0
    fit_time_libsvm /= len(random_states)
    predict_time_libsvm /= len(random_states)
    auc_libsvm = auc(x_axis, tpr_libsvm)

    # 将LibSVM模型的结果存储到结果字典中
    results_libsvm[dat] = [
        fit_time_libsvm,
        predict_time_libsvm,
        auc_libsvm,
        n_samples_train,
        n_features,
    ] + list(tpr_libsvm)

    # 计算Online OCSVM模型的平均真正率和运行时间
    tpr_online /= len(random_states)
    tpr_online[0] = 0.0
    fit_time_online /= len(random_states)
    predict_time_online /= len(random_states)
    auc_online = auc(x_axis, tpr_online)

    # 将Online OCSVM模型的结果存储到结果字典中
    results_online[dat] = [
        fit_time_online,
        predict_time_online,
        auc_online,
        n_samples_train,
        n_features,
    ] + list(tpr_libsvm)
# -------- Plotting bar charts -------------

# 从结果数组中提取 LibSVM 的拟合时间、预测时间和 AUC 值
fit_time_libsvm_all = results_libsvm[:, 0]
predict_time_libsvm_all = results_libsvm[:, 1]
auc_libsvm_all = results_libsvm[:, 2]
# 提取训练数据集大小和特征数量
n_train_all = results_libsvm[:, 3]
n_features_all = results_libsvm[:, 4]

# 从结果数组中提取 Online SVM 的拟合时间、预测时间和 AUC 值
fit_time_online_all = results_online[:, 0]
predict_time_online_all = results_online[:, 1]
auc_online_all = results_online[:, 2]

# 设置条形图的宽度和计算条形图的位置
width = 0.7
ind = 2 * np.arange(len(datasets))

# 准备 X 轴的标签，包含数据集名称、训练数据量和特征数量的格式化字符串
x_tickslabels = [
    (name + "\n" + r"$n={:,d}$" + "\n" + r"$d={:d}$").format(int(n), int(d))
    for name, n, d in zip(datasets, n_train_all, n_features_all)
]

# 定义函数：为每个条形图添加显示高度的文本标签
def autolabel_auc(rects, ax):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            1.05 * height,
            "%.3f" % height,
            ha="center",
            va="bottom",
        )

# 定义函数：为每个条形图添加显示高度的文本标签（用于时间）
def autolabel_time(rects, ax):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            1.05 * height,
            "%.1f" % height,
            ha="center",
            va="bottom",
        )

# 创建第一个条形图：显示 AUC 值
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_ylabel("AUC")
ax.set_ylim((0, 1.3))
rect_libsvm = ax.bar(ind, auc_libsvm_all, width=width, color="r")
rect_online = ax.bar(ind + width, auc_online_all, width=width, color="y")
ax.legend((rect_libsvm[0], rect_online[0]), ("LibSVM", "Online SVM"))
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(x_tickslabels)
autolabel_auc(rect_libsvm, ax)
autolabel_auc(rect_online, ax)
plt.show()

# 创建第二个条形图：显示训练时间（对数刻度）
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_ylabel("Training time (sec) - Log scale")
ax.set_yscale("log")
rect_libsvm = ax.bar(ind, fit_time_libsvm_all, color="r", width=width)
rect_online = ax.bar(ind + width, fit_time_online_all, color="y", width=width)
ax.legend((rect_libsvm[0], rect_online[0]), ("LibSVM", "Online SVM"))
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(x_tickslabels)
autolabel_time(rect_libsvm, ax)
autolabel_time(rect_online, ax)
plt.show()

# 创建第三个条形图：显示测试时间（对数刻度）
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_ylabel("Testing time (sec) - Log scale")
ax.set_yscale("log")
rect_libsvm = ax.bar(ind, predict_time_libsvm_all, color="r", width=width)
rect_online = ax.bar(ind + width, predict_time_online_all, color="y", width=width)
ax.legend((rect_libsvm[0], rect_online[0]), ("LibSVM", "Online SVM"))
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(x_tickslabels)
autolabel_time(rect_libsvm, ax)
autolabel_time(rect_online, ax)
plt.show()
```