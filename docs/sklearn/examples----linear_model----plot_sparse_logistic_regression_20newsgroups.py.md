# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_sparse_logistic_regression_20newsgroups.py`

```
"""
====================================================
Multiclass sparse logistic regression on 20newgroups
====================================================

Comparison of multinomial logistic L1 vs one-versus-rest L1 logistic regression
to classify documents from the newgroups20 dataset. Multinomial logistic
regression yields more accurate results and is faster to train on the larger
scale dataset.

Here we use the l1 sparsity that trims the weights of not informative
features to zero. This is good if the goal is to extract the strongly
discriminative vocabulary of each class. If the goal is to get the best
predictive accuracy, it is better to use the non sparsity-inducing l2 penalty
instead.

A more traditional (and possibly better) way to predict on a sparse subset of
input features would be to use univariate feature selection followed by a
traditional (l2-penalised) logistic regression model.

"""

# Author: Arthur Mensch

import timeit  # 导入计时模块
import warnings  # 导入警告处理模块

import matplotlib.pyplot as plt  # 导入绘图模块
import numpy as np  # 导入数值计算模块

from sklearn.datasets import fetch_20newsgroups_vectorized  # 导入20newsgroups数据集加载模块
from sklearn.exceptions import ConvergenceWarning  # 导入收敛警告异常类
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.multiclass import OneVsRestClassifier  # 导入一对多分类器

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")  # 忽略收敛警告
t0 = timeit.default_timer()  # 记录开始时间

# We use SAGA solver
solver = "saga"  # 指定优化器为SAGA（Stochastic Average Gradient descent）

# Turn down for faster run time
n_samples = 5000  # 指定样本数

X, y = fetch_20newsgroups_vectorized(subset="all", return_X_y=True)  # 加载所有20newsgroups数据集，并返回特征X和目标y
X = X[:n_samples]  # 仅选择部分样本
y = y[:n_samples]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.1
)
train_samples, n_features = X_train.shape  # 获取训练样本数和特征数
n_classes = np.unique(y).shape[0]  # 获取类别数

print(
    "Dataset 20newsgroup, train_samples=%i, n_features=%i, n_classes=%i"
    % (train_samples, n_features, n_classes)
)

models = {
    "ovr": {"name": "One versus Rest", "iters": [1, 2, 3]},
    "multinomial": {"name": "Multinomial", "iters": [1, 2, 5]},
}

for model in models:
    # Add initial chance-level values for plotting purpose
    accuracies = [1 / n_classes]  # 初始化准确率列表，用于绘图目的
    times = [0]  # 初始化时间列表，用于绘图目的
    densities = [1]  # 初始化密度列表，用于绘图目的

    model_params = models[model]

    # Small number of epochs for fast runtime
    # 遍历模型参数中的迭代次数列表
    for this_max_iter in model_params["iters"]:
        # 打印当前模型、求解器及其迭代次数
        print(
            "[model=%s, solver=%s] Number of epochs: %s"
            % (model_params["name"], solver, this_max_iter)
        )
        # 创建逻辑回归分类器对象，设置求解器、正则化方式为L1、最大迭代次数和随机种子
        clf = LogisticRegression(
            solver=solver,
            penalty="l1",
            max_iter=this_max_iter,
            random_state=42,
        )
        # 如果模型为"ovr"，则将分类器包装成One-vs-Rest分类器
        if model == "ovr":
            clf = OneVsRestClassifier(clf)
        # 计时开始
        t1 = timeit.default_timer()
        # 使用训练数据拟合分类器
        clf.fit(X_train, y_train)
        # 计算训练时间
        train_time = timeit.default_timer() - t1

        # 使用测试数据进行预测
        y_pred = clf.predict(X_test)
        # 计算准确率
        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
        
        # 如果模型为"ovr"，则将每个估计器的系数连接起来
        if model == "ovr":
            coef = np.concatenate([est.coef_ for est in clf.estimators_])
        else:
            coef = clf.coef_
        
        # 计算非零系数的密度，作为稀疏性度量
        density = np.mean(coef != 0, axis=1) * 100
        
        # 将准确率、稀疏度和训练时间添加到对应的列表中
        accuracies.append(accuracy)
        densities.append(density)
        times.append(train_time)
    
    # 将当前模型的训练时间、稀疏度和准确率存储到模型字典中
    models[model]["times"] = times
    models[model]["densities"] = densities
    models[model]["accuracies"] = accuracies
    
    # 打印当前模型在测试数据上的准确率
    print("Test accuracy for model %s: %.4f" % (model, accuracies[-1]))
    # 打印当前模型的非零系数百分比
    print(
        "%% non-zero coefficients for model %s, per class:\n %s"
        % (model, densities[-1])
    )
    # 打印当前模型的运行时间
    print(
        "Run time (%i epochs) for model %s:%.2f"
        % (model_params["iters"][-1], model, times[-1])
    )
# 创建一个新的图形窗口
fig = plt.figure()
# 在图形窗口中添加一个子图，使用单一的子图布局（1行1列中的第一个）
ax = fig.add_subplot(111)

# 遍历模型列表中的每个模型
for model in models:
    # 从模型字典中获取模型名称
    name = models[model]["name"]
    # 从模型字典中获取模型训练时间数据
    times = models[model]["times"]
    # 从模型字典中获取模型测试准确率数据
    accuracies = models[model]["accuracies"]
    
    # 在子图中绘制时间与准确率的关系图，使用圆圈标记
    ax.plot(times, accuracies, marker="o", label="Model: %s" % name)
    # 设置子图的横坐标标签
    ax.set_xlabel("Train time (s)")
    # 设置子图的纵坐标标签
    ax.set_ylabel("Test accuracy")

# 在子图中添加图例
ax.legend()
# 设置整个图形的标题
fig.suptitle("Multinomial vs One-vs-Rest Logistic L1\nDataset %s" % "20newsgroups")
# 调整图形布局使其更紧凑
fig.tight_layout()
# 调整子图之间的间距，使得标题不重叠
fig.subplots_adjust(top=0.85)

# 计算代码运行时间
run_time = timeit.default_timer() - t0
# 打印代码运行时间
print("Example run in %.3f s" % run_time)
# 显示图形窗口
plt.show()
```