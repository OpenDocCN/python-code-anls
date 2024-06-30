# `D:\src\scipysrc\scikit-learn\examples\applications\plot_model_complexity_influence.py`

```
"""
==========================
Model Complexity Influence
==========================

Demonstrate how model complexity influences both prediction accuracy and
computational performance.

We will be using two datasets:
    - :ref:`diabetes_dataset` for regression.
      This dataset consists of 10 measurements taken from diabetes patients.
      The task is to predict disease progression;
    - :ref:`20newsgroups_dataset` for classification. This dataset consists of
      newsgroup posts. The task is to predict on which topic (out of 20 topics)
      the post is written about.

We will model the complexity influence on three different estimators:
    - :class:`~sklearn.linear_model.SGDClassifier` (for classification data)
      which implements stochastic gradient descent learning;

    - :class:`~sklearn.svm.NuSVR` (for regression data) which implements
      Nu support vector regression;

    - :class:`~sklearn.ensemble.GradientBoostingRegressor` builds an additive
      model in a forward stage-wise fashion. Notice that
      :class:`~sklearn.ensemble.HistGradientBoostingRegressor` is much faster
      than :class:`~sklearn.ensemble.GradientBoostingRegressor` starting with
      intermediate datasets (`n_samples >= 10_000`), which is not the case for
      this example.


We make the model complexity vary through the choice of relevant model
parameters in each of our selected models. Next, we will measure the influence
on both computational performance (latency) and predictive power (MSE or
Hamming Loss).

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time  # 导入时间模块，用于测量执行时间

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 模块，进行数值计算

from sklearn import datasets  # 导入 sklearn 中的 datasets 模块，用于加载数据集
from sklearn.ensemble import GradientBoostingRegressor  # 导入梯度提升回归模型
from sklearn.linear_model import SGDClassifier  # 导入随机梯度下降分类器
from sklearn.metrics import hamming_loss, mean_squared_error  # 导入评估指标：Hamming Loss 和 MSE
from sklearn.model_selection import train_test_split  # 导入数据集分割函数 train_test_split
from sklearn.svm import NuSVR  # 导入 Nu 支持向量回归器

# Initialize random generator
np.random.seed(0)  # 设置随机种子，确保实验可重复性

##############################################################################
# Load the data
# -------------
#
# First we load both datasets.
#
# .. note:: We are using
#    :func:`~sklearn.datasets.fetch_20newsgroups_vectorized` to download 20
#    newsgroups dataset. It returns ready-to-use features.
#
# .. note:: ``X`` of the 20 newsgroups dataset is a sparse matrix while ``X``
#    of diabetes dataset is a numpy array.
#

def generate_data(case):
    """Generate regression/classification data."""
    if case == "regression":
        X, y = datasets.load_diabetes(return_X_y=True)  # 加载糖尿病数据集
        train_size = 0.8  # 训练集比例为 80%
    elif case == "classification":
        X, y = datasets.fetch_20newsgroups_vectorized(subset="all", return_X_y=True)  # 加载 20newsgroups 数据集
        train_size = 0.4  # 为了加快示例运行速度，设置训练集比例为 40%

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=0
    )
    # 创建一个字典变量，包含训练集和测试集的特征数据和标签数据
    data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    # 返回包含数据集分割结果的字典
    return data
##############################################################################
# Benchmark influence
# -------------------
# Next, we can calculate the influence of the parameters on the given
# estimator. In each round, we will set the estimator with the new value of
# ``changing_param`` and we will be collecting the prediction times, prediction
# performance and complexities to see how those changes affect the estimator.
# We will calculate the complexity using ``complexity_computer`` passed as a
# parameter.
#

def benchmark_influence(conf):
    """
    Benchmark influence of `changing_param` on both MSE and latency.
    """
    prediction_times = []  # 存储每次预测所需的时间
    prediction_powers = []  # 存储每次预测的性能指标
    complexities = []  # 存储每次评估复杂度指标
    for param_value in conf["changing_param_values"]:
        conf["tuned_params"][conf["changing_param"]] = param_value  # 设置参数的新值到调整后的参数字典中
        estimator = conf["estimator"](**conf["tuned_params"])  # 使用调整后的参数创建评估器对象

        print("Benchmarking %s" % estimator)  # 打印当前评估器对象的信息
        estimator.fit(conf["data"]["X_train"], conf["data"]["y_train"])  # 使用训练集训练评估器
        conf["postfit_hook"](estimator)  # 执行训练后的钩子函数
        complexity = conf["complexity_computer"](estimator)  # 计算评估器的复杂度
        complexities.append(complexity)  # 将复杂度添加到列表中
        start_time = time.time()  # 记录开始时间
        for _ in range(conf["n_samples"]):
            y_pred = estimator.predict(conf["data"]["X_test"])  # 使用测试集进行预测
        elapsed_time = (time.time() - start_time) / float(conf["n_samples"])  # 计算每次预测的平均时间
        prediction_times.append(elapsed_time)  # 将平均预测时间添加到列表中
        pred_score = conf["prediction_performance_computer"](
            conf["data"]["y_test"], y_pred
        )  # 计算预测性能指标
        prediction_powers.append(pred_score)  # 将性能指标添加到列表中
        print(
            "Complexity: %d | %s: %.4f | Pred. Time: %fs\n"
            % (
                complexity,
                conf["prediction_performance_label"],
                pred_score,
                elapsed_time,
            )
        )  # 打印评估器的复杂度、性能指标和预测时间信息
    return prediction_powers, prediction_times, complexities


##############################################################################
# Choose parameters
# -----------------
#
# We choose the parameters for each of our estimators by making
# a dictionary with all the necessary values.
# ``changing_param`` is the name of the parameter which will vary in each
# estimator.
# Complexity will be defined by the ``complexity_label`` and calculated using
# `complexity_computer`.
# Also note that depending on the estimator type we are passing
# different data.
#

def _count_nonzero_coefficients(estimator):
    a = estimator.coef_.toarray()  # 转换评估器的系数矩阵为稀疏数组
    return np.count_nonzero(a)  # 统计稀疏数组中非零元素的个数


configurations = [
    {
        "estimator": SGDClassifier,  # 指定使用的分类器为 SGDClassifier
        "tuned_params": {  # 定义了用于调优的参数字典
            "penalty": "elasticnet",  # 指定惩罚项为 elasticnet
            "alpha": 0.001,  # 指定 alpha 参数为 0.001
            "loss": "modified_huber",  # 指定损失函数为 modified_huber
            "fit_intercept": True,  # 设置是否拟合截距为 True
            "tol": 1e-1,  # 设置容忍度为 0.1
            "n_iter_no_change": 2,  # 设置迭代不改变次数为 2
        },
        "changing_param": "l1_ratio",  # 指定变化的参数为 l1_ratio
        "changing_param_values": [0.25, 0.5, 0.75, 0.9],  # 定义 l1_ratio 可选的值列表
        "complexity_label": "non_zero coefficients",  # 指定复杂度度量标签为非零系数个数
        "complexity_computer": _count_nonzero_coefficients,  # 指定计算复杂度的函数为 _count_nonzero_coefficients
        "prediction_performance_computer": hamming_loss,  # 指定计算预测性能的函数为 hamming_loss
        "prediction_performance_label": "Hamming Loss (Misclassification Ratio)",  # 指定预测性能标签为 Hamming Loss
        "postfit_hook": lambda x: x.sparsify(),  # 定义后处理钩子，稀疏化处理
        "data": classification_data,  # 指定使用的数据为分类数据
        "n_samples": 5,  # 指定样本数为 5
    },
    {
        "estimator": NuSVR,  # 指定使用的回归器为 NuSVR
        "tuned_params": {"C": 1e3, "gamma": 2**-15},  # 定义了用于调优的参数字典
        "changing_param": "nu",  # 指定变化的参数为 nu
        "changing_param_values": [0.05, 0.1, 0.2, 0.35, 0.5],  # 定义 nu 可选的值列表
        "complexity_label": "n_support_vectors",  # 指定复杂度度量标签为支持向量数目
        "complexity_computer": lambda x: len(x.support_vectors_),  # 指定计算复杂度的函数为 lambda 表达式，计算支持向量的数量
        "data": regression_data,  # 指定使用的数据为回归数据
        "postfit_hook": lambda x: x,  # 定义后处理钩子，无操作
        "prediction_performance_computer": mean_squared_error,  # 指定计算预测性能的函数为 mean_squared_error
        "prediction_performance_label": "MSE",  # 指定预测性能标签为 MSE
        "n_samples": 15,  # 指定样本数为 15
    },
    {
        "estimator": GradientBoostingRegressor,  # 指定使用的回归器为 GradientBoostingRegressor
        "tuned_params": {  # 定义了用于调优的参数字典
            "loss": "squared_error",  # 指定损失函数为 squared_error
            "learning_rate": 0.05,  # 指定学习率为 0.05
            "max_depth": 2,  # 指定最大深度为 2
        },
        "changing_param": "n_estimators",  # 指定变化的参数为 n_estimators
        "changing_param_values": [10, 25, 50, 75, 100],  # 定义 n_estimators 可选的值列表
        "complexity_label": "n_trees",  # 指定复杂度度量标签为树的数量
        "complexity_computer": lambda x: x.n_estimators,  # 指定计算复杂度的函数为 lambda 表达式，返回树的数量
        "data": regression_data,  # 指定使用的数据为回归数据
        "postfit_hook": lambda x: x,  # 定义后处理钩子，无操作
        "prediction_performance_computer": mean_squared_error,  # 指定计算预测性能的函数为 mean_squared_error
        "prediction_performance_label": "MSE",  # 指定预测性能标签为 MSE
        "n_samples": 15,  # 指定样本数为 15
    },
##############################################################################
# Run the code and plot the results
# ---------------------------------
#
# We defined all the functions required to run our benchmark. Now, we will loop
# over the different configurations that we defined previously. Subsequently,
# we can analyze the plots obtained from the benchmark:
# Relaxing the `L1` penalty in the SGD classifier reduces the prediction error
# but leads to an increase in the training time.
# We can draw a similar analysis regarding the training time which increases
# with the number of support vectors with a Nu-SVR. However, we observed that
# there is an optimal number of support vectors which reduces the prediction
# error. Indeed, too few support vectors lead to an under-fitted model while
# too many support vectors lead to an over-fitted model.
# The exact same conclusion can be drawn for the gradient-boosting model. The
# only the difference with the Nu-SVR is that having too many trees in the
# ensemble is not as detrimental.
#

def plot_influence(conf, mse_values, prediction_times, complexities):
    """
    Plot influence of model complexity on both accuracy and latency.
    """
    # 创建一个新的图形
    fig = plt.figure()
    fig.subplots_adjust(right=0.75)

    # 第一个轴（预测误差）
    ax1 = fig.add_subplot(111)
    # 绘制预测误差随模型复杂度变化的图线，使用蓝色
    line1 = ax1.plot(complexities, mse_values, c="tab:blue", ls="-")[0]
    ax1.set_xlabel("Model Complexity (%s)" % conf["complexity_label"])
    y1_label = conf["prediction_performance_label"]
    ax1.set_ylabel(y1_label)

    ax1.spines["left"].set_color(line1.get_color())
    ax1.yaxis.label.set_color(line1.get_color())
    ax1.tick_params(axis="y", colors=line1.get_color())

    # 第二个轴（延迟）
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    # 绘制预测延迟随模型复杂度变化的图线，使用橙色
    line2 = ax2.plot(complexities, prediction_times, c="tab:orange", ls="-")[0]
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    y2_label = "Time (s)"
    ax2.set_ylabel(y2_label)
    ax1.spines["right"].set_color(line2.get_color())
    ax2.yaxis.label.set_color(line2.get_color())
    ax2.tick_params(axis="y", colors=line2.get_color())

    # 添加图例
    plt.legend(
        (line1, line2), ("prediction error", "prediction latency"), loc="upper center"
    )

    # 设置图的标题
    plt.title(
        "Influence of varying '%s' on %s"
        % (conf["changing_param"], conf["estimator"].__name__)
    )


for conf in configurations:
    # 获取预测性能、预测时间和复杂度的数据
    prediction_performances, prediction_times, complexities = benchmark_influence(conf)
    # 绘制影响图
    plot_influence(conf, prediction_performances, prediction_times, complexities)
plt.show()

##############################################################################
# Conclusion
# ----------
#
# As a conclusion, we can deduce the following insights:
#
# * a model which is more complex (or expressive) will require a larger
#   training time;
# * a more complex model does not guarantee to reduce the prediction error.
#
# 这些方面涉及模型的泛化能力以及避免模型欠拟合或过拟合。
```