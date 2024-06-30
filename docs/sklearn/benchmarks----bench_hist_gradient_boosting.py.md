# `D:\src\scipysrc\scikit-learn\benchmarks\bench_hist_gradient_boosting.py`

```
# 导入必要的模块
import argparse  # 用于解析命令行参数的模块
from time import time  # 时间相关的模块，用于记录运行时间

import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库

# 导入数据生成和模型相关的类和函数
from sklearn.datasets import make_classification, make_regression  
from sklearn.ensemble import (  # 导入集成学习相关的类
    HistGradientBoostingClassifier,  # 分类器
    HistGradientBoostingRegressor,  # 回归器
)
from sklearn.ensemble._hist_gradient_boosting.utils import get_equivalent_estimator  # 辅助函数
from sklearn.model_selection import train_test_split  # 数据集划分函数

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--n-leaf-nodes", type=int, default=31)  # 叶节点数
parser.add_argument("--n-trees", type=int, default=10)  # 树的数量
parser.add_argument(
    "--lightgbm", action="store_true", default=False, help="also plot lightgbm"
)  # 是否绘制 lightgbm 结果的标志
parser.add_argument(
    "--xgboost", action="store_true", default=False, help="also plot xgboost"
)  # 是否绘制 xgboost 结果的标志
parser.add_argument(
    "--catboost", action="store_true", default=False, help="also plot catboost"
)  # 是否绘制 catboost 结果的标志
parser.add_argument("--learning-rate", type=float, default=0.1)  # 学习率
parser.add_argument(
    "--problem",
    type=str,
    default="classification",
    choices=["classification", "regression"],  # 问题类型，分类或回归
)
parser.add_argument("--loss", type=str, default="default")  # 损失函数类型
parser.add_argument("--missing-fraction", type=float, default=0)  # 缺失值比例
parser.add_argument("--n-classes", type=int, default=2)  # 分类问题的类别数
parser.add_argument("--n-samples-max", type=int, default=int(1e6))  # 最大样本数
parser.add_argument("--n-features", type=int, default=20)  # 特征数
parser.add_argument("--max-bins", type=int, default=255)  # 最大箱数
parser.add_argument(
    "--random-sample-weights",
    action="store_true",
    default=False,
    help="generate and use random sample weights",  # 是否使用随机样本权重的标志
)
args = parser.parse_args()

n_leaf_nodes = args.n_leaf_nodes  # 叶节点数
n_trees = args.n_trees  # 树的数量
lr = args.learning_rate  # 学习率
max_bins = args.max_bins  # 最大箱数


def get_estimator_and_data():
    if args.problem == "classification":
        # 生成分类问题数据
        X, y = make_classification(
            args.n_samples_max * 2,  # 样本数
            n_features=args.n_features,  # 特征数
            n_classes=args.n_classes,  # 类别数
            n_clusters_per_class=1,
            n_informative=args.n_classes,
            random_state=0,
        )
        return X, y, HistGradientBoostingClassifier  # 返回数据和分类器类
    elif args.problem == "regression":
        # 生成回归问题数据
        X, y = make_regression(
            args.n_samples_max * 2, n_features=args.n_features, random_state=0
        )
        return X, y, HistGradientBoostingRegressor  # 返回数据和回归器类


X, y, Estimator = get_estimator_and_data()  # 获取数据和对应的估计器类

if args.missing_fraction:
    # 如果有缺失值要求，生成随机缺失值
    mask = np.random.binomial(1, args.missing_fraction, size=X.shape).astype(bool)
    X[mask] = np.nan

if args.random_sample_weights:
    # 如果需要使用随机样本权重，生成随机权重
    sample_weight = np.random.rand(len(X)) * 10
else:
    sample_weight = None

if sample_weight is not None:
    # 如果有样本权重，使用样本权重划分训练集和测试集
    (X_train_, X_test_, y_train_, y_test_, sample_weight_train_, _) = train_test_split(
        X, y, sample_weight, test_size=0.5, random_state=0
    )
else:
    # 没有样本权重，普通划分训练集和测试集
    X_train_, X_test_, y_train_, y_test_ = train_test_split(
        X, y, test_size=0.5, random_state=0
    )
    sample_weight_train_ = None


def one_run(n_samples):
    # 定义一个函数，用于运行一次训练和测试
    X_train = X_train_[:n_samples]  # 根据输入的样本数划分训练集
    # 从 X_test_ 数组中选取前 n_samples 个样本
    X_test = X_test_[:n_samples]
    # 从 y_train_ 数组中选取前 n_samples 个样本
    y_train = y_train_[:n_samples]
    # 从 y_test_ 数组中选取前 n_samples 个样本
    y_test = y_test_[:n_samples]
    # 如果 sample_weight 不为 None，则从 sample_weight_train_ 数组中选取前 n_samples 个样本的权重；否则设置为 None
    if sample_weight is not None:
        sample_weight_train = sample_weight_train_[:n_samples]
    else:
        sample_weight_train = None
    # 断言 X_train 的行数等于 n_samples
    assert X_train.shape[0] == n_samples
    # 断言 X_test 的行数等于 n_samples
    assert X_test.shape[0] == n_samples
    # 打印训练集和测试集的样本数
    print("Data size: %d samples train, %d samples test." % (n_samples, n_samples))
    # 打印提示信息，说明正在拟合一个 sklearn 模型
    print("Fitting a sklearn model...")
    # 记录当前时间
    tic = time()
    # 创建 Estimator 对象，设置参数包括学习率、迭代次数、最大分箱数、最大叶节点数等
    est = Estimator(
        learning_rate=lr,
        max_iter=n_trees,
        max_bins=max_bins,
        max_leaf_nodes=n_leaf_nodes,
        early_stopping=False,
        random_state=0,
        verbose=0,
    )
    # 获取损失函数参数
    loss = args.loss
    # 如果问题类型为分类问题且损失函数为"default"，则将损失函数设置为"log_loss"
    if args.problem == "classification":
        if loss == "default":
            loss = "log_loss"
    else:
        # 如果问题类型为回归问题且损失函数为"default"，则将损失函数设置为"squared_error"
        if loss == "default":
            loss = "squared_error"
    # 设置 Estimator 对象的损失函数参数
    est.set_params(loss=loss)
    # 使用 X_train 和 y_train 拟合 Estimator 对象，同时可以传入样本权重
    est.fit(X_train, y_train, sample_weight=sample_weight_train)
    # 计算 sklearn 模型拟合的时间
    sklearn_fit_duration = time() - tic
    # 记录当前时间
    tic = time()
    # 计算 sklearn 模型在测试集上的得分
    sklearn_score = est.score(X_test, y_test)
    # 计算计算得分耗时
    sklearn_score_duration = time() - tic
    # 打印 sklearn 模型的得分
    print("score: {:.4f}".format(sklearn_score))
    # 打印 sklearn 模型的拟合时间
    print("fit duration: {:.3f}s,".format(sklearn_fit_duration))
    # 打印 sklearn 模型的得分计算时间
    print("score duration: {:.3f}s,".format(sklearn_score_duration))

    # 初始化 LightGBM 模型得分、拟合时间和得分计算时间为 None
    lightgbm_score = None
    lightgbm_fit_duration = None
    lightgbm_score_duration = None
    # 如果用户指定使用 LightGBM 模型
    if args.lightgbm:
        # 打印提示信息，说明正在拟合一个 LightGBM 模型
        print("Fitting a LightGBM model...")
        # 获取与 sklearn Estimator 等效的 LightGBM Estimator 对象
        lightgbm_est = get_equivalent_estimator(
            est, lib="lightgbm", n_classes=args.n_classes
        )
        # 记录当前时间
        tic = time()
        # 使用 X_train 和 y_train 拟合 LightGBM Estimator 对象，同时可以传入样本权重
        lightgbm_est.fit(X_train, y_train, sample_weight=sample_weight_train)
        # 计算 LightGBM 模型的拟合时间
        lightgbm_fit_duration = time() - tic
        # 记录当前时间
        tic = time()
        # 计算 LightGBM 模型在测试集上的得分
        lightgbm_score = lightgbm_est.score(X_test, y_test)
        # 计算 LightGBM 模型得分的计算时间
        lightgbm_score_duration = time() - tic
        # 打印 LightGBM 模型的得分
        print("score: {:.4f}".format(lightgbm_score))
        # 打印 LightGBM 模型的拟合时间
        print("fit duration: {:.3f}s,".format(lightgbm_fit_duration))
        # 打印 LightGBM 模型的得分计算时间
        print("score duration: {:.3f}s,".format(lightgbm_score_duration))

    # 初始化 XGBoost 模型得分、拟合时间和得分计算时间为 None
    xgb_score = None
    xgb_fit_duration = None
    xgb_score_duration = None
    # 如果用户指定使用 XGBoost 模型
    if args.xgboost:
        # 打印提示信息，说明正在拟合一个 XGBoost 模型
        print("Fitting an XGBoost model...")
        # 获取与 sklearn Estimator 等效的 XGBoost Estimator 对象
        xgb_est = get_equivalent_estimator(est, lib="xgboost", n_classes=args.n_classes)
        # 记录当前时间
        tic = time()
        # 使用 X_train 和 y_train 拟合 XGBoost Estimator 对象，同时可以传入样本权重
        xgb_est.fit(X_train, y_train, sample_weight=sample_weight_train)
        # 计算 XGBoost 模型的拟合时间
        xgb_fit_duration = time() - tic
        # 记录当前时间
        tic = time()
        # 计算 XGBoost 模型在测试集上的得分
        xgb_score = xgb_est.score(X_test, y_test)
        # 计算 XGBoost 模型得分的计算时间
        xgb_score_duration = time() - tic
        # 打印 XGBoost 模型的得分
        print("score: {:.4f}".format(xgb_score))
        # 打印 XGBoost 模型的拟合时间
        print("fit duration: {:.3f}s,".format(xgb_fit_duration))
        # 打印 XGBoost 模型的得分计算时间
        print("score duration: {:.3f}s,".format(xgb_score_duration))

    # 初始化 CatBoost 模型得分、拟合时间和得分计算时间为 None
    cat_score = None
    cat_fit_duration = None
    cat_score_duration = None
    # 如果参数中指定了使用 CatBoost 模型
    if args.catboost:
        # 打印消息，开始拟合 CatBoost 模型
        print("Fitting a CatBoost model...")
        # 获取与给定模型等效的 CatBoost 估计器
        cat_est = get_equivalent_estimator(
            est, lib="catboost", n_classes=args.n_classes
        )

        # 开始计时
        tic = time()
        # 使用训练数据拟合 CatBoost 模型，支持样本权重
        cat_est.fit(X_train, y_train, sample_weight=sample_weight_train)
        # 计算拟合过程的持续时间
        cat_fit_duration = time() - tic

        # 再次开始计时
        tic = time()
        # 对测试数据计算 CatBoost 模型的得分
        cat_score = cat_est.score(X_test, y_test)
        # 计算得分过程的持续时间
        cat_score_duration = time() - tic

        # 打印 CatBoost 模型在测试数据上的得分
        print("score: {:.4f}".format(cat_score))
        # 打印 CatBoost 模型拟合过程的持续时间
        print("fit duration: {:.3f}s,".format(cat_fit_duration))
        # 打印 CatBoost 模型得分过程的持续时间
        print("score duration: {:.3f}s,".format(cat_score_duration))

    # 返回多个模型的评估结果和持续时间
    return (
        sklearn_score,
        sklearn_fit_duration,
        sklearn_score_duration,
        lightgbm_score,
        lightgbm_fit_duration,
        lightgbm_score_duration,
        xgb_score,
        xgb_fit_duration,
        xgb_score_duration,
        cat_score,
        cat_fit_duration,
        cat_score_duration,
    )
n_samples_list = [1000, 10000, 100000, 500000, 1000000, 5000000, 10000000]
# 筛选出不超过命令行参数 args.n_samples_max 的样本数量列表
n_samples_list = [
    n_samples for n_samples in n_samples_list if n_samples <= args.n_samples_max
]

sklearn_scores = []
sklearn_fit_durations = []
sklearn_score_durations = []
lightgbm_scores = []
lightgbm_fit_durations = []
lightgbm_score_durations = []
xgb_scores = []
xgb_fit_durations = []
xgb_score_durations = []
cat_scores = []
cat_fit_durations = []
cat_score_durations = []

# 遍历每个符合条件的样本数量
for n_samples in n_samples_list:
    # 调用函数 one_run() 获取各个模型的评分、拟合时间和评分时间
    (
        sklearn_score,
        sklearn_fit_duration,
        sklearn_score_duration,
        lightgbm_score,
        lightgbm_fit_duration,
        lightgbm_score_duration,
        xgb_score,
        xgb_fit_duration,
        xgb_score_duration,
        cat_score,
        cat_fit_duration,
        cat_score_duration,
    ) = one_run(n_samples)

    # 将各个模型的评分、拟合时间和评分时间存入对应的列表中
    for scores, score in (
        (sklearn_scores, sklearn_score),
        (sklearn_fit_durations, sklearn_fit_duration),
        (sklearn_score_durations, sklearn_score_duration),
        (lightgbm_scores, lightgbm_score),
        (lightgbm_fit_durations, lightgbm_fit_duration),
        (lightgbm_score_durations, lightgbm_score_duration),
        (xgb_scores, xgb_score),
        (xgb_fit_durations, xgb_fit_duration),
        (xgb_score_durations, xgb_score_duration),
        (cat_scores, cat_score),
        (cat_fit_durations, cat_fit_duration),
        (cat_score_durations, cat_score_duration),
    ):
        scores.append(score)

# 创建包含3个子图的图表
fig, axs = plt.subplots(3, sharex=True)

# 在第一个子图上绘制 sklearn 模型的评分数据
axs[0].plot(n_samples_list, sklearn_scores, label="sklearn")
# 在第二个子图上绘制 sklearn 模型的拟合时间数据
axs[1].plot(n_samples_list, sklearn_fit_durations, label="sklearn")
# 在第三个子图上绘制 sklearn 模型的评分时间数据
axs[2].plot(n_samples_list, sklearn_score_durations, label="sklearn")

# 如果命令行参数中包含 lightgbm 模型，绘制其评分、拟合时间和评分时间数据
if args.lightgbm:
    axs[0].plot(n_samples_list, lightgbm_scores, label="lightgbm")
    axs[1].plot(n_samples_list, lightgbm_fit_durations, label="lightgbm")
    axs[2].plot(n_samples_list, lightgbm_score_durations, label="lightgbm")

# 如果命令行参数中包含 xgboost 模型，绘制其评分、拟合时间和评分时间数据
if args.xgboost:
    axs[0].plot(n_samples_list, xgb_scores, label="XGBoost")
    axs[1].plot(n_samples_list, xgb_fit_durations, label="XGBoost")
    axs[2].plot(n_samples_list, xgb_score_durations, label="XGBoost")

# 如果命令行参数中包含 catboost 模型，绘制其评分、拟合时间和评分时间数据
if args.catboost:
    axs[0].plot(n_samples_list, cat_scores, label="CatBoost")
    axs[1].plot(n_samples_list, cat_fit_durations, label="CatBoost")
    axs[2].plot(n_samples_list, cat_score_durations, label="CatBoost")

# 设置每个子图的 x 轴为对数刻度，添加图例到最佳位置，设置 x 轴标签
for ax in axs:
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.set_xlabel("n_samples")

# 设置每个子图的标题
axs[0].set_title("scores")
axs[1].set_title("fit duration (s)")
axs[2].set_title("score duration (s)")

# 根据命令行参数设置主标题
title = args.problem
if args.problem == "classification":
    title += " n_classes = {}".format(args.n_classes)
fig.suptitle(title)

# 调整子图布局，展示图表
plt.tight_layout()
plt.show()
```