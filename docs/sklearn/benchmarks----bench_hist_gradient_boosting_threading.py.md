# `D:\src\scipysrc\scikit-learn\benchmarks\bench_hist_gradient_boosting_threading.py`

```
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import os  # 提供与操作系统交互的功能
from pprint import pprint  # 用于美化打印数据结构
from time import time  # 提供时间相关的功能

import numpy as np  # 数值计算库
from threadpoolctl import threadpool_limits  # 控制线程池的并发数量

import sklearn  # 机器学习库
from sklearn.datasets import make_classification, make_regression  # 生成分类和回归数据集的工具
from sklearn.ensemble import (  # 集成学习模型，包括梯度提升决策树
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.ensemble._hist_gradient_boosting.utils import get_equivalent_estimator  # 获取等效的估算器
from sklearn.model_selection import train_test_split  # 划分训练集和测试集的工具

# 创建命令行参数解析器
parser = argparse.ArgumentParser()

# 添加命令行参数
parser.add_argument("--n-leaf-nodes", type=int, default=31)  # 叶节点数目，默认为31
parser.add_argument("--n-trees", type=int, default=10)  # 树的数量，默认为10
parser.add_argument(
    "--lightgbm", action="store_true", default=False, help="also benchmark lightgbm"
)  # 是否评估 LightGBM，如果设置则为 True
parser.add_argument(
    "--xgboost", action="store_true", default=False, help="also benchmark xgboost"
)  # 是否评估 XGBoost，如果设置则为 True
parser.add_argument(
    "--catboost", action="store_true", default=False, help="also benchmark catboost"
)  # 是否评估 CatBoost，如果设置则为 True
parser.add_argument("--learning-rate", type=float, default=0.1)  # 学习率，默认为0.1
parser.add_argument(
    "--problem",
    type=str,
    default="classification",
    choices=["classification", "regression"],  # 问题类型，可选分类或回归
)  
parser.add_argument("--loss", type=str, default="default")  # 损失函数，默认为"default"
parser.add_argument("--missing-fraction", type=float, default=0)  # 缺失值的比例，默认为0
parser.add_argument("--n-classes", type=int, default=2)  # 分类问题的类别数，默认为2
parser.add_argument("--n-samples", type=int, default=int(1e6))  # 样本数，默认为1000000
parser.add_argument("--n-features", type=int, default=100)  # 特征数，默认为100
parser.add_argument("--max-bins", type=int, default=255)  # 最大箱数，默认为255

parser.add_argument("--print-params", action="store_true", default=False)  # 是否打印参数信息，如果设置则为 True
parser.add_argument(
    "--random-sample-weights",
    action="store_true",
    default=False,
    help="generate and use random sample weights",  # 是否生成和使用随机样本权重，如果设置则为 True
)
parser.add_argument("--plot", action="store_true", default=False, help="show a plot results")  # 是否显示绘图结果，如果设置则为 True
parser.add_argument(
    "--plot-filename", default=None, help="filename to save the figure to disk"
)  # 保存绘图结果的文件名

# 解析命令行参数
args = parser.parse_args()

# 从命令行参数中获取数据样本数目
n_samples = args.n_samples
# 从命令行参数中获取叶节点数目
n_leaf_nodes = args.n_leaf_nodes
# 从命令行参数中获取树的数量
n_trees = args.n_trees
# 从命令行参数中获取学习率
lr = args.learning_rate
# 从命令行参数中获取最大箱数
max_bins = args.max_bins

# 打印数据集大小
print("Data size: %d samples train, %d samples test." % (n_samples, n_samples))
# 打印特征数目
print(f"n_features: {args.n_features}")


def get_estimator_and_data():
    # 根据问题类型选择生成分类或回归数据集
    if args.problem == "classification":
        X, y = make_classification(
            args.n_samples * 2,
            n_features=args.n_features,
            n_classes=args.n_classes,
            n_clusters_per_class=1,
            n_informative=args.n_features // 2,
            random_state=0,
        )
        return X, y, HistGradientBoostingClassifier  # 返回生成的数据和分类器
    elif args.problem == "regression":
        X, y = make_regression(
            args.n_samples_max * 2, n_features=args.n_features, random_state=0
        )
        return X, y, HistGradientBoostingRegressor  # 返回生成的数据和回归器


# 调用函数获取数据和估算器
X, y, Estimator = get_estimator_and_data()

# 如果存在缺失值分数
if args.missing_fraction:
    # 生成随机的二项分布掩码
    mask = np.random.binomial(1, args.missing_fraction, size=X.shape).astype(bool)
    # 将缺失值分数应用到数据集上
    X[mask] = np.nan
# 如果命令行参数中包含 `--random_sample_weights`，则生成一个长度为 X 的随机样本权重数组，每个权重值乘以 10
if args.random_sample_weights:
    sample_weight = np.random.rand(len(X)) * 10
else:
    sample_weight = None

# 根据是否有样本权重，选择不同的拆分方式
if sample_weight is not None:
    # 使用样本权重进行数据拆分，保留训练集和测试集的样本权重
    (X_train_, X_test_, y_train_, y_test_, sample_weight_train_, _) = train_test_split(
        X, y, sample_weight, test_size=0.5, random_state=0
    )
else:
    # 没有样本权重时，普通拆分数据集
    (X_train_, X_test_, y_train_, y_test_) = train_test_split(
        X, y, test_size=0.5, random_state=0
    )
    sample_weight_train_ = None

# 创建一个 scikit-learn 的 Estimator 对象，并设置其各种参数
sklearn_est = Estimator(
    learning_rate=lr,
    max_iter=n_trees,
    max_bins=max_bins,
    max_leaf_nodes=n_leaf_nodes,
    early_stopping=False,
    random_state=0,
    verbose=0,
)
# 根据问题类型和损失函数设定 Estimator 对象的损失函数参数
loss = args.loss
if args.problem == "classification":
    if loss == "default":
        # 对于分类问题，默认损失函数为 "log_loss"
        loss = "log_loss"
else:
    # 对于回归问题，默认损失函数为 "squared_error"
    if loss == "default":
        loss = "squared_error"
sklearn_est.set_params(loss=loss)

# 如果命令行参数指定了打印参数，则打印 scikit-learn Estimator 对象的参数设置
if args.print_params:
    print("scikit-learn")
    pprint(sklearn_est.get_params())

    # 对于每个指定的机器学习库，如 "lightgbm", "xgboost", "catboost"，获取相应的等效估计器并打印其参数设置
    for libname in ["lightgbm", "xgboost", "catboost"]:
        if getattr(args, libname):
            print(libname)
            est = get_equivalent_estimator(
                sklearn_est, lib=libname, n_classes=args.n_classes
            )
            pprint(est.get_params())

# 定义一个函数 `one_run`，用于执行单次模型拟合和评分
def one_run(n_threads, n_samples):
    # 根据指定的样本数，从训练集和测试集中获取对应数量的样本数据
    X_train = X_train_[:n_samples]
    X_test = X_test_[:n_samples]
    y_train = y_train_[:n_samples]
    y_test = y_test_[:n_samples]
    # 根据是否有样本权重，选择是否获取对应数量的样本权重数据
    if sample_weight is not None:
        sample_weight_train = sample_weight_train_[:n_samples]
    else:
        sample_weight_train = None
    # 检查获取的样本数量与期望的样本数量一致
    assert X_train.shape[0] == n_samples
    assert X_test.shape[0] == n_samples
    print("Fitting a sklearn model...")
    tic = time()
    # 克隆一个 scikit-learn Estimator 对象，以确保每次运行独立于原始对象
    est = sklearn.base.clone(sklearn_est)

    # 使用线程池限制，执行模型拟合过程
    with threadpool_limits(n_threads, user_api="openmp"):
        est.fit(X_train, y_train, sample_weight=sample_weight_train)
        sklearn_fit_duration = time() - tic
        tic = time()
        # 计算模型在测试集上的得分
        sklearn_score = est.score(X_test, y_test)
        sklearn_score_duration = time() - tic
    print("score: {:.4f}".format(sklearn_score))
    print("fit duration: {:.3f}s,".format(sklearn_fit_duration))
    print("score duration: {:.3f}s,".format(sklearn_score_duration))

    # 初始化用于记录其他机器学习库评分和拟合时间的变量
    lightgbm_score = None
    lightgbm_fit_duration = None
    lightgbm_score_duration = None
    # 如果选择了LightGBM模型
    if args.lightgbm:
        # 打印提示信息
        print("Fitting a LightGBM model...")
        # 获取与给定模型相当的LightGBM估计器对象
        lightgbm_est = get_equivalent_estimator(
            est, lib="lightgbm", n_classes=args.n_classes
        )
        # 设置LightGBM估计器的线程数参数
        lightgbm_est.set_params(num_threads=n_threads)

        # 记录开始拟合模型的时间
        tic = time()
        # 使用训练数据拟合LightGBM模型，考虑样本权重
        lightgbm_est.fit(X_train, y_train, sample_weight=sample_weight_train)
        # 计算拟合模型的持续时间
        lightgbm_fit_duration = time() - tic
        # 记录开始评分的时间
        tic = time()
        # 对测试数据进行评分
        lightgbm_score = lightgbm_est.score(X_test, y_test)
        # 计算评分的持续时间
        lightgbm_score_duration = time() - tic
        # 打印LightGBM模型的评分结果
        print("score: {:.4f}".format(lightgbm_score))
        # 打印LightGBM模型拟合的持续时间
        print("fit duration: {:.3f}s,".format(lightgbm_fit_duration))
        # 打印LightGBM模型评分的持续时间
        print("score duration: {:.3f}s,".format(lightgbm_score_duration))

    # 初始化XGBoost模型的评分和持续时间变量
    xgb_score = None
    xgb_fit_duration = None
    xgb_score_duration = None
    # 如果选择了XGBoost模型
    if args.xgboost:
        # 打印提示信息
        print("Fitting an XGBoost model...")
        # 获取与给定模型相当的XGBoost估计器对象
        xgb_est = get_equivalent_estimator(est, lib="xgboost", n_classes=args.n_classes)
        # 设置XGBoost估计器的线程数参数
        xgb_est.set_params(nthread=n_threads)

        # 记录开始拟合模型的时间
        tic = time()
        # 使用训练数据拟合XGBoost模型，考虑样本权重
        xgb_est.fit(X_train, y_train, sample_weight=sample_weight_train)
        # 计算拟合模型的持续时间
        xgb_fit_duration = time() - tic
        # 记录开始评分的时间
        tic = time()
        # 对测试数据进行评分
        xgb_score = xgb_est.score(X_test, y_test)
        # 计算评分的持续时间
        xgb_score_duration = time() - tic
        # 打印XGBoost模型的评分结果
        print("score: {:.4f}".format(xgb_score))
        # 打印XGBoost模型拟合的持续时间
        print("fit duration: {:.3f}s,".format(xgb_fit_duration))
        # 打印XGBoost模型评分的持续时间
        print("score duration: {:.3f}s,".format(xgb_score_duration))

    # 初始化CatBoost模型的评分和持续时间变量
    cat_score = None
    cat_fit_duration = None
    cat_score_duration = None
    # 如果选择了CatBoost模型
    if args.catboost:
        # 打印提示信息
        print("Fitting a CatBoost model...")
        # 获取与给定模型相当的CatBoost估计器对象
        cat_est = get_equivalent_estimator(est, lib="catboost", n_classes=args.n_classes)
        # 设置CatBoost估计器的线程数参数
        cat_est.set_params(thread_count=n_threads)

        # 记录开始拟合模型的时间
        tic = time()
        # 使用训练数据拟合CatBoost模型，考虑样本权重
        cat_est.fit(X_train, y_train, sample_weight=sample_weight_train)
        # 计算拟合模型的持续时间
        cat_fit_duration = time() - tic
        # 记录开始评分的时间
        tic = time()
        # 对测试数据进行评分
        cat_score = cat_est.score(X_test, y_test)
        # 计算评分的持续时间
        cat_score_duration = time() - tic
        # 打印CatBoost模型的评分结果
        print("score: {:.4f}".format(cat_score))
        # 打印CatBoost模型拟合的持续时间
        print("fit duration: {:.3f}s,".format(cat_fit_duration))
        # 打印CatBoost模型评分的持续时间
        print("score duration: {:.3f}s,".format(cat_score_duration))

    # 返回模型评估指标和持续时间的元组
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
# 获取可用的最大线程数
max_threads = os.cpu_count()

# 生成线程数列表，使用2的指数增长直到小于最大线程数
n_threads_list = [2**i for i in range(8) if (2**i) < max_threads]
# 将最大线程数添加到线程数列表末尾
n_threads_list.append(max_threads)

# 初始化用于存储不同库的评分、拟合时间和评分时间的空列表
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

# 遍历每个线程数
for n_threads in n_threads_list:
    # 打印当前线程数
    print(f"n_threads: {n_threads}")

    # 调用函数one_run，获取各库的评分、拟合时间和评分时间
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
    ) = one_run(n_threads, n_samples)

    # 将各库的评分、拟合时间和评分时间添加到相应的列表中
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

# 如果命令行参数中包含绘图选项
if args.plot or args.plot_filename:
    # 导入 matplotlib 库
    import matplotlib
    import matplotlib.pyplot as plt

    # 创建一个包含2个子图的图形窗口，设置大小为12x12英寸
    fig, axs = plt.subplots(2, figsize=(12, 12))

    # 在第一个子图中绘制 sklearn 库的拟合时间曲线和评分时间曲线
    label = f"sklearn {sklearn.__version__}"
    axs[0].plot(n_threads_list, sklearn_fit_durations, label=label)
    axs[1].plot(n_threads_list, sklearn_score_durations, label=label)

    # 如果命令行参数中包含 lightgbm 选项，导入 lightgbm 库并绘制其曲线
    if args.lightgbm:
        import lightgbm

        label = f"LightGBM {lightgbm.__version__}"
        axs[0].plot(n_threads_list, lightgbm_fit_durations, label=label)
        axs[1].plot(n_threads_list, lightgbm_score_durations, label=label)

    # 如果命令行参数中包含 xgboost 选项，导入 xgboost 库并绘制其曲线
    if args.xgboost:
        import xgboost

        label = f"XGBoost {xgboost.__version__}"
        axs[0].plot(n_threads_list, xgb_fit_durations, label=label)
        axs[1].plot(n_threads_list, xgb_score_durations, label=label)

    # 如果命令行参数中包含 catboost 选项，导入 catboost 库并绘制其曲线
    if args.catboost:
        import catboost

        label = f"CatBoost {catboost.__version__}"
        axs[0].plot(n_threads_list, cat_fit_durations, label=label)
        axs[1].plot(n_threads_list, cat_score_durations, label=label)

    # 对每个子图进行设置
    for ax in axs:
        ax.set_xscale("log")  # 设置x轴为对数尺度
        ax.set_xlabel("n_threads")  # 设置x轴标签为线程数
        ax.set_ylabel("duration (s)")  # 设置y轴标签为持续时间（秒）
        ax.set_ylim(0, None)  # 设置y轴范围从0到最大值
        ax.set_xticks(n_threads_list)  # 设置x轴刻度为线程数列表
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())  # 设置x轴主要刻度格式
        ax.legend(loc="best")  # 设置图例位置为最佳位置

    # 设置子图标题
    axs[0].set_title("fit duration (s)")
    axs[1].set_title("score duration (s)")

    # 设置整体标题
    title = args.problem
    # 如果参数中指定问题类型为分类问题
    if args.problem == "classification":
        # 在标题中添加分类数信息
        title += " n_classes = {}".format(args.n_classes)
    
    # 设置图形的总标题
    fig.suptitle(title)
    
    # 调整子图的布局，确保各个子图之间的间距合适
    plt.tight_layout()
    
    # 如果指定了绘图文件名参数
    if args.plot_filename:
        # 将当前图形保存为指定文件名的图像文件
        plt.savefig(args.plot_filename)
    
    # 如果指定了绘图参数
    if args.plot:
        # 显示当前绘制的图形
        plt.show()
```