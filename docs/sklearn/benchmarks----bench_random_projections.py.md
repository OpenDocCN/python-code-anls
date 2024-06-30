# `D:\src\scipysrc\scikit-learn\benchmarks\bench_random_projections.py`

```
"""
===========================
Random projection benchmark
===========================

Benchmarks for random projections.

"""

# 导入必要的库和模块
import collections  # 导入collections模块，用于特定数据结构操作
import gc  # 导入gc模块，用于垃圾回收
import optparse  # 导入optparse模块，用于命令行选项解析
import sys  # 导入sys模块，用于与Python解释器交互
from datetime import datetime  # 从datetime模块中导入datetime类

import numpy as np  # 导入NumPy库，用于数值计算
import scipy.sparse as sp  # 导入SciPy库的稀疏矩阵模块

from sklearn import clone  # 从sklearn库中导入clone函数，用于克隆估算器对象
from sklearn.random_projection import (  # 从sklearn库的随机投影模块中导入以下类
    GaussianRandomProjection,
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)


def type_auto_or_float(val):
    if val == "auto":
        return "auto"
    else:
        return float(val)


def type_auto_or_int(val):
    if val == "auto":
        return "auto"
    else:
        return int(val)


def compute_time(t_start, delta):
    mu_second = 0.0 + 10**6  # number of microseconds in a second

    return delta.seconds + delta.microseconds / mu_second


def bench_scikit_transformer(X, transformer):
    gc.collect()  # 手动触发垃圾回收

    clf = clone(transformer)  # 克隆传入的转换器对象

    # 记录开始时间
    t_start = datetime.now()
    clf.fit(X)  # 使用克隆的转换器对象拟合数据集
    delta = datetime.now() - t_start  # 计算拟合所用时间
    # 记录停止时间
    time_to_fit = compute_time(t_start, delta)  # 计算拟合时间，包括秒和微秒

    # 记录开始时间
    t_start = datetime.now()
    clf.transform(X)  # 使用克隆的转换器对象对数据集进行转换
    delta = datetime.now() - t_start  # 计算转换所用时间
    # 记录停止时间
    time_to_transform = compute_time(t_start, delta)  # 计算转换时间，包括秒和微秒

    return time_to_fit, time_to_transform  # 返回拟合和转换的时间


# 生成具有均匀分布非零条目和高斯分布值的随机数据
def make_sparse_random_data(n_samples, n_features, n_nonzeros, random_state=None):
    rng = np.random.RandomState(random_state)
    data_coo = sp.coo_matrix(
        (
            rng.randn(n_nonzeros),  # 随机生成n_nonzeros个高斯分布值
            (
                rng.randint(n_samples, size=n_nonzeros),  # 随机生成n_nonzeros个样本索引
                rng.randint(n_features, size=n_nonzeros),  # 随机生成n_nonzeros个特征索引
            ),
        ),
        shape=(n_samples, n_features),  # 稀疏矩阵的形状为(n_samples, n_features)
    )
    return data_coo.toarray(), data_coo.tocsr()  # 返回密集矩阵和压缩稀疏行矩阵的形式


def print_row(clf_type, time_fit, time_transform):
    print(
        "%s | %s | %s"
        % (
            clf_type.ljust(30),  # 左对齐的分类器类型字符串，长度为30
            ("%.4fs" % time_fit).center(12),  # 格式化拟合时间为字符串，长度为12，包括四位小数
            ("%.4fs" % time_transform).center(12),  # 格式化转换时间为字符串，长度为12，包括四位小数
        )
    )


if __name__ == "__main__":
    ###########################################################################
    # Option parser
    ###########################################################################

    op = optparse.OptionParser()  # 创建选项解析器对象

    op.add_option(
        "--n-times",  # 增加--n-times选项
        dest="n_times",  # 将选项值存储在n_times变量中
        default=5,  # 默认值为5
        type=int,  # 类型为整数
        help="Benchmark results are average over n_times experiments",  # 帮助文本
    )

    op.add_option(
        "--n-features",  # 增加--n-features选项
        dest="n_features",  # 将选项值存储在n_features变量中
        default=10**4,  # 默认值为10000
        type=int,  # 类型为整数
        help="Number of features in the benchmarks",  # 帮助文本
    )

    op.add_option(
        "--n-components",  # 增加--n-components选项
        dest="n_components",  # 将选项值存储在n_components变量中
        default="auto",  # 默认值为"auto"
        help="Size of the random subspace. ('auto' or int > 0)",  # 帮助文本
    )
    op.add_option(
        "--ratio-nonzeros",
        dest="ratio_nonzeros",
        default=10**-3,
        type=float,
        help="Number of features in the benchmarks",
    )

# 添加命令行选项 "--ratio-nonzeros"，用于指定稀疏性比例，默认为 0.001。


    op.add_option(
        "--n-samples",
        dest="n_samples",
        default=500,
        type=int,
        help="Number of samples in the benchmarks",
    )

# 添加命令行选项 "--n-samples"，用于指定样本数量，默认为 500。


    op.add_option(
        "--random-seed",
        dest="random_seed",
        default=13,
        type=int,
        help="Seed used by the random number generators.",
    )

# 添加命令行选项 "--random-seed"，用于指定随机种子，默认为 13。


    op.add_option(
        "--density",
        dest="density",
        default=1 / 3,
        help=(
            "Density used by the sparse random projection. ('auto' or float (0.0, 1.0]"
        ),
    )

# 添加命令行选项 "--density"，用于指定稀疏随机投影的密度，可以是 'auto' 或 0 到 1 之间的浮点数，默认为 1/3。


    op.add_option(
        "--eps",
        dest="eps",
        default=0.5,
        type=float,
        help="See the documentation of the underlying transformers.",
    )

# 添加命令行选项 "--eps"，用于指定转换器的参数 eps，默认为 0.5。


    op.add_option(
        "--transformers",
        dest="selected_transformers",
        default="GaussianRandomProjection,SparseRandomProjection",
        type=str,
        help=(
            "Comma-separated list of transformer to benchmark. "
            "Default: %default. Available: "
            "GaussianRandomProjection,SparseRandomProjection"
        ),
    )

# 添加命令行选项 "--transformers"，用于指定要进行基准测试的转换器列表，默认为 "GaussianRandomProjection,SparseRandomProjection"。


    op.add_option(
        "--dense",
        dest="dense",
        default=False,
        action="store_true",
        help="Set input space as a dense matrix.",
    )

# 添加命令行选项 "--dense"，用于将输入空间设置为密集矩阵，默认为 False。


    (opts, args) = op.parse_args()

# 解析命令行参数，将结果存储在 opts 和 args 中。


    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

# 检查是否有未预期的命令行参数，如果有则报错并退出脚本。


    opts.n_components = type_auto_or_int(opts.n_components)
    opts.density = type_auto_or_float(opts.density)

# 调用函数 type_auto_or_int 和 type_auto_or_float 分别处理 opts.n_components 和 opts.density，使其类型正确。


    selected_transformers = opts.selected_transformers.split(",")

# 将 opts.selected_transformers 按逗号分隔转换为列表 selected_transformers。


    ###########################################################################
    # Generate dataset
    ###########################################################################

# 生成数据集的开始标记。


    n_nonzeros = int(opts.ratio_nonzeros * opts.n_features)

# 计算稀疏特性的非零元素数量，即 opts.ratio_nonzeros 乘以 opts.n_features。


    print("Dataset statistics")
    print("===========================")
    print("n_samples \t= %s" % opts.n_samples)
    print("n_features \t= %s" % opts.n_features)

# 打印数据集统计信息，包括样本数量和特征数量。


    if opts.n_components == "auto":
        print(
            "n_components \t= %s (auto)"
            % johnson_lindenstrauss_min_dim(n_samples=opts.n_samples, eps=opts.eps)
        )
    else:
        print("n_components \t= %s" % opts.n_components)

# 根据 opts.n_components 的值，打印降维组件数量或调用函数计算最小维数。


    print("n_elements \t= %s" % (opts.n_features * opts.n_samples))
    print("n_nonzeros \t= %s per feature" % n_nonzeros)
    print("ratio_nonzeros \t= %s" % opts.ratio_nonzeros)
    print("")

# 打印数据集的总元素数量、每个特征的非零元素数量和 opts.ratio_nonzeros 的值。


    ###########################################################################
    # Set transformer input
    ###########################################################################

# 设置转换器输入的开始标记。


    transformers = {}

# 初始化一个空的字典 transformers 用于存储转换器。


    ###########################################################################
    # Set GaussianRandomProjection input

# 设置高斯随机投影输入的开始标记。
    # 定义高斯随机投影的参数字典，包括成分数量和随机种子
    gaussian_matrix_params = {
        "n_components": opts.n_components,
        "random_state": opts.random_seed,
    }
    # 将高斯随机投影添加到变换器字典中，使用预定义的参数
    transformers["GaussianRandomProjection"] = GaussianRandomProjection(
        **gaussian_matrix_params
    )

    ###########################################################################
    # 设置稀疏随机投影的输入参数
    sparse_matrix_params = {
        "n_components": opts.n_components,
        "random_state": opts.random_seed,
        "density": opts.density,
        "eps": opts.eps,
    }

    # 将稀疏随机投影添加到变换器字典中，使用预定义的参数
    transformers["SparseRandomProjection"] = SparseRandomProjection(
        **sparse_matrix_params
    )

    ###########################################################################
    # 执行基准测试
    ###########################################################################
    # 创建用于存储拟合时间和变换时间的默认字典
    time_fit = collections.defaultdict(list)
    time_transform = collections.defaultdict(list)

    # 打印基准测试的信息
    print("Benchmarks")
    print("===========================")
    print("Generate dataset benchmarks... ", end="")
    # 生成稀疏随机数据集，根据选择的密度和稠密性生成相应的数据
    X_dense, X_sparse = make_sparse_random_data(
        opts.n_samples, opts.n_features, n_nonzeros, random_state=opts.random_seed
    )
    # 根据参数选择稠密或稀疏的数据集
    X = X_dense if opts.dense else X_sparse
    print("done")

    # 对每个选定的变换器执行基准测试
    for name in selected_transformers:
        print("Perform benchmarks for %s..." % name)

        # 多次迭代，对每个变换器进行拟合和变换的基准测试
        for iteration in range(opts.n_times):
            print("\titer %s..." % iteration, end="")
            # 测量拟合和变换的时间，并记录到字典中
            time_to_fit, time_to_transform = bench_scikit_transformer(
                X_dense, transformers[name]
            )
            time_fit[name].append(time_to_fit)
            time_transform[name].append(time_to_transform)
            print("done")

    print("")

    ###########################################################################
    # 打印结果
    ###########################################################################
    # 打印脚本参数
    print("Script arguments")
    print("===========================")
    arguments = vars(opts)
    # 打印参数和其值
    print(
        "%s \t | %s "
        % (
            "Arguments".ljust(16),
            "Value".center(12),
        )
    )
    print(25 * "-" + ("|" + "-" * 14) * 1)
    for key, value in arguments.items():
        print("%s \t | %s " % (str(key).ljust(16), str(value).strip().center(12)))
    print("")

    # 打印变换器的性能表现
    print("Transformer performance:")
    print("===========================")
    print("Results are averaged over %s repetition(s)." % opts.n_times)
    print("")
    # 打印表头：变换器、拟合时间、变换时间
    print(
        "%s | %s | %s"
        % ("Transformer".ljust(30), "fit".center(12), "transform".center(12))
    )
    print(31 * "-" + ("|" + "-" * 14) * 2)

    # 对选定的变换器按名称排序，并打印每个变换器的性能数据
    for name in sorted(selected_transformers):
        print_row(name, np.mean(time_fit[name]), np.mean(time_transform[name]))

    print("")
    print("")
```