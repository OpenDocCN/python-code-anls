# `D:\src\scipysrc\scikit-learn\benchmarks\bench_sample_without_replacement.py`

```
"""
Benchmarks for sampling without replacement of integer.

"""

import gc  # 引入垃圾回收模块，用于在需要时手动触发垃圾回收
import operator  # 引入操作符模块，用于执行基本的操作符功能
import optparse  # 引入选项解析模块，用于处理命令行参数
import random  # 引入随机数生成模块，用于生成随机数
import sys  # 引入系统模块，用于访问与 Python 解释器交互的变量和函数
from datetime import datetime  # 从 datetime 模块中导入 datetime 类

import matplotlib.pyplot as plt  # 引入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 引入 NumPy 模块，用于支持多维数组和矩阵运算

from sklearn.utils.random import sample_without_replacement  # 从 scikit-learn 中导入无重复抽样函数


def compute_time(t_start, delta):
    mu_second = 0.0 + 10**6  # number of microseconds in a second
    # 计算从开始时间到结束时间的总秒数（包括微秒）
    return delta.seconds + delta.microseconds / mu_second


def bench_sample(sampling, n_population, n_samples):
    gc.collect()  # 手动触发垃圾回收，清理内存中的无用对象
    # 获取当前时间作为开始时间
    t_start = datetime.now()
    # 调用传入的抽样函数进行抽样操作
    sampling(n_population, n_samples)
    # 计算从开始时间到抽样结束的时间差
    delta = datetime.now() - t_start
    # 计算总消耗时间（秒）
    time = compute_time(t_start, delta)
    return time


if __name__ == "__main__":
    ###########################################################################
    # Option parser
    ###########################################################################
    # 创建选项解析器对象
    op = optparse.OptionParser()
    # 添加命令行参数 --n-times，用于指定进行实验的次数，默认为5次
    op.add_option(
        "--n-times",
        dest="n_times",
        default=5,
        type=int,
        help="Benchmark results are average over n_times experiments",
    )

    # 添加命令行参数 --n-population，用于指定总体大小，默认为100,000
    op.add_option(
        "--n-population",
        dest="n_population",
        default=100000,
        type=int,
        help="Size of the population to sample from.",
    )

    # 添加命令行参数 --n-step，用于指定步长间隔，默认为5
    op.add_option(
        "--n-step",
        dest="n_steps",
        default=5,
        type=int,
        help="Number of step interval between 0 and n_population.",
    )

    default_algorithms = (
        "custom-tracking-selection,custom-auto,"
        "custom-reservoir-sampling,custom-pool,"
        "python-core-sample,numpy-permutation"
    )

    # 添加命令行参数 --algorithm，用于指定要进行基准测试的抽样算法，默认为一组算法名称
    op.add_option(
        "--algorithm",
        dest="selected_algorithm",
        default=default_algorithms,
        type=str,
        help=(
            "Comma-separated list of transformer to benchmark. "
            "Default: %default. \nAvailable: %default"
        ),
    )

    # 解析命令行参数
    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    # 将选定的算法名称拆分为列表，并检查是否在默认算法列表中
    selected_algorithm = opts.selected_algorithm.split(",")
    for key in selected_algorithm:
        if key not in default_algorithms.split(","):
            raise ValueError(
                'Unknown sampling algorithm "%s" not in (%s).'
                % (key, default_algorithms)
            )

    ###########################################################################
    # List sampling algorithm
    ###########################################################################
    # 定义一个空字典，用于存储抽样算法的名称与实际函数的映射关系
    sampling_algorithm = {}

    ###########################################################################
    # Set Python core input
    # 将一个基于随机抽样的函数添加到sampling_algorithm字典中，使用lambda函数定义
    sampling_algorithm["python-core-sample"] = (
        lambda n_population, n_sample: random.sample(range(n_population), n_sample)
    )

    ###########################################################################
    # 设置自定义的自动选择方法
    sampling_algorithm["custom-auto"] = (
        lambda n_population, n_samples, random_state=None: sample_without_replacement(
            n_population, n_samples, method="auto", random_state=random_state
        )
    )

    ###########################################################################
    # 设置基于自定义跟踪的选择方法
    sampling_algorithm["custom-tracking-selection"] = (
        lambda n_population, n_samples, random_state=None: sample_without_replacement(
            n_population,
            n_samples,
            method="tracking_selection",
            random_state=random_state,
        )
    )

    ###########################################################################
    # 设置基于自定义水库抽样的方法
    sampling_algorithm["custom-reservoir-sampling"] = (
        lambda n_population, n_samples, random_state=None: sample_without_replacement(
            n_population,
            n_samples,
            method="reservoir_sampling",
            random_state=random_state,
        )
    )

    ###########################################################################
    # 设置基于自定义池的抽样方法
    sampling_algorithm["custom-pool"] = (
        lambda n_population, n_samples, random_state=None: sample_without_replacement(
            n_population, n_samples, method="pool", random_state=random_state
        )
    )

    ###########################################################################
    # 使用NumPy的排列方法进行抽样
    sampling_algorithm["numpy-permutation"] = (
        lambda n_population, n_sample: np.random.permutation(n_population)[:n_sample]
    )

    ###########################################################################
    # 从sampling_algorithm中移除未指定的算法，根据selected_algorithm进行筛选
    sampling_algorithm = {
        key: value
        for key, value in sampling_algorithm.items()
        if key in selected_algorithm
    }

    ###########################################################################
    # 执行基准测试
    ###########################################################################
    # 初始化时间字典
    time = {}
    # 创建包含指定间隔的样本数的NumPy数组
    n_samples = np.linspace(start=0, stop=opts.n_population, num=opts.n_steps).astype(
        int
    )
    # 计算样本数与总体大小的比率
    ratio = n_samples / opts.n_population

    # 打印基准测试标题
    print("Benchmarks")
    print("===========================")
    # 遍历排序后的抽样算法名称列表，并逐个打印执行基准测试的信息
    for name in sorted(sampling_algorithm):
        print("Perform benchmarks for %s..." % name, end="")
        # 初始化一个二维数组用于存储基准测试的时间数据，大小为 (opts.n_steps, opts.n_times)
        time[name] = np.zeros(shape=(opts.n_steps, opts.n_times))

        # 循环执行基准测试的步骤数
        for step in range(opts.n_steps):
            # 循环执行每个步骤中的基准测试次数
            for it in range(opts.n_times):
                # 调用 bench_sample 函数进行基准测试，并将结果存储在 time[name][step, it] 中
                time[name][step, it] = bench_sample(
                    sampling_algorithm[name], opts.n_population, n_samples[step]
                )

        # 打印完成信息
        print("done")

    # 打印平均结果信息
    print("Averaging results...", end="")
    # 对每个抽样算法的时间数据取平均值，替换原始数据
    for name in sampling_algorithm:
        time[name] = np.mean(time[name], axis=1)
    print("done\n")

    # 打印结果
    ###########################################################################
    print("Script arguments")
    print("===========================")
    # 将命令行参数转换成字典并逐行打印
    arguments = vars(opts)
    print(
        "%s \t | %s "
        % (
            "Arguments".ljust(16),
            "Value".center(12),
        )
    )
    print(25 * "-" + ("|" + "-" * 14) * 1)
    for key, value in arguments.items():
        # 格式化打印每个参数及其对应的值
        print("%s \t | %s " % (str(key).ljust(16), str(value).strip().center(12)))
    print("")

    print("Sampling algorithm performance:")
    print("===============================")
    # 打印抽样算法的性能信息，指出结果是基于 opts.n_times 次重复计算的平均值
    print("Results are averaged over %s repetition(s)." % opts.n_times)
    print("")

    # 创建 matplotlib 图形对象并设置标题
    fig = plt.figure("scikit-learn sample w/o replacement benchmark results")
    fig.suptitle("n_population = %s, n_times = %s" % (opts.n_population, opts.n_times))
    # 添加一个子图
    ax = fig.add_subplot(111)
    # 针对每个抽样算法，绘制 ratio 和 time[name] 的关系图
    for name in sampling_algorithm:
        ax.plot(ratio, time[name], label=name)

    # 设置 X 轴和 Y 轴的标签
    ax.set_xlabel("ratio of n_sample / n_population")
    ax.set_ylabel("Time (s)")
    # 添加图例
    ax.legend()

    # 对图例标签进行排序并重新设置图例
    handles, labels = ax.get_legend_handles_labels()
    hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
    handles2, labels2 = zip(*hl)
    ax.legend(handles2, labels2, loc=0)

    # 显示绘制的图形
    plt.show()
```