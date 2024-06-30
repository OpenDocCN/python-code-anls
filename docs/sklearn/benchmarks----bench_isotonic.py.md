# `D:\src\scipysrc\scikit-learn\benchmarks\bench_isotonic.py`

```
"""
Isotonic regression性能基准测试。

我们生成一个大小为10^n的合成数据集，其中n在[min, max]范围内变化，
并检查运行等距回归（isotonic regression）所需的时间。

然后将时间输出到标准输出，或者在matplotlib上以对数-对数刻度进行可视化。

这样可以可视化和理解算法随问题规模的扩展情况。
"""

import argparse  # 导入命令行参数解析模块
import gc  # 导入垃圾回收模块
from timeit import default_timer  # 导入计时器模块

import matplotlib.pyplot as plt  # 导入绘图库matplotlib
import numpy as np  # 导入数值计算库numpy
from scipy.special import expit  # 导入expit函数

from sklearn.isotonic import isotonic_regression  # 导入sklearn的等距回归函数


def generate_perturbed_logarithm_dataset(size):
    """
    生成一个扰动对数数据集。

    参数:
    size -- 数据集大小

    返回:
    生成的数据集
    """
    return np.random.randint(-50, 50, size=size) + 50.0 * np.log(1 + np.arange(size))


def generate_logistic_dataset(size):
    """
    生成一个逻辑数据集。

    参数:
    size -- 数据集大小

    返回:
    生成的数据集
    """
    X = np.sort(np.random.normal(size=size))
    return np.random.random(size=size) < expit(X)


def generate_pathological_dataset(size):
    """
    生成一个病态数据集。

    参数:
    size -- 数据集大小

    返回:
    生成的数据集
    """
    # 在原始实现上触发O(n^2)复杂度。
    return np.r_[
        np.arange(size), np.arange(-(size - 1), size), np.arange(-(size - 1), 1)
    ]


DATASET_GENERATORS = {
    "perturbed_logarithm": generate_perturbed_logarithm_dataset,
    "logistic": generate_logistic_dataset,
    "pathological": generate_pathological_dataset,
}


def bench_isotonic_regression(Y):
    """
    运行一次等距回归（isotonic regression）在输入数据上，
    并报告总共花费的时间（秒）。
    
    参数:
    Y -- 输入数据

    返回:
    运行的时间
    """
    gc.collect()  # 手动触发垃圾回收

    tstart = default_timer()  # 记录开始时间
    isotonic_regression(Y)  # 运行等距回归
    return default_timer() - tstart  # 返回运行时间


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isotonic Regression benchmark tool")
    parser.add_argument("--seed", type=int, help="RNG seed")
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Number of iterations to average timings over for each problem size",
    )
    parser.add_argument(
        "--log_min_problem_size",
        type=int,
        required=True,
        help="Base 10 logarithm of the minimum problem size",
    )
    parser.add_argument(
        "--log_max_problem_size",
        type=int,
        required=True,
        help="Base 10 logarithm of the maximum problem size",
    )
    parser.add_argument(
        "--show_plot", action="store_true", help="Plot timing output with matplotlib"
    )
    parser.add_argument("--dataset", choices=DATASET_GENERATORS.keys(), required=True)

    args = parser.parse_args()  # 解析命令行参数

    np.random.seed(args.seed)  # 设定随机数种子

    timings = []  # 存储时间数据的列表
    for exponent in range(args.log_min_problem_size, args.log_max_problem_size):
        n = 10**exponent  # 计算当前问题规模
        Y = DATASET_GENERATORS[args.dataset](n)  # 生成数据集Y
        time_per_iteration = [
            bench_isotonic_regression(Y) for i in range(args.iterations)
        ]  # 对当前数据集运行多次等距回归，并记录时间
        timing = (n, np.mean(time_per_iteration))  # 计算平均时间
        timings.append(timing)  # 将结果加入列表

        # 如果不需要绘图，将时间数据打印到标准输出
        if not args.show_plot:
            print(n, np.mean(time_per_iteration))
    # 如果参数中包含 show_plot 标志，显示绘图结果
    if args.show_plot:
        # 使用 plt.plot 绘制图表，*zip(*timings)将timings列表转置并展开为两个列表
        plt.plot(*zip(*timings))
        # 设置图表标题
        plt.title("Average time taken running isotonic regression")
        # 设置 x 轴标签
        plt.xlabel("Number of observations")
        # 设置 y 轴标签
        plt.ylabel("Time (s)")
        # 自动调整坐标轴范围
        plt.axis("tight")
        # 设置坐标轴为对数刻度
        plt.loglog()
        # 显示图表
        plt.show()
```