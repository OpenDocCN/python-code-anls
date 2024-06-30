# `D:\src\scipysrc\scikit-learn\benchmarks\bench_multilabel_metrics.py`

```
# 指定脚本的解释器路径和描述信息
#!/usr/bin/env python
"""
A comparison of multilabel target formats and metrics over them
"""

# 导入必要的库和模块
import argparse                 # 导入命令行参数解析模块
import itertools                # 导入迭代工具模块
import sys                      # 导入系统相关模块
from functools import partial   # 导入偏函数模块
from timeit import timeit       # 导入计时模块

import matplotlib.pyplot as plt # 导入绘图模块
import numpy as np              # 导入数值计算模块
import scipy.sparse as sp       # 导入稀疏矩阵模块

from sklearn.datasets import make_multilabel_classification # 导入多标签分类数据生成器
from sklearn.metrics import (   # 导入评估指标
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_similarity_score,
)
from sklearn.utils._testing import ignore_warnings # 导入忽略警告函数

# 定义评估指标和相应的函数
METRICS = {
    "f1": partial(f1_score, average="micro"),          # 使用 micro 平均的 F1 分数
    "f1-by-sample": partial(f1_score, average="samples"), # 使用样本平均的 F1 分数
    "accuracy": accuracy_score,                        # 准确率评估指标
    "hamming": hamming_loss,                           # 汉明损失评估指标
    "jaccard": jaccard_similarity_score,               # Jaccard 相似度评估指标
}

# 定义不同多标签格式转换函数
FORMATS = {
    "sequences": lambda y: [list(np.flatnonzero(s)) for s in y], # 将密集表示转换为序列列表
    "dense": lambda y: y,                                       # 密集表示保持不变
    "csr": sp.csr_matrix,                                       # 转换为 CSR 稀疏矩阵
    "csc": sp.csc_matrix,                                       # 转换为 CSC 稀疏矩阵
}

# 定义忽略警告装饰函数的基准测试函数
@ignore_warnings
def benchmark(
    metrics=tuple(v for k, v in sorted(METRICS.items())),     # 待测试的评估指标函数列表
    formats=tuple(v for k, v in sorted(FORMATS.items())),     # 待测试的格式转换函数列表
    samples=1000,      # 生成的样本数量
    classes=4,         # 样本中的类别数量
    density=0.2,       # 正标签的密度
    n_times=5,         # 计时函数调用次数
):
    """Times metric calculations for a number of inputs

    Parameters
    ----------
    metrics : array-like of callables (1d or 0d)
        The metric functions to time.

    formats : array-like of callables (1d or 0d)
        These may transform a dense indicator matrix into multilabel
        representation.

    samples : array-like of ints (1d or 0d)
        The number of samples to generate as input.

    classes : array-like of ints (1d or 0d)
        The number of classes in the input.

    density : array-like of ints (1d or 0d)
        The density of positive labels in the input.

    n_times : int
        Time calling the metric n_times times.

    Returns
    -------
    array of floats shaped like (metrics, formats, samples, classes, density)
        Time in seconds.
    """
    # 将输入参数转换为至少一维数组
    metrics = np.atleast_1d(metrics)
    samples = np.atleast_1d(samples)
    classes = np.atleast_1d(classes)
    density = np.atleast_1d(density)
    formats = np.atleast_1d(formats)
    
    # 初始化结果数组
    out = np.zeros(
        (len(metrics), len(formats), len(samples), len(classes), len(density)),
        dtype=float,
    )
    
    # 使用 itertools 生成所有可能的输入组合
    it = itertools.product(samples, classes, density)
    for i, (s, c, d) in enumerate(it):
        # 生成多标签分类数据集
        _, y_true = make_multilabel_classification(
            n_samples=s, n_features=1, n_classes=c, n_labels=d * c, random_state=42
        )
        _, y_pred = make_multilabel_classification(
            n_samples=s, n_features=1, n_classes=c, n_labels=d * c, random_state=84
        )
        for j, f in enumerate(formats):
            # 对真实标签和预测标签进行格式转换
            f_true = f(y_true)
            f_pred = f(y_pred)
            for k, metric in enumerate(metrics):
                # 计算每种格式和评估指标的运行时间
                t = timeit(partial(metric, f_true, f_pred), number=n_times)
                out[k, j].flat[i] = t
    
    # 返回运行时间结果数组
    return out


# 定义打印结果的函数，按照指标和格式进行打印
def _tabulate(results, metrics, formats):
    """Prints results by metric and format
    # 计算格式化输出时每一列的宽度，至少为8个字符或者格式字符串最大长度加1
    column_width = max(max(len(k) for k in formats) + 1, 8)
    # 计算第一列（Metric列）的宽度，取所有metrics中字符串的最大长度
    first_width = max(len(k) for k in metrics)
    # 头部格式化字符串，Metric列左对齐，其余各列右对齐，列宽由column_width和first_width控制
    head_fmt = "{:<{fw}s}" + "{:>{cw}s}" * len(formats)
    # 行格式化字符串，Metric列左对齐，其余各列为浮点数保留三位小数，列宽由column_width和first_width控制
    row_fmt = "{:<{fw}s}" + "{:>{cw}.3f}" * len(formats)
    # 输出表头，格式化打印Metric和各formats对应的列标题
    print(head_fmt.format("Metric", *formats, cw=column_width, fw=first_width))
    # 遍历metrics和results中最后一个元素的最后一维度的数据，按行格式化打印
    for metric, row in zip(metrics, results[:, :, -1, -1, -1]):
        # 输出每行的Metric和对应的浮点数数据，格式化打印各列数据
        print(row_fmt.format(metric, *row, cw=column_width, fw=first_width))
# 导入所需的库和模块
import matplotlib.pyplot as plt
import argparse
import numpy as np

# 绘制多标签指标基准测试的图表
def _plot(
    results,
    metrics,
    formats,
    title,
    x_ticks,
    x_label,
    format_markers=("x", "|", "o", "+"),
    metric_colors=("c", "m", "y", "k", "g", "r", "b"),
):
    """
    Plot the results by metric, format and some other variable given by
    x_label
    """
    # 创建一个新的图形对象，设置标题
    fig = plt.figure("scikit-learn multilabel metrics benchmarks")
    plt.title(title)
    # 添加一个子图
    ax = fig.add_subplot(111)
    # 遍历指标和格式
    for i, metric in enumerate(metrics):
        for j, format in enumerate(formats):
            # 绘制折线图，x轴为x_ticks，y轴为results中的数据，设置标签、标记和颜色
            ax.plot(
                x_ticks,
                results[i, j].flat,
                label="{}, {}".format(metric, format),
                marker=format_markers[j],
                color=metric_colors[i % len(metric_colors)],
            )
    # 设置x轴标签和y轴标签
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time (s)")
    # 添加图例
    ax.legend()
    # 显示图形
    plt.show()


if __name__ == "__main__":
    # 创建参数解析器
    ap = argparse.ArgumentParser()
    # 添加位置参数metrics，用于指定基准测试的指标，默认为所有指标
    ap.add_argument(
        "metrics",
        nargs="*",
        default=sorted(METRICS),
        help="Specifies metrics to benchmark, defaults to all. Choices are: {}".format(
            sorted(METRICS)
        ),
    )
    # 添加可选参数--formats，用于指定多标签格式进行基准测试，默认为所有格式
    ap.add_argument(
        "--formats",
        nargs="+",
        choices=sorted(FORMATS),
        help="Specifies multilabel formats to benchmark (defaults to all).",
    )
    # 添加可选参数--samples，指定要生成的样本数，默认为1000
    ap.add_argument("--samples", type=int, default=1000, help="The number of samples to generate")
    # 添加可选参数--classes，指定类的数量，默认为10
    ap.add_argument("--classes", type=int, default=10, help="The number of classes")
    # 添加可选参数--density，指定每个样本的平均标签密度，默认为0.2
    ap.add_argument(
        "--density",
        type=float,
        default=0.2,
        help="The average density of labels per sample",
    )
    # 添加可选参数--plot，用于指定绘图的参数，可以是classes、density或samples
    ap.add_argument(
        "--plot",
        choices=["classes", "density", "samples"],
        default=None,
        help=(
            "Plot time with respect to this parameter varying up to the specified value"
        ),
    )
    # 添加可选参数--n-steps，指定每个指标要绘制的点数，默认为10
    ap.add_argument(
        "--n-steps", default=10, type=int, help="Plot this many points for each metric"
    )
    # 添加可选参数--n-times，指定执行基准测试的次数，默认为5
    ap.add_argument(
        "--n-times", default=5, type=int, help="Time performance over n_times trials"
    )
    # 解析命令行参数
    args = ap.parse_args()

    # 根据绘图参数plot对其进行处理
    if args.plot is not None:
        max_val = getattr(args, args.plot)
        if args.plot in ("classes", "samples"):
            min_val = 2
        else:
            min_val = 0
        steps = np.linspace(min_val, max_val, num=args.n_steps + 1)[1:]
        if args.plot in ("classes", "samples"):
            steps = np.unique(np.round(steps).astype(int))
        setattr(args, args.plot, steps)

    # 如果metrics参数为None，则默认使用所有METRICS中的指标
    if args.metrics is None:
        args.metrics = sorted(METRICS)
    # 如果formats参数为None，则默认使用所有FORMATS中的格式
    if args.formats is None:
        args.formats = sorted(FORMATS)

    # 进行基准测试，得到结果
    results = benchmark(
        [METRICS[k] for k in args.metrics],
        [FORMATS[k] for k in args.formats],
        args.samples,
        args.classes,
        args.density,
        args.n_times,
    )

    # 将结果进行表格化输出
    _tabulate(results, args.metrics, args.formats)
    # 如果命令行参数 args.plot 不是 None，则执行以下代码块
    if args.plot is not None:
        # 在标准错误流中打印显示信息
        print("Displaying plot", file=sys.stderr)
        
        # 根据命令行参数组合标题字符串，描述多标签指标的内容
        title = "Multilabel metrics with %s" % ", ".join(
            "{0}={1}".format(field, getattr(args, field))
            for field in ["samples", "classes", "density"]
            if args.plot != field
        )
        
        # 调用 _plot 函数，显示图形化结果
        _plot(results, args.metrics, args.formats, title, steps, args.plot)
```