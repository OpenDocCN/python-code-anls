# `D:\src\scipysrc\matplotlib\tools\memleak.py`

```py
#!/usr/bin/env python

import gc  # 导入垃圾回收模块
from io import BytesIO  # 导入字节流模块
import tracemalloc  # 导入内存分配跟踪模块

try:
    import psutil  # 尝试导入psutil模块，用于系统进程和系统利用率信息
except ImportError as err:
    raise ImportError("This script requires psutil") from err  # 如果导入失败，抛出ImportError

import numpy as np  # 导入数值计算库numpy


def run_memleak_test(bench, iterations, report):
    tracemalloc.start()  # 启动内存分配跟踪

    starti = min(50, iterations // 2)  # 计算起始迭代数，取50和iterations // 2的最小值
    endi = iterations  # 迭代次数

    malloc_arr = np.empty(endi, dtype=np.int64)  # 创建一个空的numpy数组，用于存储内存分配量
    rss_arr = np.empty(endi, dtype=np.int64)  # 创建一个空的numpy数组，用于存储实际内存使用量
    rss_peaks = np.empty(endi, dtype=np.int64)  # 创建一个空的numpy数组，用于存储实际内存使用量的峰值
    nobjs_arr = np.empty(endi, dtype=np.int64)  # 创建一个空的numpy数组，用于存储Python对象数量
    garbage_arr = np.empty(endi, dtype=np.int64)  # 创建一个空的numpy数组，用于存储垃圾对象数量
    open_files_arr = np.empty(endi, dtype=np.int64)  # 创建一个空的numpy数组，用于存储打开文件的数量
    rss_peak = 0  # 初始实际内存使用量的峰值为0

    p = psutil.Process()  # 获取当前进程信息

    for i in range(endi):
        bench()  # 执行性能测试函数

        gc.collect()  # 手动触发垃圾回收

        rss = p.memory_info().rss  # 获取当前进程的实际内存使用量
        malloc, peak = tracemalloc.get_traced_memory()  # 获取当前内存分配量和分配峰值
        nobjs = len(gc.get_objects())  # 获取当前Python中的对象数量
        garbage = len(gc.garbage)  # 获取当前Python中的垃圾对象数量
        open_files = len(p.open_files())  # 获取当前进程打开的文件数量
        print(f"{i: 4d}: pymalloc {malloc: 10d}, rss {rss: 10d}, "
              f"nobjs {nobjs: 10d}, garbage {garbage: 4d}, "
              f"files: {open_files: 4d}")  # 打印每次迭代的内存和对象信息
        if i == starti:
            print(f'{" warmup done ":-^86s}')  # 在达到起始迭代数时打印热身结束提示
        malloc_arr[i] = malloc  # 将当前内存分配量存入数组
        rss_arr[i] = rss  # 将当前实际内存使用量存入数组
        if rss > rss_peak:
            rss_peak = rss  # 更新实际内存使用量的峰值
        rss_peaks[i] = rss_peak  # 将当前实际内存使用量的峰值存入数组
        nobjs_arr[i] = nobjs  # 将当前对象数量存入数组
        garbage_arr[i] = garbage  # 将当前垃圾对象数量存入数组
        open_files_arr[i] = open_files  # 将当前打开文件数量存入数组

    print('Average memory consumed per loop: {:1.4f} bytes\n'.format(
        np.sum(rss_peaks[starti+1:] - rss_peaks[starti:-1]) / (endi - starti)))  # 打印每次迭代后的平均内存消耗量

    from matplotlib import pyplot as plt  # 导入matplotlib用于绘图
    from matplotlib.ticker import EngFormatter  # 导入EngFormatter用于格式化刻度
    bytes_formatter = EngFormatter(unit='B')  # 创建字节单位的格式化对象
    fig, (ax1, ax2, ax3) = plt.subplots(3)  # 创建包含三个子图的图表对象
    for ax in (ax1, ax2, ax3):
        ax.axvline(starti, linestyle='--', color='k')  # 在每个子图中绘制起始迭代数的竖直虚线
    ax1b = ax1.twinx()  # 在ax1上创建第二个y轴
    ax1b.yaxis.set_major_formatter(bytes_formatter)  # 设置第二个y轴的格式化器为字节单位
    ax1.plot(malloc_arr, 'C0')  # 在ax1中绘制内存分配量的折线图
    ax1b.plot(rss_arr, 'C1', label='rss')  # 在ax1b中绘制实际内存使用量的折线图，并添加标签
    ax1b.plot(rss_peaks, 'C1', linestyle='--', label='rss max')  # 在ax1b中绘制实际内存使用量峰值的折线图，并添加标签
    ax1.set_ylabel('pymalloc', color='C0')  # 设置ax1的y轴标签颜色和内容
    ax1b.set_ylabel('rss', color='C1')  # 设置ax1b的y轴标签颜色和内容
    ax1b.legend()  # 在ax1b中添加图例

    ax2b = ax2.twinx()  # 在ax2上创建第二个y轴
    ax2.plot(nobjs_arr, 'C0')  # 在ax2中绘制对象数量的折线图
    ax2b.plot(garbage_arr, 'C1')  # 在ax2b中绘制垃圾对象数量的折线图
    ax2.set_ylabel('total objects', color='C0')  # 设置ax2的y轴标签颜色和内容
    ax2b.set_ylabel('garbage objects', color='C1')  # 设置ax2b的y轴标签颜色和内容

    ax3.plot(open_files_arr)  # 在ax3中绘制打开文件数量的折线图
    ax3.set_ylabel('open file handles')  # 设置ax3的y轴标签内容

    if not report.endswith('.pdf'):
        report = report + '.pdf'  # 如果报告文件名不以.pdf结尾，则加上.pdf后缀
    fig.tight_layout()  # 调整图表布局
    fig.savefig(report, format='pdf')  # 将图表保存为PDF文件


class MemleakTest:
    def __init__(self, empty):
        self.empty = empty  # 初始化函数，接受一个参数empty，并将其存储在实例变量self.empty中
    # 定义一个 __call__ 方法，使对象可被调用，这里引入 matplotlib.pyplot 库
    def __call__(self):
        # 导入 matplotlib.pyplot 库，并创建一个新的图形对象 fig
        import matplotlib.pyplot as plt
        fig = plt.figure(1)

        # 如果不是空的情况下执行以下操作
        if not self.empty:
            # 生成一个包含0到2之间值的数组 t1
            t1 = np.arange(0.0, 2.0, 0.01)
            # 计算 sin 函数在 t1 值上的值，并保存在 y1 中
            y1 = np.sin(2 * np.pi * t1)
            # 生成一个与 t1 同长度的随机数组，并保存在 y2 中
            y2 = np.random.rand(len(t1))
            # 生成一个 50x50 的随机数组 X

            # 将一个子图添加到 fig 中，位置是第1行第1列第1个位置
            ax = fig.add_subplot(221)
            ax.plot(t1, y1, '-')  # 在 ax 上绘制 t1 vs y1 的线条
            ax.plot(t1, y2, 's')  # 在 ax 上绘制 t1 vs y2 的散点图

            # 将一个子图添加到 fig 中，位置是第1行第1列第2个位置
            ax = fig.add_subplot(222)
            ax.imshow(X)  # 在 ax 上显示数组 X 的图像

            # 将一个子图添加到 fig 中，位置是第1行第2列第1个位置
            ax = fig.add_subplot(223)
            ax.scatter(np.random.rand(50), np.random.rand(50),
                       s=100 * np.random.rand(50), c=np.random.rand(50))
            # 在 ax 上绘制随机散点图，设置点的大小和颜色

            # 将一个子图添加到 fig 中，位置是第1行第2列第2个位置
            ax = fig.add_subplot(224)
            ax.pcolor(10 * np.random.rand(50, 50))
            # 在 ax 上绘制颜色图，填充颜色是由一个50x50的随机数组成

        # 将 fig 保存为字节流格式，分辨率为 75dpi
        fig.savefig(BytesIO(), dpi=75)
        # 刷新绘图区域的事件
        fig.canvas.flush_events()
        # 关闭编号为1的图形对象
        plt.close(1)
if __name__ == '__main__':
    # 如果作为主程序运行，则执行以下代码块

    import argparse
    # 导入 argparse 模块，用于解析命令行参数

    parser = argparse.ArgumentParser('Run memory leak tests')
    # 创建 ArgumentParser 对象，用于解析命令行参数，并设置描述信息

    parser.add_argument('backend', type=str, nargs=1,
                        help='backend to test')
    # 添加位置参数 'backend'，类型为字符串，数量为1，用于指定要测试的后端

    parser.add_argument('iterations', type=int, nargs=1,
                        help='number of iterations')
    # 添加位置参数 'iterations'，类型为整数，数量为1，表示测试迭代次数

    parser.add_argument('report', type=str, nargs=1,
                        help='filename to save report')
    # 添加位置参数 'report'，类型为字符串，数量为1，表示要保存报告的文件名

    parser.add_argument('--empty', action='store_true',
                        help="Don't plot any content, just test creating "
                        "and destroying figures")
    # 添加可选参数 '--empty'，设置为 True 表示不绘制任何内容，仅测试创建和销毁图形

    parser.add_argument('--interactive', action='store_true',
                        help="Turn on interactive mode to actually open "
                        "windows.  Only works with some GUI backends.")
    # 添加可选参数 '--interactive'，设置为 True 表示开启交互模式以实际打开窗口，仅适用于部分 GUI 后端

    args = parser.parse_args()
    # 解析命令行参数，并将结果存储在 args 变量中

    import matplotlib
    matplotlib.use(args.backend[0])
    # 使用指定的 matplotlib 后端，从参数 args 中获取后端名称并设置

    if args.interactive:
        import matplotlib.pyplot as plt
        plt.ion()
        # 如果设置了交互模式，则导入 matplotlib.pyplot 并打开交互模式

    run_memleak_test(
        MemleakTest(args.empty), args.iterations[0], args.report[0])
    # 调用 run_memleak_test 函数，传递 MemleakTest 对象（根据 --empty 参数创建）、迭代次数和报告文件名作为参数
```