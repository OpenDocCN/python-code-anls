# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_determinism.py`

```py
"""
Test output reproducibility.
"""

# 导入必要的库
import os  # 导入操作系统接口模块
import sys  # 导入系统特定的参数和函数
import pytest  # 导入pytest测试框架

import matplotlib as mpl  # 导入matplotlib库并命名为mpl
import matplotlib.testing.compare  # 导入matplotlib的测试比较模块
from matplotlib import pyplot as plt  # 从matplotlib库中导入pyplot子模块并命名为plt
from matplotlib.testing._markers import needs_ghostscript, needs_usetex  # 导入测试标记
from matplotlib.testing import subprocess_run_for_testing  # 导入测试子进程运行模块


def _save_figure(objects='mhi', fmt="pdf", usetex=False):
    # 设置matplotlib使用的输出格式和TeX使用的选项
    mpl.use(fmt)
    mpl.rcParams.update({'svg.hashsalt': 'asdf', 'text.usetex': usetex})

    # 创建一个新的图形对象
    fig = plt.figure()

    if 'm' in objects:
        # 如果包含'm'，则在第1行第6列中添加子图，并绘制具有不同标记的图形
        ax1 = fig.add_subplot(1, 6, 1)
        x = range(10)
        ax1.plot(x, [1] * 10, marker='D')
        ax1.plot(x, [2] * 10, marker='x')
        ax1.plot(x, [3] * 10, marker='^')
        ax1.plot(x, [4] * 10, marker='H')
        ax1.plot(x, [5] * 10, marker='v')

    if 'h' in objects:
        # 如果包含'h'，则在第1行第6列中添加子图，并绘制具有不同填充图案的条形图
        ax2 = fig.add_subplot(1, 6, 2)
        bars = (ax2.bar(range(1, 5), range(1, 5)) +
                ax2.bar(range(1, 5), [6] * 4, bottom=range(1, 5)))
        ax2.set_xticks([1.5, 2.5, 3.5, 4.5])

        patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)

    if 'i' in objects:
        # 如果包含'i'，则在第1行第6列中添加子图，并显示不同的图像
        A = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
        fig.add_subplot(1, 6, 3).imshow(A, interpolation='nearest')
        A = [[1, 3, 2], [1, 2, 3], [3, 1, 2]]
        fig.add_subplot(1, 6, 4).imshow(A, interpolation='bilinear')
        A = [[2, 3, 1], [1, 2, 3], [2, 1, 3]]
        fig.add_subplot(1, 6, 5).imshow(A, interpolation='bicubic')

    # 在第1行第6列中添加子图，并绘制简单的线图
    x = range(5)
    ax = fig.add_subplot(1, 6, 6)
    ax.plot(x, x)
    ax.set_title('A string $1+2+\\sigma$')
    ax.set_xlabel('A string $1+2+\\sigma$')
    ax.set_ylabel('A string $1+2+\\sigma$')

    # 获取标准输出，并将图形保存为指定格式
    stdout = getattr(sys.stdout, 'buffer', sys.stdout)
    fig.savefig(stdout, format=fmt)


@pytest.mark.parametrize(
    "objects, fmt, usetex", [
        ("", "pdf", False),  # 测试空对象输出为PDF格式，不使用TeX
        ("m", "pdf", False),  # 测试标记对象输出为PDF格式，不使用TeX
        ("h", "pdf", False),  # 测试填充图案对象输出为PDF格式，不使用TeX
        ("i", "pdf", False),  # 测试图像对象输出为PDF格式，不使用TeX
        ("mhi", "pdf", False),  # 测试标记、填充图案和图像对象输出为PDF格式，不使用TeX
        ("mhi", "ps", False),  # 测试标记、填充图案和图像对象输出为PS格式，不使用TeX
        pytest.param(
            "mhi", "ps", True, marks=[needs_usetex, needs_ghostscript]),  # 使用TeX需求和Ghostscript的PS格式测试
        ("mhi", "svg", False),  # 测试标记、填充图案和图像对象输出为SVG格式，不使用TeX
        pytest.param("mhi", "svg", True, marks=needs_usetex),  # 使用TeX需求的SVG格式测试
    ]
)
def test_determinism_check(objects, fmt, usetex):
    """
    Output three times the same graphs and checks that the outputs are exactly
    the same.

    Parameters
    ----------
    objects : str
        Objects to be included in the test document: 'm' for markers, 'h' for
        hatch patterns, 'i' for images.
    fmt : {"pdf", "ps", "svg"}
        Output format.
    """
    # 生成三次子进程，每次调用 matplotlib 的测试函数来保存图形
    plots = [
        subprocess_run_for_testing(
            [sys.executable, "-R", "-c",
             # 导入 matplotlib.tests.test_determinism 模块中的 _save_figure 函数，并调用之
             f"from matplotlib.tests.test_determinism import _save_figure;"
             # 调用 _save_figure 函数保存图形，传入 objects、fmt、usetex 作为参数
             f"_save_figure({objects!r}, {fmt!r}, {usetex})"],
            # 设置环境变量，包括 SOURCE_DATE_EPOCH 和 MPLBACKEND
            env={**os.environ, "SOURCE_DATE_EPOCH": "946684800", "MPLBACKEND": "Agg"},
            # 不捕获输出为文本，捕获子进程输出，检查返回状态码为真
            text=False, capture_output=True, check=True).stdout
        # 重复进行三次子进程调用，结果存入 plots 列表中
        for _ in range(3)
    ]
    # 对除第一个元素外的每个图形结果进行检查
    for p in plots[1:]:
        # 如果格式为 "ps" 且使用了 LaTeX 渲染
        if fmt == "ps" and usetex:
            # 检查当前图形结果是否与第一个图形结果相同，如果不同则跳过测试
            if p != plots[0]:
                pytest.skip("failed, maybe due to ghostscript timestamps")
        else:
            # 对于其他格式或不使用 LaTeX 渲染，断言当前图形结果与第一个图形结果相同
            assert p == plots[0]
@pytest.mark.parametrize(
    "fmt, string", [
        ("pdf", b"/CreationDate (D:20000101000000Z)"),  # 设置参数 fmt 为 'pdf'，string 为 PDF 文件的创建时间戳
        # 在不使用 text.usetex 的情况下，不测试 SOURCE_DATE_EPOCH 的支持，
        # 因为生成的时间戳来自 ghostscript: %%CreationDate: D:20000101000000Z00\'00\',
        # 而这可能会因为 ghostscript 的不同版本而改变。
        ("ps", b"%%CreationDate: Sat Jan 01 00:00:00 2000"),  # 设置参数 fmt 为 'ps'，string 为 PostScript 文件的创建时间戳
    ]
)
def test_determinism_source_date_epoch(fmt, string):
    """
    测试 SOURCE_DATE_EPOCH 的支持。使用环境变量 SOURCE_DATE_EPOCH 设置为 2000-01-01 00:00 UTC，
    检查生成的文档是否包含对应于此日期的时间戳（作为参数给出）。

    Parameters
    ----------
    fmt : {"pdf", "ps", "svg"}
        输出格式。
    string : bytes
        用于 2000-01-01 00:00 UTC 的时间戳字符串。
    """
    buf = subprocess_run_for_testing(
        [sys.executable, "-R", "-c",
         f"from matplotlib.tests.test_determinism import _save_figure; "
         f"_save_figure('', {fmt!r})"],
        env={**os.environ, "SOURCE_DATE_EPOCH": "946684800",
             "MPLBACKEND": "Agg"}, capture_output=True, text=False, check=True).stdout
    assert string in buf  # 断言生成的文档内容包含预期的时间戳字符串
```