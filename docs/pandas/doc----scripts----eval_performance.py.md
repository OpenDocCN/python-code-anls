# `D:\src\scipysrc\pandas\doc\scripts\eval_performance.py`

```
from timeit import repeat as timeit  # 导入时间测量工具

import numpy as np  # 导入NumPy库
import seaborn as sns  # 导入Seaborn库

from pandas import DataFrame  # 从Pandas库中导入DataFrame类

setup_common = """from pandas import DataFrame
import numpy as np
df = DataFrame(np.random.randn(%d, 3), columns=list('abc'))
%s"""

setup_with = "s = 'a + b * (c ** 2 + b ** 2 - a) / (a * c) ** 3'"  # 设定eval方法的表达式

def bench_with(n, times=10, repeat=3, engine="numexpr"):
    return (
        np.array(
            timeit(
                f"df.eval(s, engine={engine!r})",  # 使用eval方法评估表达式s
                setup=setup_common % (n, setup_with),  # 设置评估环境
                repeat=repeat,
                number=times,
            )
        )
        / times
    )

setup_subset = "s = 'a <= b <= c ** 2 + b ** 2 - a and b > c'"  # 设定query方法的表达式

def bench_subset(n, times=20, repeat=3, engine="numexpr"):
    return (
        np.array(
            timeit(
                f"df.query(s, engine={engine!r})",  # 使用query方法查询符合条件的行
                setup=setup_common % (n, setup_subset),  # 设置查询环境
                repeat=repeat,
                number=times,
            )
        )
        / times
    )

def bench(mn=3, mx=7, num=100, engines=("python", "numexpr"), verbose=False):
    r = np.logspace(mn, mx, num=num).round().astype(int)  # 生成mn到mx之间的对数间隔数列

    ev = DataFrame(np.empty((num, len(engines))), columns=engines)  # 创建空的DataFrame用于存储eval方法的性能数据
    qu = ev.copy(deep=True)  # 复制DataFrame用于存储query方法的性能数据

    ev["size"] = qu["size"] = r  # 添加列'size'到两个DataFrame中

    for engine in engines:
        for i, n in enumerate(r):
            if verbose & (i % 10 == 0):  # 如果verbose为True并且i是10的倍数，打印当前进度信息
                print(f"engine: {engine!r}, i == {i:d}")
            ev_times = bench_with(n, times=1, repeat=1, engine=engine)  # 评估eval方法的性能
            ev.loc[i, engine] = np.mean(ev_times)  # 计算并存储eval方法的平均性能时间
            qu_times = bench_subset(n, times=1, repeat=1, engine=engine)  # 查询query方法的性能
            qu.loc[i, engine] = np.mean(qu_times)  # 计算并存储query方法的平均性能时间

    return ev, qu  # 返回eval和query方法的性能数据

def plot_perf(df, engines, title, filename=None) -> None:
    from matplotlib.pyplot import figure  # 导入matplotlib的figure模块

    sns.set()  # 设置Seaborn默认样式
    sns.set_palette("Set2")  # 设置Seaborn的调色板为Set2

    fig = figure(figsize=(4, 3), dpi=120)  # 创建图形对象
    ax = fig.add_subplot(111)  # 添加子图

    for engine in engines:
        ax.loglog(df["size"], df[engine], label=engine, lw=2)  # 绘制对数-对数图

    ax.set_xlabel("Number of Rows")  # 设置X轴标签
    ax.set_ylabel("Time (s)")  # 设置Y轴标签
    ax.set_title(title)  # 设置图表标题
    ax.legend(loc="best")  # 添加图例并设定位置为最佳
    ax.tick_params(top=False, right=False)  # 关闭顶部和右侧的刻度线

    fig.tight_layout()  # 调整图形布局使其紧凑

    if filename is not None:
        fig.savefig(filename)  # 如果指定了文件名，则保存图形到文件

if __name__ == "__main__":
    import os  # 导入os模块

    pandas_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    )  # 获取Pandas库的路径

    static_path = os.path.join(pandas_dir, "doc", "source", "_static")  # 组合静态文件夹的路径

    join = lambda p: os.path.join(static_path, p)  # 定义一个lambda函数用于拼接路径

    fn = join("eval-query-perf-data.h5")  # 设置文件名和路径

    engines = "python", "numexpr"  # 定义要比较的引擎类型

    ev, qu = bench(verbose=True)  # 运行性能评估函数，并获取结果

    plot_perf(ev, engines, "DataFrame.eval()", filename=join("eval-perf.png"))  # 绘制eval方法的性能图
    plot_perf(qu, engines, "DataFrame.query()", filename=join("query-perf.png"))  # 绘制query方法的性能图
```