# `D:\src\scipysrc\pandas\asv_bench\benchmarks\plotting.py`

```
# 导入必要的模块
import contextlib  # 提供上下文管理工具的模块
import importlib.machinery  # 提供导入机制的模块
import importlib.util  # 提供导入模块工具的模块
import os  # 提供与操作系统交互的功能
import pathlib  # 提供操作路径的对象
import sys  # 提供与解释器交互的功能
import tempfile  # 提供生成临时文件和目录的功能
from unittest import mock  # 提供用于测试的mock对象

import matplotlib  # 用于绘图的库
import numpy as np  # 提供数值计算的库

from pandas import (  # 导入 pandas 中的若干对象
    DataFrame,  # 用于操作和处理数据的主要数据结构
    DatetimeIndex,  # 用于处理日期时间索引的对象
    Series,  # 用于操作和处理一维数据的对象
    date_range,  # 用于生成日期范围的函数
)

try:
    from pandas.plotting import andrews_curves  # 尝试导入 pandas 中的 andrews_curves 函数
except ImportError:
    from pandas.tools.plotting import andrews_curves  # 如果导入失败，则尝试导入旧版的 andrews_curves 函数

from pandas.plotting._core import _get_plot_backend  # 导入 pandas 绘图模块中的 _get_plot_backend 函数

matplotlib.use("Agg")  # 设置 matplotlib 使用非交互式后端 "Agg"


class SeriesPlotting:
    params = [["line", "bar", "area", "barh", "hist", "kde", "pie"]]  # 参数化测试的参数列表
    param_names = ["kind"]  # 参数的名称列表

    def setup(self, kind):
        # 根据不同的图表类型选择不同的数据规模
        if kind in ["bar", "barh", "pie"]:
            n = 100
        elif kind in ["kde"]:
            n = 10000
        else:
            n = 1000000

        self.s = Series(np.random.randn(n))  # 生成具有随机数据的 Series 对象
        if kind in ["area", "pie"]:
            self.s = self.s.abs()  # 对于面积图和饼图，取绝对值处理数据

    def time_series_plot(self, kind):
        self.s.plot(kind=kind)  # 根据指定的类型绘制 Series 对象的图表


class FramePlotting:
    params = [
        ["line", "bar", "area", "barh", "hist", "kde", "pie", "scatter", "hexbin"]
    ]  # 参数化测试的参数列表
    param_names = ["kind"]  # 参数的名称列表

    def setup(self, kind):
        # 根据不同的图表类型选择不同的数据规模
        if kind in ["bar", "barh", "pie"]:
            n = 100
        elif kind in ["kde", "scatter", "hexbin"]:
            n = 10000
        else:
            n = 1000000

        self.x = Series(np.random.randn(n))  # 生成具有随机数据的 Series 对象
        self.y = Series(np.random.randn(n))  # 生成具有随机数据的 Series 对象
        if kind in ["area", "pie"]:
            self.x = self.x.abs()  # 对于面积图和饼图，取绝对值处理数据
            self.y = self.y.abs()  # 对于面积图和饼图，取绝对值处理数据
        self.df = DataFrame({"x": self.x, "y": self.y})  # 根据 x 和 y 构建 DataFrame 对象

    def time_frame_plot(self, kind):
        self.df.plot(x="x", y="y", kind=kind)  # 根据指定的类型绘制 DataFrame 对象的图表


class TimeseriesPlotting:
    def setup(self):
        N = 2000
        M = 5
        idx = date_range("1/1/1975", periods=N)  # 生成一个日期范围
        self.df = DataFrame(np.random.randn(N, M), index=idx)  # 生成具有随机数据和日期索引的 DataFrame 对象

        idx_irregular = DatetimeIndex(
            np.concatenate((idx.values[0:10], idx.values[12:]))
        )  # 创建不规则日期时间索引
        self.df2 = DataFrame(
            np.random.randn(len(idx_irregular), M), index=idx_irregular
        )  # 生成具有随机数据和不规则日期索引的 DataFrame 对象

    def time_plot_regular(self):
        self.df.plot()  # 绘制常规日期时间索引的 DataFrame 对象

    def time_plot_regular_compat(self):
        self.df.plot(x_compat=True)  # 兼容绘制常规日期时间索引的 DataFrame 对象

    def time_plot_irregular(self):
        self.df2.plot()  # 绘制不规则日期时间索引的 DataFrame 对象

    def time_plot_table(self):
        self.df.plot(table=True)  # 绘制带表格的 DataFrame 对象的图表


class Misc:
    def setup(self):
        N = 500
        M = 10
        self.df = DataFrame(np.random.randn(N, M))  # 生成具有随机数据的 DataFrame 对象
        self.df["Name"] = ["A"] * N  # 添加一个名为 "Name" 的列，所有行都是 "A"

    def time_plot_andrews_curves(self):
        andrews_curves(self.df, "Name")  # 绘制 Andrews 曲线


class BackendLoading:
    repeat = 1
    number = 1
    warmup_time = 0
    # 设置测试环境的准备工作
    def setup(self):
        # 创建一个名为 pandas_dummy_backend 的模块
        mod = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec("pandas_dummy_backend", None)
        )
        # 定义 mod 对象的 plot 方法为一个简单的 lambda 函数，始终返回 1
        mod.plot = lambda *args, **kwargs: 1

        # 使用 contextlib.ExitStack 确保退出时资源自动清理
        with contextlib.ExitStack() as stack:
            # 在 sys.modules 中模拟替换 "pandas_dummy_backend" 模块为新创建的 mod 对象
            stack.enter_context(
                mock.patch.dict(sys.modules, {"pandas_dummy_backend": mod})
            )

            # 创建一个临时目录 tmp_path
            tmp_path = pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory()))

            # 将临时目录路径添加到 sys.path 中，使其成为导入路径的首选
            sys.path.insert(0, os.fsdecode(tmp_path))
            # 注册一个回调函数，确保在退出时从 sys.path 中移除临时目录路径
            stack.callback(sys.path.remove, os.fsdecode(tmp_path))

            # 在临时目录下创建一个名为 "my_backend-0.0.0.dist-info" 的目录
            dist_info = tmp_path / "my_backend-0.0.0.dist-info"
            dist_info.mkdir()
            # 在 "my_backend-0.0.0.dist-info" 目录下创建 entry_points.txt 文件，并写入内容
            (dist_info / "entry_points.txt").write_bytes(
                b"[pandas_plotting_backends]\n"
                b"my_ep_backend = pandas_dummy_backend\n"
                b"my_ep_backend0 = pandas_dummy_backend\n"
                b"my_ep_backend1 = pandas_dummy_backend\n"
                b"my_ep_backend2 = pandas_dummy_backend\n"
                b"my_ep_backend3 = pandas_dummy_backend\n"
                b"my_ep_backend4 = pandas_dummy_backend\n"
                b"my_ep_backend5 = pandas_dummy_backend\n"
                b"my_ep_backend6 = pandas_dummy_backend\n"
                b"my_ep_backend7 = pandas_dummy_backend\n"
                b"my_ep_backend8 = pandas_dummy_backend\n"
                b"my_ep_backend9 = pandas_dummy_backend\n"
            )
            # 将 contextlib.ExitStack 的状态保存到 self.stack 中
            self.stack = stack.pop_all()

    # 清理测试环境的工作
    def teardown(self):
        # 关闭 self.stack 中所有资源的上下文管理器
        self.stack.close()

    # 测试获取绘图后端的性能（时间）
    def time_get_plot_backend(self):
        # 调用 _get_plot_backend 函数，查找第一个名为 "my_ep_backend" 的绘图后端
        _get_plot_backend("my_ep_backend")

    # 测试获取绘图后端的性能（时间），在失败后使用 importlib.import_module 方法
    def time_get_plot_backend_fallback(self):
        # 调用 _get_plot_backend 函数，遍历所有名为 "my_ep_backend[0-9]" 的绘图后端，然后回退到 importlib.import_module 方法
        _get_plot_backend("pandas_dummy_backend")
# 导入 pandas_vb_common 模块中的 setup 函数
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```