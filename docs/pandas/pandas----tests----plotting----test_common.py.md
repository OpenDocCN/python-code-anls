# `D:\src\scipysrc\pandas\pandas\tests\plotting\test_common.py`

```
import pytest  # 导入 pytest 库

from pandas import DataFrame  # 从 pandas 库导入 DataFrame 类
from pandas.tests.plotting.common import (  # 从 pandas 的测试模块中导入多个函数
    _check_plot_works,
    _check_ticks_props,
    _gen_two_subplots,
)

plt = pytest.importorskip("matplotlib.pyplot")  # 导入 matplotlib.pyplot 模块，如果导入失败则跳过测试


class TestCommon:
    def test__check_ticks_props(self):
        # 测试用例：GH 34768
        df = DataFrame({"b": [0, 1, 0], "a": [1, 2, 3]})
        ax = _check_plot_works(df.plot, rot=30)  # 调用 _check_plot_works 函数测试绘图
        ax.yaxis.set_tick_params(rotation=30)  # 设置 y 轴刻度的旋转角度
        msg = "expected 0.00000 but got "
        with pytest.raises(AssertionError, match=msg):
            _check_ticks_props(ax, xrot=0)  # 断言 x 轴刻度旋转角度为 0 度
        with pytest.raises(AssertionError, match=msg):
            _check_ticks_props(ax, xlabelsize=0)  # 断言 x 轴标签大小为 0
        with pytest.raises(AssertionError, match=msg):
            _check_ticks_props(ax, yrot=0)  # 断言 y 轴刻度旋转角度为 0 度
        with pytest.raises(AssertionError, match=msg):
            _check_ticks_props(ax, ylabelsize=0)  # 断言 y 轴标签大小为 0

    def test__gen_two_subplots_with_ax(self):
        fig = plt.gcf()  # 获取当前的图形对象
        gen = _gen_two_subplots(f=lambda **kwargs: None, fig=fig, ax="test")  # 生成两个子图的生成器对象
        # 第一次生成，不应添加子图，因为传入了已有的轴对象
        next(gen)
        assert fig.get_axes() == []  # 断言当前图形中没有子图
        # 第二次生成，应该添加一个子图，并检查其布局
        next(gen)
        axes = fig.get_axes()  # 获取当前图形中的所有轴对象
        assert len(axes) == 1  # 断言轴对象数量为 1
        subplot_geometry = list(axes[0].get_subplotspec().get_geometry()[:-1])  # 获取子图的几何布局信息
        subplot_geometry[-1] += 1  # 调整子图的布局信息
        assert subplot_geometry == [2, 1, 2]  # 断言调整后的子图布局信息符合预期

    def test_colorbar_layout(self):
        fig = plt.figure()  # 创建一个新的图形对象

        axes = fig.subplot_mosaic(
            """
            AB
            CC
            """
        )  # 在图形中创建指定布局的子图

        x = [1, 2, 3]
        y = [1, 2, 3]

        cs0 = axes["A"].scatter(x, y)  # 在子图 A 中绘制散点图
        axes["B"].scatter(x, y)  # 在子图 B 中绘制散点图

        fig.colorbar(cs0, ax=[axes["A"], axes["B"]], location="right")  # 在指定的轴对象上添加颜色条
        DataFrame(x).plot(ax=axes["C"])  # 在子图 C 中绘制 DataFrame 的图表
```